"""Evaluate the released checkpoints with the notebook's metric and timing code."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from train_oxford_pet import EvalShard, PetCache, SmallUNet, evaluate, latency, validate_scores


def load_paths(root):
    """Use the same file extensions and filename ordering as the notebook."""
    images = {
        path.stem: path
        for path in sorted((root / "images").iterdir())
        if path.is_file() and not path.name.startswith(".")
        and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    trimaps = {
        path.stem: path
        for path in sorted((root / "annotations" / "trimaps").iterdir())
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() == ".png"
    }
    return images, trimaps


def prepare_test_cache(root, metadata, image_size, cache_path):
    images, trimaps = load_paths(root)
    counts = metadata["counts"]
    test_ids = metadata["ids"][counts["train"] + counts["val"]:]
    if len(test_ids) != counts["test"] or len(set(test_ids)) != len(test_ids):
        raise ValueError("The test count or IDs in dataset.json are inconsistent.")
    missing = [name for name in test_ids if name not in images or name not in trimaps]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test pairs, starting with {missing[:5]}")

    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint8, shape=(len(test_ids), image_size, image_size, 4)
    )
    size = (image_size, image_size)
    for index, name in enumerate(test_ids):
        with Image.open(images[name]) as image:
            cache[index, :, :, :3] = np.asarray(
                image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            )
        with Image.open(trimaps[name]) as trimap:
            if trimap.mode not in {"L", "P"}:
                raise ValueError(f"Expected an indexed trimap at {trimaps[name]}")
            raw = np.asarray(trimap.resize(size, Image.Resampling.NEAREST), dtype=np.uint8)
        cache[index, :, :, 3] = np.where(raw == 1, 1, np.where(raw == 2, 0, 255))
    cache.flush()
    return len(test_ids)


def evaluate_checkpoint(path, dataset, config, device, measure_latency):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["image_size"] != config["image_size"]:
        raise ValueError(f"Checkpoint input size differs from config.json for {path.name}")
    model = SmallUNet(checkpoint["dilation"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, memory_format=torch.channels_last)

    world, batch_size = config["processes"], config["batch_per_gpu"]
    matrix = torch.zeros((2, 2), dtype=torch.int64, device=device)
    for rank in range(world):
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=EvalShard(dataset, rank, world),
            pin_memory=device.type == "cuda",
        )
        shard, _ = evaluate(model, loader, device)
        matrix += shard
    (background, pet), mean = validate_scores(matrix)
    result = {
        "background_IoU": background,
        "pet_IoU": pet,
        "mIoU": mean,
        "test_confusion_matrix": matrix.cpu().tolist(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": checkpoint["epoch"],
    }
    if measure_latency:
        result["latency"] = latency(model, dataset, device, batch_size * world, config["smoke_test"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Extracted dataset folder containing images/ and annotations/trimaps/")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--latency", action="store_true", help="Also benchmark model forwards")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch runtime.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.set_num_threads(2)

    config = json.loads((args.results_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.results_dir / "dataset.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="oxford-pet-eval-") as scratch:
        pixels = Path(scratch) / "pixels.npy"
        count = prepare_test_cache(args.data_root, metadata, config["image_size"], pixels)
        dataset = PetCache(pixels, 0, count, config["seed"])
        try:
            results = {
                name: evaluate_checkpoint(
                    args.results_dir / f"{name}_best.pt", dataset, config, device, args.latency
                )
                for name in ("baseline", "dilated")
            }
        finally:
            del dataset
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
