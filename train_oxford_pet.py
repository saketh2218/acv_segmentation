from datetime import timedelta
from pathlib import Path
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=dilation,dilation=dilation,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class SmallUNet(nn.Module):
    def __init__(self, bottleneck_dilation=1):
        super().__init__()
        self.enc1=ConvBlock(3,32); self.pool1=nn.MaxPool2d(2)
        self.enc2=ConvBlock(32,64); self.pool2=nn.MaxPool2d(2)
        self.enc3=ConvBlock(64,128); self.pool3=nn.MaxPool2d(2)
        self.bottleneck=ConvBlock(128,256,dilation=bottleneck_dilation)
        self.up3=nn.ConvTranspose2d(256,128,2,2); self.dec3=ConvBlock(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2); self.dec2=ConvBlock(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2); self.dec1=ConvBlock(64,32)
        self.head=nn.Conv2d(32,2,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool1(a)); c=self.enc3(self.pool2(b))
        z=self.bottleneck(self.pool3(c))
        z=self.dec3(torch.cat([self.up3(z),c],1))
        z=self.dec2(torch.cat([self.up2(z),b],1))
        z=self.dec1(torch.cat([self.up1(z),a],1))
        return self.head(z)


class PetCache(Dataset):
    def __init__(self, pixels, offset, count, seed, augment=False):
        self.path = str(pixels)
        self.offset, self.count, self.seed = offset, count, seed
        self.augment = augment
        self.epoch = 0
        self.pixels = None  # Open lazily inside each DataLoader worker.

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        if self.pixels is None:
            self.pixels = np.load(self.path, mmap_mode="r")
        sample = self.pixels[self.offset + i]
        if self.augment and random.Random(self.seed + self.epoch * self.count + i).random() < 0.5:
            sample = sample[:, ::-1, :]
        return np.array(sample, copy=True), i


class EvalShard(Sampler):
    """Disjoint shards without padding; each validation/test item contributes exactly once."""
    def __init__(self, dataset, rank, world):
        self.indices = range(rank, len(dataset), world)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def split_batch(samples, device):
    samples = samples.to(device, non_blocking=True)
    image = samples[..., :3].permute(0, 3, 1, 2).to(dtype=torch.float32,
                                                   memory_format=torch.channels_last)
    return image.mul_(2.0 / 255).sub_(1), samples[..., 3].long().contiguous()


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def sum_across_ranks(tensor, async_op=False):
    if dist.is_initialized():
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)


def validate_scores(cm):
    cm = cm.detach().cpu().numpy().astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm)
    iou = [float(cm[i, i] / denom[i]) if denom[i] else None for i in range(2)]
    defined = [x for x in iou if x is not None]
    if not defined:
        raise ValueError("No labeled foreground or background pixels found during evaluation.")
    return iou, float(np.mean(defined))


def report_progress(title, step, total):
    event = {"title": title, "step": step, "total": total, "unit": "batches"}
    print("NOTEBOOK_PROGRESS " + json.dumps(event), flush=True)


def track_batches(batches, label, phase):
    if label is None:
        yield from batches
        return

    title = f"{label} | {phase}"
    total = len(batches)
    report_progress(title, 0, total)
    last_update = time.monotonic()
    for step, batch in enumerate(batches, start=1):
        yield batch
        now = time.monotonic()
        if step == 1 or step == total or now - last_update >= 1:
            report_progress(title, step, total)
            last_update = now


@torch.inference_mode()
def evaluate(model, batches, device, collect_errors=False, limit=6, progress_label=None,
             progress_phase="validation (rank 0 shard)"):
    # Invoke the raw model: the shards may have different numbers of batches, including zero.
    # Broadcast batch-normalization buffers once so every rank evaluates the same model.
    model.eval()
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    cm = torch.zeros((2, 2), dtype=torch.int64, device=device)
    per_image = []
    for samples, ids in track_batches(batches, progress_label, progress_phase):
        x, y = split_batch(samples, device)
        with autocast(device):
            pred = model(x).argmax(1)
        bg, pet = y == 0, y == 1
        pred_pet = pred == 1
        dims = (1, 2) if collect_errors else None
        if collect_errors:
            # Compute errors per test image during the same batched forward passes.
            a = (bg & ~pred_pet).sum(dims)
            b = (bg & pred_pet).sum(dims)
            c = (pet & ~pred_pet).sum(dims)
            d = (pet & pred_pet).sum(dims)
            rows = torch.stack((a, b, c, d), dim=1)
            cm += rows.sum(0).reshape(2, 2)
            per_image.append((ids, rows))
        else:
            cm += torch.stack(((bg & ~pred_pet).sum(), (bg & pred_pet).sum(),
                               (pet & ~pred_pet).sum(), (pet & pred_pet).sum())).reshape(2, 2)
    sum_across_ranks(cm)
    picks = []
    if collect_errors and per_image:
        ids = torch.cat([x[0] for x in per_image]).tolist()
        counts = torch.cat([x[1] for x in per_image]).cpu().tolist()
        errors = [((b + c) / (a + b + c + d), i)
                  for i, (a, b, c, d) in zip(ids, counts) if a + b + c + d]
        representative = [("representative", err, i) for err, i in errors if err < 0.15][:limit]
        highest = [("highest", err, i) for err, i in sorted(errors, reverse=True)[:limit]]
        picks = representative + highest
    return cm, picks


def train_epoch(model, batches, optimizer, scaler, device, world, progress_label):
    model.train()
    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    total_valid = torch.zeros((), dtype=torch.int64, device=device)
    for samples, _ in track_batches(batches, progress_label, "training"):
        x, y = split_batch(samples, device)
        local_count = (y != 255).sum()
        global_count = local_count.clone()
        count_work = sum_across_ranks(global_count, async_op=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            nll = F.cross_entropy(model(x), y, ignore_index=255, reduction="sum")
        if count_work is not None:
            count_work.wait()
        # DDP averages gradients. Multiplying by world size gives the true
        # mean over all labeled pixels, even when trimap coverage differs by rank.
        loss = nll * (world / global_count.clamp_min(1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_nll += nll.detach()
        total_valid += local_count
    sum_across_ranks(total_nll)
    sum_across_ranks(total_valid)
    return float((total_nll / total_valid.clamp_min(1)).item())


@torch.inference_mode()
def latency(model, dataset, device, batch_size, smoke):
    # Reuse a fixed batch in device memory. Measure model forwards averaged per image.
    n = min(batch_size, len(dataset))
    samples = torch.from_numpy(np.stack([dataset[i][0] for i in range(n)]))
    x, _ = split_batch(samples, device)
    warmup, repeats = (2, 3) if smoke else (10, 50)
    model.eval()
    with autocast(device):
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {"ms_per_image": (time.perf_counter() - t0) * 1000 / (repeats * n),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": "FP16 autocast" if device.type == "cuda" else "FP32",
            "batch_size": n, "warmup": warmup, "repeats": repeats}


@torch.inference_mode()
def save_examples(model, dataset, picks, device, output_path):
    representative = sorted((item for item in picks if item[0] == "representative"),
                            key=lambda item: item[2])[:6]
    highest = sorted((item for item in picks if item[0] == "highest"),
                     key=lambda item: (item[1], item[2]), reverse=True)[:6]
    items = representative + highest
    if not items:
        return
    samples = torch.from_numpy(np.stack([dataset[i][0] for _, _, i in items]))
    x, y = split_batch(samples, device)
    with autocast(device):
        pred = torch.cat([model(part).argmax(1).to(torch.uint8).cpu() for part in x.split(4)])
    np.savez_compressed(output_path, images=samples[..., :3].numpy(), masks=y.cpu().numpy().astype(np.uint8),
                        predictions=pred.numpy(), groups=[group for group, _, _ in items],
                        errors=[err for _, err, _ in items], indices=[i for _, _, i in items])


def run(run_dir, pixels, cfg, device, rank, world):
    is_main = rank == 0
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    meta = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    counts = meta["counts"]
    batch_size = cfg["batch_per_gpu"]
    train = PetCache(pixels, 0, counts["train"], seed, augment=True)
    val = PetCache(pixels, counts["train"], counts["val"], seed)
    test = PetCache(pixels, counts["train"] + counts["val"], counts["test"], seed)
    train_sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=seed)
    loader_opts = {"batch_size": batch_size, "pin_memory": device.type == "cuda",
                   "num_workers": cfg["workers_per_gpu"]}
    if loader_opts["num_workers"]:
        # Start children without inheriting a live CUDA/NCCL context.
        loader_opts.update(multiprocessing_context="spawn", prefetch_factor=2)
    train_loader = DataLoader(train, sampler=train_sampler, **loader_opts)
    val_loader = DataLoader(val, sampler=EvalShard(val, rank, world), **loader_opts)
    test_loader = DataLoader(test, sampler=EvalShard(test, rank, world), **loader_opts)
    if is_main:
        print(f"Global batch={batch_size * world}, steps/epoch={len(train_loader)}, training examples="
              f"{len(train):,}. The sampler repeats {train_sampler.total_size - len(train)} "
              "example(s) per epoch to give ranks equal sample counts.", flush=True)

    initial_state = SmallUNet(1).state_dict()
    models = {}
    for key, dilation in (("baseline", 1), ("dilated", 2)):
        raw = SmallUNet(dilation)
        raw.load_state_dict(initial_state)
        raw.to(device=device, memory_format=torch.channels_last)
        parallel = (DDP(raw, device_ids=[device.index] if device.type == "cuda" else None,
                        gradient_as_bucket_view=True) if world > 1 else raw)
        models[key] = {"raw": raw, "model": parallel, "best": -math.inf, "stale": 0,
                       "optimizer": torch.optim.Adam(parallel.parameters(), lr=cfg["learning_rate"]),
                       "scaler": torch.amp.GradScaler("cuda", enabled=device.type == "cuda"),
                       "history": [], "stop": "epoch_cap"}

    for key, m in models.items():
        for epoch in range(1, cfg["max_epochs"] + 1):
            train.epoch = epoch
            train_sampler.set_epoch(epoch)
            label = f"{key} | epoch {epoch:02d}/{cfg['max_epochs']}"
            progress_label = label if is_main else None
            t0 = time.perf_counter()
            train_loss = train_epoch(m["model"], train_loader, m["optimizer"], m["scaler"],
                                     device, world, progress_label)
            cm, _ = evaluate(m["raw"], val_loader, device, progress_label=progress_label)
            _, val_miou = validate_scores(cm)
            seconds = time.perf_counter() - t0
            m["history"].append({"epoch": epoch, "train_loss": train_loss,
                                 "val_miou": val_miou, "epoch_seconds": seconds})
            if val_miou > m["best"]:
                m["best"], m["stale"], m["best_epoch"] = val_miou, 0, epoch
                if is_main:
                    state = {k: v.detach().cpu() for k, v in m["raw"].state_dict().items()}
                    checkpoint = {"model_state_dict": state, "dilation": 1 if key == "baseline" else 2,
                                  "epoch": epoch, "val_miou": val_miou, "image_size": cfg["image_size"]}
                    tmp = run_dir / f"{key}.pt.tmp"
                    torch.save(checkpoint, tmp)
                    tmp.replace(run_dir / f"{key}_best.pt")
            else:
                m["stale"] += 1
                if m["stale"] >= cfg["patience"]:
                    m["stop"] = "early_stopping"
            if is_main:
                history_tmp = run_dir / "history.json.tmp"
                history_tmp.write_text(
                    json.dumps({name: model["history"] for name, model in models.items()}, indent=2),
                    encoding="utf-8")
                history_tmp.replace(run_dir / "history.json")
                print(f"{label} | complete | train loss={train_loss:.4f} | val mIoU={val_miou:.4f} | "
                      f"{seconds:.1f}s | best epoch={m['best_epoch']}", flush=True)
            if m["stop"] == "early_stopping":
                if is_main:
                    print(f"{label} | early stopping | best val mIoU={m['best']:.4f}", flush=True)
                break

    if world > 1:
        dist.barrier()  # Rank 0 must finish writing the final checkpoints before other ranks load them.
    results = {}
    if is_main:
        print("Training complete. Evaluating the best checkpoints on the held-out split.")
    for key, m in models.items():
        checkpoint = torch.load(run_dir / f"{key}_best.pt", map_location="cpu", weights_only=True)
        raw = m["raw"]
        raw.load_state_dict(checkpoint["model_state_dict"])
        cm, picks = evaluate(raw, test_loader, device, collect_errors=key == "dilated",
                             progress_label=key if is_main else None,
                             progress_phase="held-out evaluation (rank 0 shard)")
        if key == "dilated":
            if world > 1:
                from_ranks = [None] * world if is_main else None
                dist.gather_object(picks, object_gather_list=from_ranks, dst=0)
                if is_main:
                    picks = [p for shard in from_ranks for p in shard]
            if is_main:
                save_examples(raw, test, picks, device, run_dir / "examples.npz")
        if is_main:
            (bg, pet), miou = validate_scores(cm)
            results[key] = {"background_IoU": bg, "pet_IoU": pet, "mIoU": miou,
                            "test_confusion_matrix": cm.cpu().tolist(),
                            "parameters": sum(p.numel() for p in raw.parameters()),
                            "best_epoch": m["best_epoch"], "epochs_run": len(m["history"]),
                            "stop_reason": m["stop"],
                            "latency": latency(raw, test, device, batch_size * world, cfg["smoke_test"])}
            print(f"{key} held-out mIoU: {miou:.4f}")
    if is_main:
        summary = {"training_processes": world, "official_test_split": meta["official_test_split"],
                   "smoke_test": cfg["smoke_test"], "elapsed_minutes": (time.time() - cfg["started_at"]) / 60,
                   "results": results}
        (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if world > 1:
        dist.barrier()


def main(local_cpu_rank=None):
    run_dir, pixels = map(Path, sys.argv[1:3])
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.set_num_threads(1 if world > 1 else 2)
    if cfg["device"] == "cuda":
        if torch.cuda.device_count() < world:
            raise RuntimeError(f"Found {torch.cuda.device_count()} GPUs for {world} training processes.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        backend = "nccl"
        name = torch.cuda.get_device_name(device)
    else:
        device, backend, name = torch.device("cpu"), "gloo", "CPU"
    if local_cpu_rank is not None:
        rank, world = local_cpu_rank, cfg["processes"]
        torch.set_num_threads(1)
    if world != cfg["processes"]:
        raise RuntimeError(f"Expected {cfg['processes']} processes, found {world}. Use the notebook launch cell.")
    if world > 1:
        kwargs = {}
        if local_cpu_rank is not None:
            kwargs.update(store=dist.FileStore(str(pixels.with_name("ddp_store")), world),
                          rank=rank, world_size=world)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=5), **kwargs)
    try:
        print(f"rank {rank} on {device} ({name}) | backend={backend if world > 1 else 'single process'}",
              flush=True)
        run(run_dir, pixels, cfg, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    config = json.loads((Path(sys.argv[1]) / "config.json").read_text(encoding="utf-8"))
    if os.name == "nt" and config["device"] == "cpu" and config["processes"] > 1:
        # Local verification on Windows, where NCCL isn't supported.
        torch.multiprocessing.spawn(main, nprocs=config["processes"], join=True)
    else:
        main()
