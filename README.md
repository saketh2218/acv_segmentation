# Oxford-IIIT Pet Segmentation

Advanced Computer Vision project comparing a compact U-Net with a variant that uses dilation 2 in the first bottleneck convolution. The task is binary pixel classification on Oxford-IIIT Pet trimaps.

[Download the project report](Project_Report.docx)

![U-Net architecture with a dilated bottleneck](figures/architecture.png)

## Official test results

| Model | Background IoU | Pet IoU | mIoU | Trainable parameters | Forward ms/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline U-Net | 93.11% | 86.77% | 89.94% | 1,927,042 | 2.611 |
| Dilated U-Net | 93.77% | 88.11% | 90.94% | 1,927,042 | 2.619 |

The dilated model gained **1.00 percentage point in mIoU** across 3,669 test images. Evaluation used the checkpoints with the highest validation mIoU, from epochs 39 and 38 respectively. Timing used a single Tesla T4, FP16 autocast, a device-resident batch of 16 at 256 × 256, 10 warm-up forwards and 50 timed forwards. These numbers are batched forward times averaged per image. The run used one seed and the notebook also evaluated the official test split after the first training stage.

## Reproduce training on Kaggle

1. Import [ACV_project.ipynb](ACV_project.ipynb) into a Kaggle notebook. Choose the **GPU T4 ×2** accelerator and attach the [dataset mirror recorded in the run](https://www.kaggle.com/datasets/devdgohil/the-oxfordiiit-pet-dataset). The [Oxford dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) also provides the original archives.
2. Use **Python 3.13.16** and **PyTorch 2.11.0+cu130** to match the saved notebook metadata and setup output. The two Tesla T4s used CUDA with the NCCL backend. NumPy, Pillow, Matplotlib and IPython are required by the notebook. Their exact run versions, the CUDA driver and the cuDNN version were unrecorded.
3. Keep seed **42**, resolution **256**, batch per GPU **8**, learning rate **0.001**, and workers per GPU **0**. Keep **SMOKE_TEST = False**. Leave **DATA_ROOT = None** for automatic discovery, or point it to the attachment. Set **REQUIRE_OFFICIAL_SPLITS = True** to catch missing official split lists.
4. Run the setup, data, worker definition and **Launch training** cells. The first stage trains both models with a cap of 20 epochs and patience 5. Confirm the split counts are **3,128 train / 552 validation / 3,669 test** and compare the ordered IDs with [dataset.json](results/dataset.json).
5. Run the **EXTRA_EPOCHS = 20** continuation cell next, using the **run_dir** from the launch cell. Skip the intervening restore cell, which searches for a specific previously completed continuation folder. The continuation reloads each first-stage best model, starts fresh Adam and gradient-scaler state, and uses patience 21. Both models completed 40 logged epochs in the archived run.
6. Run the **Held-out results** and export cells. Training and evaluation outputs are written below **/kaggle/working/oxford_pet_ddp**.

The notebook maps trimap ID 1 to pet (1), ID 2 to background (0), and ID 3 to ignored (255). It resizes photos bilinearly and indexed trimaps by nearest neighbor. Photos are scaled to [−1, 1]. During training, it applies a synchronized horizontal flip with probability 0.5. Both models use the same data, initial tensors, per-epoch sample order and augmentations.

## Evaluate the saved checkpoints

Extract the original Oxford image and annotation archives into the same folder so it contains **images/** and **annotations/trimaps/**. From the repository root run the command below in a Python environment with PyTorch, NumPy and Pillow. Temporary disk use for the full test cache is about 0.90 GiB.

    python evaluate.py --data-root /path/to/oxford-pet --device cuda:0 --latency

The evaluator uses the saved test IDs, resize and label rules, original model and metric functions, and the same two evaluation shards and per-shard batch sizes. It runs the shards sequentially on one device. Use **--device cpu** for FP32 inference. Exact predictions or timings may vary by precision, GPU and library versions. The archived timings and scores came from the notebook run.

## Repository files

| File | Contents |
| --- | --- |
| [Project_Report.docx](Project_Report.docx) | Report with architecture, training protocol, results, and failure analysis |
| [ACV_project.ipynb](ACV_project.ipynb) | Original Kaggle notebook with saved outputs |
| [train_oxford_pet.py](train_oxford_pet.py) | Worker extracted from the notebook, including the original architecture, evaluator and timing code |
| [evaluate.py](evaluate.py) | Command-line checkpoint evaluator |
| [results/](results/) | Both checkpoints, metrics, split manifest, configuration, 40-epoch history, stage-one notebook log, continuation log, plots and examples |
| [figures/](figures/) | Report architecture and selected test panels |

## References and image attribution

- Parkhi, Vedaldi, Zisserman and Jawahar. *Cats and Dogs*. CVPR, 2012. The example panels use Oxford-IIIT Pet photographs and trimaps, resized and shown with model predictions. The dataset is available under CC BY-SA 4.0. The original image owners retain copyright. [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Ronneberger, Fischer and Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015. [Paper](https://arxiv.org/abs/1505.04597)
- Yu and Koltun. *Multi-Scale Context Aggregation by Dilated Convolutions*. ICLR, 2016. [Paper](https://arxiv.org/abs/1511.07122)
