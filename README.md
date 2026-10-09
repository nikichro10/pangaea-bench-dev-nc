# Benchmarking Any-Sensor Foundation Models for Earth Observation

Code for the multispectral and SAR experiments of the Master's thesis
*Benchmarking Any-Sensor Foundation Models for Earth Observation*
(Niklas Chromik, OTH Regensburg, 2026).

The thesis evaluates sensor-agnostic foundation models, which accept an arbitrary set of spectral
bands together with their wavelength metadata, under a single frozen-encoder protocol. It asks:

1. How well do current sensor-agnostic models perform across tasks and sensors, including sensors
   unseen during pretraining?
2. Do they provide an advantage over simple baselines (random frozen ViTs, a supervised ViT, a UNet)?
3. How much do additional spectral bands and wavelength metadata contribute to their performance?
4. How sensitive are the results to protocol choices such as the decoder, normalization and band padding?

This repository is a fork of [PANGAEA](https://github.com/VMarsocci/pangaea-bench) and covers
Chapter 5.1 of the thesis (PANGAEA and GeoBench). The hyperspectral experiments (Chapter 5.2) live
in a separate repository built on SpectralEarth.

## What this fork adds to PANGAEA

- **Encoders**: DOFA+ (DOFAv2), Copernicus-FM, Panopticon, CARL, SMARTIES, DINOv3 (SAT-493M),
  UniverSat, random-init ViTs with patch size 8 and 16, in `pangaea/encoders/`.
  Wavelength metadata is derived from the bands actually present in each dataset.
- **Decoder**: `MLPFusion` (`configs/decoder/seg_mlp_fusion.yaml`), a lightweight decoder that fuses
  features from evenly spaced encoder layers, used next to UPerNet.
- **No band padding by default**: each model receives only the bands the dataset provides
  (`BandFilter` instead of PANGAEA's `BandPadding`).
- **Wavelength ablations**: true sensor wavelengths, shuffled, random and constant wavelength
  metadata (`scripts/slurm_scripts/wavelength_ablation/`).
- **GeoBench** segmentation and classification tasks run with the same protocol.
- **Thesis results snapshot** in [`results/thesis/`](results/thesis/README.md).

## Models

| Thesis name | Encoder config | Notes |
|---|---|---|
| DOFA | `dofa` | Hypernetwork |
| DOFA+ | `dofav2` | Hypernetwork |
| Copernicus-FM | `copernicus_fm_base` | Hypernetwork |
| Panopticon | `panopticon` | Cross-attention |
| CARL | `carl_imagewise` | Cross-attention, image-wise normalization with `*_no_norm` preprocessing |
| SMARTIES | `smarties_vitb` | Discrete spectral projection |
| DINOv2 | `dinov2_vitb14` | RGB only |
| DINOv3-ViT-L | `dinov3_vitl16_sat493m` | RGB only |
| ViT-p16-Frozen / ViT-p8-Frozen | `vit_scratch` / `vit_scratch_p8` | Random init, frozen (`finetune=False`) |
| ViT-p16-Sup | `vit_scratch` | Trained end-to-end (`finetune=True`) |
| UNet | `unet_encoder` | Fully supervised baseline |

Multi-temporal datasets (PASTIS, DynamicEarthNet) use the `*_mi` variants of the ViT and UNet configs.
Pretrained weights are downloaded automatically to `pretrained_models/` on first use.

## Setup

```bash
conda env create -f environment.yaml     # or: mamba / micromamba
conda activate pangaea-bench
pip install --no-build-isolation --no-deps -e .
```

Datasets are downloaded as described in the [PANGAEA dataset guide](DATASET_GUIDE.md).
Set `dataset.root_path` to their location.

## Running an experiment

All runs go through `pangaea/run.py` with [Hydra](https://github.com/facebookresearch/hydra) configs
from `configs/`. Example: Panopticon with the MLPFusion decoder on HLS-BurnScars:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hlsburnscars \
   encoder=panopticon \
   decoder=seg_mlp_fusion \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   dataset.root_path=/path/to/datasets/hlsburnscars \
   work_dir=/path/to/experiments
```

The test set is evaluated after training. To evaluate a trained run again:

```bash
torchrun pangaea/run.py --config-name=test ckpt_dir=/path/to/run_dir
```

Protocol used in the thesis: frozen encoder, AdamW (lr 1e-4, weight decay 0.05), step decay at 60 % and
90 % of training, batch size 8, 80 epochs for segmentation (50 for classification), best checkpoint
selected on validation.

## Reproducing the thesis results

Every experiment was launched with a SLURM script in `scripts/slurm_scripts/`. Each script defines the
per-dataset task, decoder, loss and preprocessing, and submits one job per dataset:

```bash
RUN_DATASETS="hlsburnscars mados" bash scripts/slurm_scripts/run_panopticon_mlp_fusion_benchmark.sh
```

The scripts contain the cluster paths and SLURM settings of the LRZ Terrabyte cluster used for the
thesis; adjust `DATASET_PATH`, `WORK_DIR` and the `#SBATCH` lines for another environment.

| Folder | Experiments |
|---|---|
| `scripts/slurm_scripts/run_*_{seg,mlp_fusion}_benchmark.sh` | Main PANGAEA results (UPerNet / MLPFusion) |
| `scripts/slurm_scripts/run_geobench_*` | GeoBench segmentation and classification |
| `scripts/slurm_scripts/rgb/` | RGB-only inputs |
| `scripts/slurm_scripts/Ablations/` | Normalization and band padding ablations |
| `scripts/slurm_scripts/band_degradation/` | Incremental band groups |
| `scripts/slurm_scripts/wavelength_ablation/` | True, shuffled, random and constant wavelength metadata |
| `scripts/slurm_scripts/qualitative/`, `scripts/qualitative/` | Qualitative segmentation examples |

Tables and figures are produced by the notebooks in `notebooks/` (figures in `notebooks/figures/`).
[`results/thesis/README.md`](results/thesis/README.md) maps each table and figure of Chapter 5.1 to
its experiment group, SLURM scripts and notebook. `results/thesis/runs.csv` holds the test metrics
and `results/thesis/configs/` the resolved config of every run, so the numbers can be checked without
access to the original experiment storage.

Experiments that are not part of the thesis are kept in [`legacy/`](legacy/README.md).

## Repository layout

```
pangaea/        encoders, decoders, datasets, training and evaluation engine
configs/        Hydra configs (encoder, decoder, dataset, preprocessing, task, ...)
scripts/        SLURM launch scripts, qualitative plots, results export
notebooks/      analysis notebooks and thesis figures
results/thesis/ snapshot of all thesis runs (metrics + configs)
legacy/         experiments not used in the thesis
```

## Acknowledgements

This code builds on the PANGAEA benchmark. If you use it, please also cite PANGAEA:

```bibtex
@misc{marsocci2024pangaeaglobalinclusivebenchmark,
      title={PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models},
      author={Valerio Marsocci and Yuru Jia and Georges Le Bellier and David Kerekes and Liang Zeng and Sebastian Hafner and Sebastian Gerard and Eric Brune and Ritu Yadav and Ali Shibli and Heng Fang and Yifang Ban and Maarten Vergauwen and Nicolas Audebert and Andrea Nascetti},
      year={2024},
      eprint={2412.04204},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04204},
}
```

The encoders are adapted from the original implementations of the respective models. The experiments were run on the LRZ Terrabyte cluster.
