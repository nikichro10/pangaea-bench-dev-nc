"""Dump one fixed test sample (input RGB, GT mask, predicted mask) for a trained run.

Usage:
    python scripts/qualitative/dump_test_sample.py --run_dir <run_dir> --out <file.npz>

The run's saved configs/config.yaml drives everything (encoder, decoder, dataset,
preprocessing, evaluator inference mode), so the prediction is produced exactly
as the benchmark's own test pass would produce it.

The sample is the first test-set item whose *raw* target has at least
MIN_FG_FRACTION foreground pixels (ignore_index excluded). The choice depends
only on the raw dataset, so every model on the same dataset gets the same patch.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pangaea.datasets.base import GeoFMDataset
from pangaea.utils.collate_fn import get_collate_fn

MIN_FG_FRACTION = 0.05
RGB_NAMES = [("B4", "B04"), ("B3", "B03"), ("B2", "B02")]


class _ModuleShim(torch.nn.Module):
    """Evaluator.sliding_inference expects a DDP-style wrapper with `.module`."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def pick_rgb_indices(band_names: list[str]) -> list[int]:
    idx = []
    for alternatives in RGB_NAMES:
        match = next((i for i, b in enumerate(band_names) if b in alternatives), None)
        idx.append(match if match is not None else len(idx))
    return idx


def find_fixed_index(raw_dataset, ignore_index: int) -> int:
    for i in range(len(raw_dataset)):
        target = np.asarray(raw_dataset[i]["target"])
        valid = target != ignore_index
        if valid.sum() == 0:
            continue
        if (target[valid] > 0).mean() >= MIN_FG_FRACTION:
            return i
    raise RuntimeError("No test sample with enough foreground found")


def main():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    logger = logging.getLogger("dump_test_sample")
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.run_dir / "configs" / "config.yaml")

    encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    uses_band_padding = "BandPadding" in OmegaConf.to_yaml(cfg.preprocessing)
    if not uses_band_padding and hasattr(encoder, "set_wv_list_from_data_bands"):
        encoder.set_wv_list_from_data_bands(OmegaConf.to_container(cfg.dataset.bands, resolve=True))

    decoder = instantiate(cfg.decoder, encoder=encoder)
    ckpt = torch.load(args.run_dir / "checkpoint__best.pth", map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    decoder.load_state_dict(state)
    decoder.to(device).eval()
    model = _ModuleShim(decoder)

    raw_test = instantiate(cfg.dataset, split="test")
    test_pre = instantiate(
        cfg.preprocessing.test, dataset_cfg=cfg.dataset, encoder_cfg=cfg.encoder, _recursive_=False
    )
    test_ds = GeoFMDataset(raw_test, test_pre)

    ignore_index = int(OmegaConf.select(cfg, "task.evaluator.ignore_index", default=-1))
    sample_idx = find_fixed_index(raw_test, ignore_index)
    sample = test_ds[sample_idx]

    modalities = list(encoder.input_bands.keys())
    batch = get_collate_fn(modalities)([sample])
    image = {k: v.to(device) for k, v in batch["image"].items()}
    target = batch["target"].to(device)

    loader = DataLoader(test_ds, batch_size=1, collate_fn=get_collate_fn(modalities))
    evaluator = instantiate(
        cfg.task.evaluator, val_loader=loader, exp_dir=args.out.parent, device=device
    )
    with torch.no_grad():
        if evaluator.inference_mode == "sliding":
            input_size = encoder.input_size
            logits = evaluator.sliding_inference(
                model, image, input_size, output_shape=target.shape[-2:],
                max_batch=evaluator.sliding_inference_batch,
            )
        else:
            logits = model(image, output_shape=target.shape[-2:])
        # Binary datasets (num_classes=1, BCE/Dice loss) output a single logit
        # channel; argmax over a size-1 dim is always 0 (silent background
        # collapse). Mirror SegEvaluator.evaluate()'s branch instead.
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits) > 0.5).long()[0].cpu().numpy().astype(np.int32)
        else:
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)

    first_mod = list(image.keys())[0]
    img = batch["image"][first_mod][0].cpu()
    if img.ndim == 4:  # C,T,H,W single-temporal -> C,H,W
        img = img[:, 0]
    band_names = OmegaConf.to_container(cfg.dataset.bands, resolve=True)[first_mod]
    rgb_idx = pick_rgb_indices(list(band_names))
    rgb = img[rgb_idx].numpy().astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rgb=rgb,
        gt=target[0].cpu().numpy().astype(np.int32),
        pred=pred,
        sample_idx=sample_idx,
        ignore_index=ignore_index,
    )
    logger.info("Saved %s (sample_idx=%d)", args.out, sample_idx)


if __name__ == "__main__":
    main()
