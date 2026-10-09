"""Dump a pool of candidate test samples (input RGB, GT, prediction, per-sample mIoU)
for one trained run, so a later selection step can pick the most useful example.

Usage:
    python scripts/qualitative/dump_candidates.py --run_dir <run_dir> --out <file.npz>

The run's saved configs/config.yaml drives everything (encoder, decoder, dataset,
preprocessing, evaluator inference mode), so predictions match the benchmark's own
test pass exactly.

Candidate selection: scans the *preprocessed* test set (what the model actually
sees -- fixes an earlier bug where candidates were picked from the raw, pre-crop
target) for samples with at least MIN_FG_FRACTION foreground, stopping once
POOL_SIZE candidates are found or SCAN_LIMIT samples have been scanned. Runs
inference on every candidate and stores per-sample mIoU (same formula as the
benchmark's own SegEvaluator.compute_metrics) alongside the arrays, so a
dataset-specific selection rule (e.g. "UNet high, others low") can be applied
downstream without re-running the model.
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

MIN_MINORITY_FRACTION = 0.05
MIN_CLASSES_PRESENT = 2
SCAN_LIMIT = 250
POOL_SIZE = 20
# Batched collate requires equal tensor sizes within a batch; some datasets
# (e.g. ai4smallfarms) don't resize to a fixed patch size, so different test
# tiles can have different H/W. Process one sample at a time to stay correct
# across all datasets -- the pool is small (<=20) so this is cheap either way.
BATCH_SIZE = 1
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


def diversity_score(target: np.ndarray, ignore_index: int) -> tuple[float, int]:
    """(minority_fraction, n_classes_present) over non-ignore pixels.

    minority_fraction = 1 - (largest class's pixel share). Generalizes "has
    foreground" across both binary datasets (0=background/1=event, where this
    is literally the event fraction) and multi-class semantic segmentation
    datasets (e.g. MADOS's 15 marine classes, where class 0 is just "Marine
    Water" -- not an uninteresting background to exclude). A patch dominated
    by a single class (binary empty-tile or monotonous landcover crop) scores
    near 0 either way.
    """
    valid = target != ignore_index
    if valid.sum() == 0:
        return 0.0, 0
    vals, counts = np.unique(target[valid], return_counts=True)
    minority_fraction = 1.0 - counts.max() / counts.sum()
    return float(minority_fraction), int(len(vals))


def main():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    logger = logging.getLogger("dump_candidates")
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
    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)
    evaluator = instantiate(cfg.task.evaluator, val_loader=loader, exp_dir=args.out.parent, device=device)
    ignore_index = evaluator.ignore_index
    num_classes = evaluator.num_classes

    # ── Scan for candidates (preprocessed target -- what the model actually sees) ──
    candidates = []  # list of (idx, sample_dict, minority_fraction, n_classes)
    scan_n = min(len(test_ds), SCAN_LIMIT)
    for i in range(scan_n):
        sample = test_ds[i]
        frac, n_cls = diversity_score(np.asarray(sample["target"]), ignore_index)
        if frac >= MIN_MINORITY_FRACTION and n_cls >= MIN_CLASSES_PRESENT:
            candidates.append((i, sample, frac, n_cls))
            if len(candidates) >= POOL_SIZE:
                break
    if not candidates:
        # fallback: nothing passed the threshold -- take whatever is most diverse
        logger.warning("No sample >= %.2f minority fraction in first %d; falling back to best-effort",
                        MIN_MINORITY_FRACTION, scan_n)
        scored = []
        for i in range(scan_n):
            sample = test_ds[i]
            frac, n_cls = diversity_score(np.asarray(sample["target"]), ignore_index)
            scored.append((i, sample, frac, n_cls))
        scored.sort(key=lambda t: t[2], reverse=True)
        candidates = scored[:POOL_SIZE]

    logger.info("Found %d candidates (scanned %d test samples)", len(candidates), scan_n)

    band_names = OmegaConf.to_container(cfg.dataset.bands, resolve=True)[modalities[0]]
    rgb_idx = pick_rgb_indices(list(band_names))

    all_rgb, all_gt, all_pred, all_idx, all_frac, all_nclasses, all_miou = [], [], [], [], [], [], []

    with torch.no_grad():
        for start in range(0, len(candidates), BATCH_SIZE):
            chunk = candidates[start : start + BATCH_SIZE]
            batch = collate_fn([c[1] for c in chunk])
            image = {k: v.to(device) for k, v in batch["image"].items()}
            target = batch["target"].to(device)

            if evaluator.inference_mode == "sliding":
                logits = evaluator.sliding_inference(
                    model, image, encoder.input_size, output_shape=target.shape[-2:],
                    max_batch=evaluator.sliding_inference_batch,
                )
            else:
                logits = model(image, output_shape=target.shape[-2:])
            # Binary datasets (num_classes=1, BCE/Dice loss -- e.g. ai4smallfarms,
            # SpaceNet7-CD) output a single logit channel; argmax over a size-1
            # dim is always 0, which silently collapses every prediction to the
            # background class. Mirror SegEvaluator.evaluate()'s branch instead.
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            else:
                pred = torch.argmax(logits, dim=1)

            for b, (idx, sample, frac, n_cls) in enumerate(chunk):
                t_np = target[b].cpu().numpy().astype(np.int64)
                p_np = pred[b].cpu().numpy().astype(np.int64)
                valid = t_np != ignore_index
                cm = torch.zeros((num_classes, num_classes))
                if valid.any():
                    count = np.bincount(
                        p_np[valid] * num_classes + t_np[valid], minlength=num_classes ** 2
                    )
                    cm = torch.tensor(count.reshape(num_classes, num_classes), dtype=torch.float32)
                miou = evaluator.compute_metrics(cm)["mIoU"]

                first_mod = modalities[0]
                img = batch["image"][first_mod][b].cpu()
                if img.ndim == 4:  # C,T,H,W single/multi-temporal -> first time step, C,H,W
                    img = img[:, 0]
                rgb = img[rgb_idx].numpy().astype(np.float32)

                all_rgb.append(rgb)
                all_gt.append(t_np.astype(np.int32))
                all_pred.append(p_np.astype(np.int32))
                all_idx.append(idx)
                all_frac.append(frac)
                all_nclasses.append(n_cls)
                all_miou.append(miou)

    def stack_or_object(arrs):
        # Most datasets resize/crop to a fixed patch size, so a plain stacked
        # array is preferable (simpler downstream indexing). A few datasets
        # (e.g. ai4smallfarms) don't resize test tiles to a common size, so
        # fall back to a dtype=object array of per-candidate arrays instead.
        shapes = {a.shape for a in arrs}
        if len(shapes) == 1:
            return np.stack(arrs)
        obj = np.empty(len(arrs), dtype=object)
        for i, a in enumerate(arrs):
            obj[i] = a
        return obj

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rgb=stack_or_object(all_rgb),
        gt=stack_or_object(all_gt),
        pred=stack_or_object(all_pred),
        idx=np.array(all_idx),
        minority_fraction=np.array(all_frac),
        n_classes_present=np.array(all_nclasses),
        miou=np.array(all_miou),
        ignore_index=ignore_index,
        num_classes=num_classes,
    )
    logger.info("Saved %s (%d candidates)", args.out, len(all_idx))


if __name__ == "__main__":
    main()
