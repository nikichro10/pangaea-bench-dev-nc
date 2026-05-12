"""
Debug script: shows what BandFilter and BandPadding actually do for DOFA
on different datasets. Run from pangaea-bench_dev root:
  python scripts/debug_bandpadding.py
"""
import sys
sys.path.insert(0, ".")

import torch
from omegaconf import OmegaConf

from pangaea.engine.data_preprocessor import BandFilter, BandPadding

datasets = {
    "hlsburnscars": {
        "optical": ["B2", "B3", "B4", "B8A", "B11", "B12"]
    },
    "mados": {
        "optical": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    },
    "spacenet7cd": {
        "optical": ["B4", "B3", "B2"]
    },
    "sen1floods11": {
        "optical": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "sar":     ["VV", "VH"],
    },
}

for ds_name, data_bands in datasets.items():
    # DOFA: encoder_bands = dataset.bands (same as data_bands)
    encoder_bands = data_bands

    meta = {
        "data_bands":    {k: list(v) for k, v in data_bands.items()},
        "encoder_bands": {k: list(v) for k, v in encoder_bands.items()},
        "data_mean":     {k: torch.zeros(len(v)) for k, v in data_bands.items()},
        "data_std":      {k: torch.ones(len(v))  for k, v in data_bands.items()},
        "data_min":      {k: torch.zeros(len(v)) for k, v in data_bands.items()},
        "data_max":      {k: torch.ones(len(v))  for k, v in data_bands.items()},
        "data_img_size": 224,
    }

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"  data_bands:    {data_bands}")
    print(f"  encoder_bands: {encoder_bands}  (= dataset.bands for DOFA)")

    # BandFilter
    bf = BandFilter(**meta)
    print(f"\n  BandFilter.used_bands_indices:")
    for k, idx in bf.used_bands_indices.items():
        print(f"    [{k}] indices={idx.tolist()}  → keeps bands: {[data_bands[k][i] for i in idx.tolist()]}")

    # update meta after BandFilter
    meta = bf.update_meta(meta)

    # BandPadding
    bp = BandPadding(**meta)
    print(f"\n  BandPadding.avail_bands_mask:")
    for k, mask in bp.avail_bands_mask.items():
        zeros = (~mask).sum().item()
        print(f"    [{k}] mask={mask.tolist()}  → {zeros} zero-padded bands")

    # Simulate a forward pass through BandPadding
    fake_image = {k: torch.ones(len(v), 1, 4, 4) for k, v in data_bands.items()}
    data = {"image": fake_image, "target": torch.zeros(4, 4), "metadata": {}}
    out = bp(data)
    print(f"\n  Output channel counts:")
    for k, t in out["image"].items():
        print(f"    [{k}] shape={list(t.shape)}  non-zero channels={int((t.sum(dim=(1,2,3)) != 0).sum())}/{t.shape[0]}")
