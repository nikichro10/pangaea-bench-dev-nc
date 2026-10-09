"""Render a 5-candidate grid per dataset from the pools dumped by dump_candidates.py,
so a human can pick a favorite row per dataset before the final figure is assembled.

Selection rule per dataset:
  - "gap"  (hlsburnscars, ai4smallfarms, sen1floods11): rank candidates by
    (UNet mIoU - mean mIoU of the other 4 models), descending -- i.e. cases
    where UNet does noticeably better than the foundation models.
  - "avg"  (everything else): rank candidates by the mean mIoU across all 5
    models, descending -- i.e. cases where every model actually predicts
    something non-trivial, which matters most for the multi-class datasets
    (MADOS, PASTIS, CropTypeMapping, DynamicEarthNet, FiveBillionPixels) where
    a model can collapse to a single dominant class.

Only candidate indices present in *every* model's pool for a dataset are
considered (should normally be all of them, since test-set order and
preprocessing geometry are deterministic and shared across the 5 runs).

Usage:
    python scripts/qualitative/plot_candidates.py                 # all datasets
    python scripts/qualitative/plot_candidates.py --dataset mados  # one dataset
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PROJECT = Path("/dss/dsshome1/08/di54zop/pangaea-bench_dev")
OUT_DIR = PROJECT / "scripts" / "qualitative" / "out"
FIG_DIR = PROJECT / "notebooks" / "figures" / "candidates"

MODELS = [
    ("unet", "UNet"),
    ("vit_p16_frozen", "ViT-p16-Frozen"),
    ("panopticon", "Panopticon"),
    ("dofa", "DOFA"),
    ("carl", "CARL"),
]

DATASET_RULE = {
    "hlsburnscars": "gap",
    "ai4smallfarms": "gap",
    "sen1floods11": "gap",
}
DATASET_LABELS = {
    "ai4smallfarms": "ai4smallfarms",
    "croptypemapping": "CropTypeMapping",
    "dynamicen": "DynamicEarthNet",
    "fivebillionpixels": "FiveBillionPixels",
    "hlsburnscars": "HLS-BurnScars",
    "mados": "MADOS",
    "pastis": "PASTIS",
    "sen1floods11": "Sen1Floods11",
    "spacenet7cd": "SpaceNet7-CD",
}
N_ROWS = 5
IGNORE_COLOR = "lightgray"


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    img = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], [2, 98])
        out[..., c] = np.clip((img[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def class_cmap(n_classes: int):
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    cmap = ListedColormap(base[: max(n_classes, 2)])
    cmap.set_bad(IGNORE_COLOR)
    return cmap


def load_dataset_pools(ds_key: str) -> dict[str, dict] | None:
    pools = {}
    for m_key, _ in MODELS:
        path = OUT_DIR / f"cand_{ds_key}_{m_key}.npz"
        if not path.exists():
            print(f"  missing {path.name}")
            return None
        pools[m_key] = dict(np.load(path, allow_pickle=True))
    return pools


def rank_indices(pools: dict[str, dict], rule: str) -> list[int]:
    idx_sets = [set(p["idx"].tolist()) for p in pools.values()]
    common = sorted(set.intersection(*idx_sets))
    if not common:
        return []

    miou_by_model = {m: dict(zip(p["idx"].tolist(), p["miou"].tolist())) for m, p in pools.items()}

    scored = []
    for idx in common:
        per_model = {m: miou_by_model[m][idx] for m in miou_by_model}
        if rule == "gap":
            others = [v for m, v in per_model.items() if m != "unet"]
            score = per_model["unet"] - (sum(others) / len(others))
        else:  # "avg"
            score = sum(per_model.values()) / len(per_model)
        scored.append((idx, score, per_model))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:N_ROWS]


def plot_dataset(ds_key: str):
    ds_label = DATASET_LABELS[ds_key]
    rule = DATASET_RULE.get(ds_key, "avg")
    print(f"=== {ds_label} (rule={rule}) ===")

    pools = load_dataset_pools(ds_key)
    if pools is None:
        print("  skipped (missing pool files)")
        return

    ranked = rank_indices(pools, rule)
    if not ranked:
        print("  no common candidate indices across all 5 models -- skipped")
        return

    n_cols = 2 + len(MODELS)
    fig, axes = plt.subplots(len(ranked), n_cols, figsize=(2.6 * n_cols, 2.7 * len(ranked)))
    if len(ranked) == 1:
        axes = axes[None, :]

    for r, (idx, score, per_model) in enumerate(ranked):
        unet_pool = pools["unet"]
        pos = int(np.where(unet_pool["idx"] == idx)[0][0])
        ignore = int(unet_pool["ignore_index"])
        gt = unet_pool["gt"][pos]
        n_classes = int(unet_pool["num_classes"])

        axes[r, 0].imshow(stretch_rgb(unet_pool["rgb"][pos]))
        axes[r, 1].imshow(np.ma.masked_where(gt == ignore, gt), cmap=class_cmap(n_classes),
                          vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[r, 0].set_ylabel(f"idx={idx}\nscore={score:.1f}", fontsize=9)

        for c, (m_key, m_label) in enumerate(MODELS, start=2):
            pool = pools[m_key]
            p = int(np.where(pool["idx"] == idx)[0][0])
            pred = np.ma.masked_where(gt == ignore, pool["pred"][p])
            axes[r, c].imshow(pred, cmap=class_cmap(n_classes), vmin=0, vmax=n_classes - 1,
                              interpolation="nearest")
            axes[r, c].text(0.02, 0.06, f"{per_model[m_key]:.1f}", transform=axes[r, c].transAxes,
                            fontsize=8, color="white",
                            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)

        print(f"  idx={idx:4d}  score={score:6.2f}  " +
              "  ".join(f"{m}={v:.1f}" for m, v in per_model.items()))

    headers = ["Input (RGB)", "Ground truth"] + [label for _, label in MODELS]
    for c, title in enumerate(headers):
        axes[0, c].set_title(title, fontsize=11)

    fig.suptitle(f"{ds_label} -- candidates ranked by '{rule}' (mIoU shown bottom-left of each pred)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"candidates_{ds_key}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="limit to one dataset key")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_LABELS.keys())
    for ds_key in datasets:
        plot_dataset(ds_key)


if __name__ == "__main__":
    main()
