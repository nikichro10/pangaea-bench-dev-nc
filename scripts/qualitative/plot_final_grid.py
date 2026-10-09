"""Final qualitative results grid: one row per dataset (user-picked example),
columns = input RGB, GT, then each model's prediction.

Pulls the exact samples from the candidate pools dumped by dump_candidates.py
(scripts/qualitative/out/cand_<dataset>_<model>.npz) by their chosen idx.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PROJECT = Path("/dss/dsshome1/08/di54zop/pangaea-bench_dev")
OUT_DIR = PROJECT / "scripts" / "qualitative" / "out"
FIG_DIR = PROJECT / "notebooks" / "figures"

# (dataset_key, display_label, chosen idx), in the requested row order
ROWS = [
    ("hlsburnscars", "HLS-BurnScars", 4),
    ("sen1floods11", "Sen1Floods11", 20),
    ("mados", "MADOS", 79),
    ("pastis", "PASTIS", 0),
    ("fivebillionpixels", "FiveBillionPixels", 8),
    ("dynamicen", "DynamicEarthNet", 6),
    ("spacenet7cd", "SpaceNet7-CD", 139),
    ("ai4smallfarms", "ai4smallfarms", 3),
]

MODELS = [
    ("unet", "UNet"),
    ("vit_p16_frozen", "ViT-p16-Frozen"),
    ("panopticon", "Panopticon"),
    ("dofa", "DOFA"),
    ("carl", "CARL"),
]
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


def load_pool(ds_key: str, model_key: str) -> dict | None:
    path = OUT_DIR / f"cand_{ds_key}_{model_key}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def main():
    n_rows = len(ROWS)
    n_cols = 2 + len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows))

    for r, (ds_key, ds_label, chosen_idx) in enumerate(ROWS):
        ref = load_pool(ds_key, "unet")
        if ref is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            axes[r, 0].text(0.5, 0.5, f"{ds_label}\n(missing)", ha="center", va="center")
            continue

        pos = np.where(ref["idx"] == chosen_idx)[0]
        if len(pos) == 0:
            raise ValueError(f"idx={chosen_idx} not found in cand_{ds_key}_unet.npz candidate pool")
        pos = int(pos[0])

        ignore = int(ref["ignore_index"])
        gt = ref["gt"][pos]
        n_classes = int(ref["num_classes"])

        axes[r, 0].imshow(stretch_rgb(ref["rgb"][pos]))
        axes[r, 1].imshow(np.ma.masked_where(gt == ignore, gt), cmap=class_cmap(n_classes),
                          vmin=0, vmax=n_classes - 1, interpolation="nearest")

        for c, (m_key, m_label) in enumerate(MODELS, start=2):
            pool = load_pool(ds_key, m_key)
            if pool is None:
                axes[r, c].text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            mpos = np.where(pool["idx"] == chosen_idx)[0]
            if len(mpos) == 0:
                axes[r, c].text(0.5, 0.5, "idx missing", ha="center", va="center")
                continue
            mpos = int(mpos[0])
            pred = np.ma.masked_where(gt == ignore, pool["pred"][mpos])
            axes[r, c].imshow(pred, cmap=class_cmap(n_classes), vmin=0, vmax=n_classes - 1,
                              interpolation="nearest")

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(ds_label, fontsize=12, fontweight="bold")

    headers = ["Input (RGB)", "Ground truth"] + [label for _, label in MODELS]
    for c, title in enumerate(headers):
        axes[0, c].set_title(title, fontsize=12)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "qualitative_grid_final.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "qualitative_grid_final.png", bbox_inches="tight", dpi=150)
    print(f"saved {FIG_DIR / 'qualitative_grid_final.png'}")


if __name__ == "__main__":
    main()
