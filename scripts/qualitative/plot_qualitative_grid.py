"""Qualitative grid: one row per dataset; columns = input RGB, GT, then each model's prediction.

Reads the .npz files written by dump_test_sample.py from scripts/qualitative/out/.
Writes notebooks/figures/qualitative_grid.pdf (and .png for quick viewing).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path("/dss/dsshome1/08/di54zop/pangaea-bench_dev")
OUT_DIR = PROJECT / "scripts" / "qualitative" / "out"
FIG_DIR = PROJECT / "notebooks" / "figures"

DATASETS = [
    ("sen1floods11", "Sen1Floods11"),
    ("hlsburnscars", "HLS-BurnScars"),
    ("mados", "MADOS"),
    ("ai4smallfarms", "ai4smallfarms"),
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
    """(3,H,W) -> (H,W,3) in [0,1] with 2-98 percentile stretch per channel."""
    img = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], [2, 98])
        out[..., c] = np.clip((img[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def class_cmap(n_classes: int):
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = base[: max(n_classes, 2)]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    cmap.set_bad(IGNORE_COLOR)
    return cmap


def masked(mask: np.ndarray, ignore_index: int) -> np.ma.MaskedArray:
    return np.ma.masked_where(mask == ignore_index, mask)


def main():
    n_rows = len(DATASETS)
    n_cols = 2 + len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows))

    for r, (ds_key, ds_label) in enumerate(DATASETS):
        ref_path = OUT_DIR / f"{ds_key}_unet.npz"
        if not ref_path.exists():
            for c in range(n_cols):
                axes[r, c].axis("off")
            axes[r, 0].text(0.5, 0.5, f"{ds_label}\n(missing)", ha="center", va="center")
            continue
        ref = np.load(ref_path)
        ignore = int(ref["ignore_index"])
        gt = ref["gt"]
        n_classes = int(max(gt[gt != ignore].max(initial=1), 1)) + 1

        axes[r, 0].imshow(stretch_rgb(ref["rgb"]))
        axes[r, 1].imshow(masked(gt, ignore), cmap=class_cmap(n_classes), vmin=0, vmax=n_classes - 1,
                          interpolation="nearest")

        for c, (m_key, m_label) in enumerate(MODELS, start=2):
            path = OUT_DIR / f"{ds_key}_{m_key}.npz"
            if not path.exists():
                axes[r, c].text(0.5, 0.5, "missing", ha="center", va="center")
                axes[r, c].set_facecolor("white")
                continue
            d = np.load(path)
            pred = np.ma.masked_where(gt == ignore, d["pred"])
            axes[r, c].imshow(pred, cmap=class_cmap(n_classes), vmin=0,
                              vmax=n_classes - 1, interpolation="nearest")

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
    fig.savefig(FIG_DIR / "qualitative_grid.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "qualitative_grid.png", bbox_inches="tight", dpi=150)
    print(f"saved {FIG_DIR / 'qualitative_grid.png'}")


if __name__ == "__main__":
    main()
