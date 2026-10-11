"""
Snapshot the multispectral thesis experiments (Master's thesis, Chapter 5.1) into the repo.

For every run under the experiment groups listed in GROUPS, this script
  * parses the final [test] block of train.log-0 (same logic as the analysis notebooks),
  * copies the resolved Hydra config (configs/config.yaml),
and writes
  notebooks/runs/runs.csv                         one row per run
  notebooks/runs/configs/<group>/<run_dir>.yaml   the run's config

The full train.log-0 files (~1.4 GB) and checkpoints stay on the DSS experiment storage.

Usage:
    python3 scripts/export_thesis_results.py
    python3 scripts/export_thesis_results.py --experiments /path/to/experiments --out notebooks/runs
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import yaml

EXPERIMENTS_BASE = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments")

# experiment group (relative to EXPERIMENTS_BASE) -> thesis table/figure it feeds
GROUPS = {
    "pangaea_full_seg/mlp_fusion": "Tab7, Fig15a",
    "pangaea_full_seg/upernet": "Tab8, Fig15a, Fig14",
    "ablations/spacenet7cd_weighted_bce_fix": "Tab7/8 (SN7-CD reruns)",
    "ablations/spacenet7cd_weighted_bce_sqrt_fix": "Tab7/8 (SN7-CD reruns)",
    "geobench/seg/MLPFusion": "Tab9, Fig15b",
    "geobench/seg/UPerNet": "Tab9, Fig15b",
    "geobench/cls": "Fig17 (m-ForestNet S2 default)",
    "ablations/norm_ablation": "Tab10",
    "carl_norm_ablation": "Tab10 (CARL)",
    "ablations/band_padding": "Tab11",
    "pangaea_rgb/upernet": "Tab12",
    "pangaea_rgb/mlp_fusion": "RGB-only, MLPFusion (not tabulated)",
    "band_subset_degradation": "Fig16",
    "wavelength_ablation": "Fig17",
    "wavelength_ablation_mlpfusion": "Fig17 (MLPFusion variant)",
    "wavelength_ablation_const_wl": "Fig19 (constant 550 nm)",
    "p1_wlswap_ablation": "not in thesis (neighbor swap)",
    "p2_wlswap_ablation": "Fig19 (within-group shuffle, draws A/B)",
    "p3_wlswap_ablation": "not in thesis (full reversal)",
    "p4_wlswap_ablation": "Fig19 (full random shuffle, draws 1-3)",
    "p6_random_wl_ablation": "Fig17, Fig19 (random wavelengths, MLPFusion)",
    "p6_random_wl_ablation_upernet": "Fig17, Fig19 (random wavelengths, UPerNet)",
}

# encoder folders that are not part of the thesis
SKIP_DIRS = ("universat",)

TEST_SPLIT = re.compile(r"\[test\] ------- (IoU|Accuracy)")
SECTION = re.compile(r"\[test\] ------- ([\w-]+) -+\n(.*?)\[test\] Mean\s+([\d.]+)", re.DOTALL)
CLASS_LINE = re.compile(r"^\s+(?!\[)(.+?)\s+([\d.]+)\s*$", re.MULTILINE)


def parse_test_metrics(log_path: Path) -> dict:
    """Mean of every metric in the last [test] block, plus per-class IoU."""
    text = log_path.read_text(errors="ignore")
    starts = [m.start() for m in TEST_SPLIT.finditer(text)]
    if not starts:
        return {}
    block = text[starts[-1]:]
    metrics = {}
    for name, body, mean in SECTION.findall(block):
        key = f"test_{name.replace('-score', '')}"
        if key in metrics:
            break  # next test block (should not happen after slicing)
        metrics[key] = float(mean)
        if name == "IoU":
            per_class = {c.strip(): float(v) for c, v in CLASS_LINE.findall(body)}
            metrics["test_IoU_per_class"] = json.dumps(per_class)
    return metrics


def run_dirs(group_dir: Path):
    for log in sorted(group_dir.rglob("train.log-0")):
        run = log.parent
        if any(s in part for part in run.relative_to(group_dir).parts for s in SKIP_DIRS):
            continue
        yield run


def describe(cfg: dict) -> dict:
    target = lambda key: (cfg.get(key) or {}).get("_target_", "").rsplit(".", 1)[-1]
    return {
        "dataset": (cfg.get("dataset") or {}).get("dataset_name", ""),
        "encoder": target("encoder"),
        "decoder": target("decoder"),
        "seed": cfg.get("seed", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS_BASE)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks" / "runs")
    args = ap.parse_args()

    cfg_out = args.out / "configs"
    if cfg_out.exists():
        shutil.rmtree(cfg_out)

    rows = []
    for group, thesis_ref in GROUPS.items():
        group_dir = args.experiments / group
        for run in run_dirs(group_dir):
            rel = run.relative_to(args.experiments)
            cfg_path = run / "configs" / "config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
            if cfg_path.exists():
                dst = cfg_out / rel.parent / f"{run.name}.yaml"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cfg_path, dst)
            rows.append({
                "thesis_ref": thesis_ref,
                "group": str(rel.parent),
                "run_dir": run.name,
                **describe(cfg),
                "finished": any(run.glob("checkpoint_*_final.pth")),
                **parse_test_metrics(run / "train.log-0"),
            })

    fields = ["thesis_ref", "group", "run_dir", "dataset", "encoder", "decoder", "seed", "finished",
              "test_IoU", "test_F1", "test_Precision", "test_Recall", "test_Accuracy", "test_IoU_per_class"]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_metrics = sum(1 for r in rows if "test_IoU" in r or "test_Accuracy" in r)
    print(f"{len(rows)} runs ({n_metrics} with test metrics) -> {args.out}")


if __name__ == "__main__":
    main()
