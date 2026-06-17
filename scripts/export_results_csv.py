"""
Export test mIoU / test_f1 results from W&B to CSV.

Usage:
    python3 scripts/export_results_csv.py                      # both projects, all decoders
    python3 scripts/export_results_csv.py --project geofm-bench-rgb
    python3 scripts/export_results_csv.py --project geofm-bench-rgb "geofm-bench:dinov2_vitb14"
    python3 scripts/export_results_csv.py --decoder seg_upernet seg_upernet_mt_ltae seg_siamupernet_conc
    python3 scripts/export_results_csv.py --finished-only
    python3 scripts/export_results_csv.py --out results

Project filter syntax: "project_name" or "project_name:encoder1,encoder2"
  e.g. "geofm-bench:dinov2_vitb14" fetches only DINOv2 runs from geofm-bench.

Output: a long CSV (one row per run) and a wide CSV (encoder x decoder rows, dataset columns).
"""

import argparse
import re
import csv
import sys
from collections import defaultdict

import wandb

ENTITY = "niklas-chromik-oth-regensburg"
PROJECTS = ["geofm-bench", "geofm-bench-rgb"]

DATASETS = [
    "hlsburnscars",
    "sen1floods11",
    "mados",
    "pastis",
    "dynamicen",
    "croptypemapping",
    "spacenet7cd",
    "ai4smallfarms",
    "fivebillionpixels",
]

# Known encoders — longest first so greedy matching works correctly
ENCODERS = [
    "copernicus_fm_base", "copernicus_fm_large",
    "smarties_vitb", "smarties_planet_rgb",
    "dinov2_vitb14",
    "dofav2", "dofa",
    "panopticon",
    "carl",
    "hierab_mm_dinov2", "hierab_mm_dino",
    "vit_scratch_mi", "vit_scratch",
    "unet_encoder_mi", "unet_encoder",
    "gfmswin", "scalemae", "satlasnet", "remoteclip",
    "ssl4eo_dino", "ssl4eo_mae", "ssl4eo_moco", "ssl4eo_data2vec",
    "prithvi", "spectralgpt", "unet",
]

# Known decoders
DECODERS = [
    "seg_upernet_mt_ltae", "seg_upernet",
    "seg_siamupernet_conc", "seg_siamupernet_diff",
    "seg_siam_mlp_fusion", "seg_mlp_fusion",
    "seg_siam_linear_probe", "seg_linear_probe",
    "seg_siamunet_conc", "seg_siamunet_diff", "seg_unet",
    "cls_linear_probe",
]

DATASET_RE = re.compile(
    r"_(" + "|".join(re.escape(d) for d in DATASETS) + r")_"
)


def parse_run_name(name: str) -> tuple[str, str, str]:
    """Return (dataset, encoder, decoder) parsed from run name, or ('', '', '')."""
    m = DATASET_RE.search(name)
    if not m:
        return "", "", ""
    dataset = m.group(1)
    rest = name[m.end():]  # e.g. "copernicus_fm_base_seg_upernet_seg_default"

    encoder = ""
    for enc in ENCODERS:
        if rest.startswith(enc + "_"):
            encoder = enc
            rest = rest[len(enc) + 1:]
            break
    if not encoder:
        encoder = rest.split("_")[0]
        rest = "_".join(rest.split("_")[1:])

    decoder = ""
    for dec in DECODERS:
        if rest.startswith(dec):
            decoder = dec
            break
    if not decoder:
        parts = rest.split("_")
        preproc_starts = {"seg", "no", "cls", "focus"}
        dec_parts = []
        for p in parts:
            if p in preproc_starts and dec_parts:
                break
            dec_parts.append(p)
        decoder = "_".join(dec_parts)

    return dataset, encoder, decoder


def fetch_runs(
    project: str,
    encoder_filter: set | None = None,
    decoder_filter: set | None = None,
    finished_only: bool = False,
) -> list:
    api = wandb.Api(timeout=60)
    try:
        runs = api.runs(f"{ENTITY}/{project}", per_page=1000)
    except Exception as e:
        print(f"  Warning: could not fetch {project}: {e}", file=sys.stderr)
        return []

    allowed_states = {"finished"} if finished_only else {"finished", "crashed"}

    rows = []
    for run in runs:
        if run.state not in allowed_states:
            continue
        name = run.name or ""
        dataset, encoder, decoder = parse_run_name(name)
        if not dataset:
            continue
        if encoder_filter and encoder not in encoder_filter:
            continue
        if decoder_filter and decoder not in decoder_filter:
            continue

        summary = run.summary._json_dict
        test_miou = summary.get("test_mIoU")
        val_miou  = summary.get("val_mIoU")
        test_f1   = summary.get("test_mF1")
        if test_miou is None:
            test_miou = summary.get("test_IoU")  # change detection fallback

        rows.append({
            "project":   project,
            "run_name":  name,
            "encoder":   encoder,
            "decoder":   decoder,
            "dataset":   dataset,
            "test_mIoU": round(test_miou, 3) if test_miou is not None else "",
            "val_mIoU":  round(val_miou,  3) if val_miou  is not None else "",
            "test_mF1":  round(test_f1,   3) if test_f1   is not None else "",
        })

    return rows


def write_long(rows: list, path: str) -> None:
    fieldnames = ["project", "encoder", "decoder", "dataset",
                  "test_mIoU", "val_mIoU", "test_mF1", "run_name"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path}  ({len(rows)} rows)")


def write_wide(rows: list, path: str) -> None:
    """Pivot: rows = (project, encoder, decoder), columns = datasets."""
    best = {}
    for r in rows:
        key = (r["project"], r["encoder"], r["decoder"], r["dataset"])
        val = r["test_mIoU"]
        if val == "":
            continue
        if key not in best or val > best[key]:
            best[key] = val

    groups = defaultdict(dict)
    for (proj, enc, dec, ds), val in best.items():
        groups[(proj, enc, dec)][ds] = val

    fieldnames = ["project", "encoder", "decoder"] + DATASETS + ["mean"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for (proj, enc, dec), ds_vals in sorted(groups.items()):
            values = [ds_vals.get(ds, "") for ds in DATASETS]
            numeric = [v for v in values if v != ""]
            mean = round(sum(numeric) / len(numeric), 3) if numeric else ""
            row = {"project": proj, "encoder": enc, "decoder": dec, "mean": mean}
            row.update({ds: ds_vals.get(ds, "") for ds in DATASETS})
            w.writerow(row)
    print(f"Written: {path}  ({len(groups)} encoder x decoder combinations)")


def parse_project_spec(spec: str):
    """Parse 'project' or 'project:enc1,enc2' into (project, encoder_set|None)."""
    if ":" in spec:
        proj, enc_str = spec.split(":", 1)
        return proj.strip(), {e.strip() for e in enc_str.split(",")}
    return spec.strip(), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", nargs="+", default=PROJECTS,
        help=(
            'W&B project(s) to fetch. Use "name:enc1,enc2" to restrict encoders per project. '
            'Default: both geofm-bench and geofm-bench-rgb.'
        ),
    )
    parser.add_argument(
        "--decoder", nargs="+", default=None,
        help="Only include these decoder names (e.g. seg_upernet seg_upernet_mt_ltae).",
    )
    parser.add_argument(
        "--finished-only", action="store_true",
        help="Exclude crashed runs (only state=finished).",
    )
    parser.add_argument(
        "--out", default="results",
        help="Output file stem (default: results -> results_long.csv + results_wide.csv)",
    )
    args = parser.parse_args()

    decoder_filter = set(args.decoder) if args.decoder else None

    all_rows = []
    for spec in args.project:
        proj, encoder_filter = parse_project_spec(spec)
        enc_desc = f" (encoders: {', '.join(sorted(encoder_filter))})" if encoder_filter else ""
        print(f"Fetching {proj}{enc_desc} ...")
        rows = fetch_runs(
            proj,
            encoder_filter=encoder_filter,
            decoder_filter=decoder_filter,
            finished_only=args.finished_only,
        )
        print(f"  {len(rows)} runs matched")
        all_rows.extend(rows)

    if not all_rows:
        print("No data found.")
        return

    write_long(all_rows, f"{args.out}_long.csv")
    write_wide(all_rows, f"{args.out}_wide.csv")


if __name__ == "__main__":
    main()
