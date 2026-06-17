"""
Clean up W&B runs: delete crashed/failed runs and duplicate runs (keep best mIoU).

Usage:
    # Preview what would be deleted (no changes made)
    python3 scripts/cleanup_wandb_runs.py

    # Preview a specific project
    python3 scripts/cleanup_wandb_runs.py --project geofm-bench-rgb

    # Actually delete
    python3 scripts/cleanup_wandb_runs.py --delete

    # Delete only crashed/failed runs, keep duplicates
    python3 scripts/cleanup_wandb_runs.py --delete --no-duplicates

    # Delete only duplicates, keep crashed runs
    python3 scripts/cleanup_wandb_runs.py --delete --no-crashed
"""

import argparse
import re
from collections import defaultdict

import wandb

ENTITY   = "niklas-chromik-oth-regensburg"
PROJECTS = ["geofm-bench", "geofm-bench-rgb"]

DATASETS = [
    "hlsburnscars", "sen1floods11", "mados", "pastis", "dynamicen",
    "croptypemapping", "spacenet7cd", "ai4smallfarms", "fivebillionpixels",
]
DATASET_RE = re.compile(r"_(" + "|".join(re.escape(d) for d in DATASETS) + r")_")

ENCODERS = [
    "copernicus_fm_base", "copernicus_fm_large",
    "smarties_vitb", "smarties_planet_rgb",
    "dinov2_vitb14", "dofav2", "dofa", "panopticon", "carl",
    "hierab_mm_dinov2", "hierab_mm_dino",
    "vit_scratch_mi", "vit_scratch",
    "unet_encoder_mi", "unet_encoder",
    "gfmswin", "scalemae", "satlasnet", "remoteclip",
    "ssl4eo_dino", "ssl4eo_mae", "ssl4eo_moco", "ssl4eo_data2vec",
    "prithvi", "spectralgpt", "unet",
]

DECODERS = [
    "seg_upernet_mt_ltae", "seg_upernet",
    "seg_siamupernet_conc", "seg_siamupernet_diff",
    "seg_siam_mlp_fusion", "seg_mlp_fusion",
    "seg_siam_linear_probe", "seg_linear_probe",
    "seg_siamunet_conc", "seg_siamunet_diff", "seg_unet",
    "cls_linear_probe",
]


def parse_run_name(name: str) -> tuple[str, str, str]:
    m = DATASET_RE.search(name)
    if not m:
        return "", "", ""
    dataset = m.group(1)
    rest = name[m.end():]

    encoder = ""
    for enc in ENCODERS:
        if rest.startswith(enc + "_"):
            encoder = enc
            rest = rest[len(enc) + 1:]
            break

    decoder = ""
    for dec in DECODERS:
        if rest.startswith(dec):
            decoder = dec
            break

    return dataset, encoder, decoder


def get_metric(run) -> float | None:
    s = run.summary._json_dict
    for key in ("test_mIoU", "test_IoU", "test_mF1"):
        v = s.get(key)
        if v is not None:
            return float(v)
    return None


def fetch_runs(project: str) -> list:
    api = wandb.Api(timeout=60)
    return list(api.runs(f"{ENTITY}/{project}", per_page=2000))


def classify(projects: list[str], delete_crashed: bool, delete_duplicates: bool):
    to_delete = []
    summary = {"crashed": 0, "duplicates": 0, "kept": 0}

    for project in projects:
        print(f"\nFetching {project} ...")
        runs = fetch_runs(project)
        print(f"  {len(runs)} total runs")

        crashed, finished = [], []
        for r in runs:
            if r.state in ("crashed", "failed"):
                crashed.append(r)
            elif r.state == "finished":
                finished.append(r)

        # --- crashed/failed ---
        if delete_crashed:
            for r in crashed:
                to_delete.append((r, "crashed/failed"))
                summary["crashed"] += 1

        # --- duplicates among finished runs ---
        if delete_duplicates:
            groups = defaultdict(list)
            for r in finished:
                ds, enc, dec = parse_run_name(r.name or "")
                if ds and enc and dec:
                    groups[(project, ds, enc, dec)].append(r)
                else:
                    summary["kept"] += 1  # unparseable — leave alone

            for key, group in groups.items():
                if len(group) == 1:
                    summary["kept"] += 1
                    continue
                # keep the one with the best metric; ties broken by most recent
                def sort_key(r):
                    m = get_metric(r)
                    return (m if m is not None else -1, r.id)
                group.sort(key=sort_key, reverse=True)
                summary["kept"] += 1
                for r in group[1:]:
                    to_delete.append((r, f"duplicate (best kept: {group[0].name})"))
                    summary["duplicates"] += 1
        else:
            summary["kept"] += len(finished)

    return to_delete, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", nargs="+", default=PROJECTS)
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete runs (default: dry-run preview only)")
    parser.add_argument("--no-crashed", action="store_true",
                        help="Skip deleting crashed/failed runs")
    parser.add_argument("--no-duplicates", action="store_true",
                        help="Skip deleting duplicate runs")
    args = parser.parse_args()

    delete_crashed    = not args.no_crashed
    delete_duplicates = not args.no_duplicates

    print("Mode:", "DELETE" if args.delete else "DRY-RUN (pass --delete to apply)")
    print("Targets: crashed/failed =", delete_crashed, "| duplicates =", delete_duplicates)

    to_delete, summary = classify(args.project, delete_crashed, delete_duplicates)

    print(f"\n{'='*60}")
    print(f"Runs to delete: {len(to_delete)}")
    print(f"  crashed/failed : {summary['crashed']}")
    print(f"  duplicates     : {summary['duplicates']}")
    print(f"  kept           : {summary['kept']}")
    print(f"{'='*60}")

    if not to_delete:
        print("Nothing to delete.")
        return

    print("\nRuns marked for deletion:")
    for r, reason in to_delete:
        print(f"  [{r.state:8s}] {r.name}  —  {reason}")

    if not args.delete:
        print(f"\nDry-run complete. Run with --delete to remove {len(to_delete)} runs.")
        return

    confirm = input(f"\nDelete {len(to_delete)} runs? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    deleted, failed = 0, 0
    for r, reason in to_delete:
        try:
            r.delete()
            print(f"  Deleted: {r.name}")
            deleted += 1
        except Exception as e:
            print(f"  FAILED to delete {r.name}: {e}")
            failed += 1

    print(f"\nDone. Deleted {deleted}, failed {failed}.")


if __name__ == "__main__":
    main()
