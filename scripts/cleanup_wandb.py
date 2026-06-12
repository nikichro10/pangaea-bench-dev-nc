"""
Delete media/images files from finished W&B runs in the geofm-bench project.
Metrics, config, and logs are preserved — only the visualization PNGs are removed.

Usage:
    python scripts/cleanup_wandb.py            # dry-run (default)
    python scripts/cleanup_wandb.py --execute  # actually delete
    python scripts/cleanup_wandb.py --execute --min-age-days 7  # only runs older than 7 days
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta

import wandb

ENTITY = "niklas-chromik-oth-regensburg"
PROJECT = "geofm-bench"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Actually delete (default: dry-run)")
    p.add_argument("--min-age-days", type=int, default=0,
                   help="Only clean runs older than this many days (default: 0 = all finished)")
    p.add_argument("--state", default="finished",
                   help="Only clean runs in this state (default: finished)")
    return p.parse_args()


def main():
    args = parse_args()
    api = wandb.Api(timeout=120)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.min_age_days)

    print(f"Project : {ENTITY}/{PROJECT}")
    print(f"State   : {args.state}")
    print(f"Min age : {args.min_age_days} days (before {cutoff.strftime('%Y-%m-%d')})")
    print(f"Mode    : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print()

    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"state": args.state}, per_page=200)
    run_list = list(runs)
    print(f"Fetched {len(run_list)} {args.state} runs\n")

    total_files = 0
    total_bytes = 0
    cleaned_runs = 0

    for i, run in enumerate(run_list):
        # Parse run creation time from the name prefix (YYYYMMDD_...)
        try:
            run_date_str = run.name.split("_")[0]
            run_date = datetime.strptime(run_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            if run_date > cutoff:
                continue
        except (ValueError, IndexError):
            pass  # can't parse date, include anyway

        image_files = [f for f in run.files() if f.name.startswith("media/images/")]
        if not image_files:
            continue

        run_bytes = sum(f.size for f in image_files)
        total_files += len(image_files)
        total_bytes += run_bytes
        cleaned_runs += 1

        status = "DELETE" if args.execute else "would delete"
        print(f"[{i+1:>4}/{len(run_list)}] {run.name[:60]:<60} "
              f"{len(image_files)} files  {run_bytes/1e6:.1f} MB  → {status}")

        if args.execute:
            for f in image_files:
                f.delete()

    print(f"\n{'=' * 70}")
    print(f"Runs affected : {cleaned_runs}")
    print(f"Files         : {total_files}")
    print(f"Space freed   : {total_bytes/1e9:.2f} GB")
    if not args.execute:
        print("\nRe-run with --execute to actually delete.")


if __name__ == "__main__":
    main()
