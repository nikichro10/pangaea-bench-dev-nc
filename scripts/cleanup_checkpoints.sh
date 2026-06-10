#!/bin/bash
# Delete periodic interval checkpoints, keeping only:
#   checkpoint__best.pth   — best validation metric
#   checkpoint_*_final.pth — end-of-training snapshot
#
# Usage:
#   bash cleanup_checkpoints.sh [--execute]
#   Default is dry-run; pass --execute to actually delete.

EXPERIMENTS_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments"
DRY_RUN=true

[[ "${1}" == "--execute" ]] && DRY_RUN=false

total_files=0
total_bytes=0

while IFS= read -r -d '' ckpt; do
    filename=$(basename "$ckpt")
    exp_dir=$(dirname "$ckpt")

    # Skip best and final checkpoints
    [[ "$filename" == "checkpoint__best.pth" ]] && continue
    [[ "$filename" == *"_final.pth" ]]          && continue
    # Match only interval checkpoints: checkpoint_<number>.pth
    [[ "$filename" =~ ^checkpoint_[0-9]+\.pth$ ]] || continue

    # Skip experiments that are still running (no final checkpoint yet)
    if ! ls "$exp_dir"/checkpoint_*_final.pth &>/dev/null; then
        echo "[skip] still running (no final ckpt): $ckpt"
        continue
    fi

    size=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
    bytes=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
    total_bytes=$((total_bytes + bytes))
    total_files=$((total_files + 1))

    if $DRY_RUN; then
        echo "[dry-run] would delete: $ckpt  ($size)"
    else
        rm "$ckpt" && echo "deleted: $ckpt  ($size)"
    fi
done < <(find "$EXPERIMENTS_DIR" -name "checkpoint_*.pth" -print0)

total_gb=$(awk "BEGIN {printf \"%.2f\", $total_bytes / 1073741824}")
echo ""
if $DRY_RUN; then
    echo "Dry run complete: $total_files files, ${total_gb} GB would be freed."
    echo "Run with --execute to delete."
else
    echo "Done: $total_files files deleted, ${total_gb} GB freed."
fi
