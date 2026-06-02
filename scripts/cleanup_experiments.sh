#!/bin/bash
# Cleans up experiment directories by removing redundant checkpoints.
#
# What gets deleted:
#   - checkpoint_*_final.pth   : periodic/final checkpoints (only best is needed)
#   - wandb/run-*/files/       : large wandb run file caches (keeps the run dir itself)
#
# Usage:
#   ./cleanup_experiments.sh              # dry run — shows what would be deleted
#   ./cleanup_experiments.sh --delete     # actually delete

set -eo pipefail

EXPERIMENTS_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments"
WANDB_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/wandb"

DRY_RUN=true
if [[ "${1:-}" == "--delete" ]]; then
    DRY_RUN=false
fi

# ── helpers ───────────────────────────────────────────────────────────────────

hr() { printf '%.0s─' {1..60}; echo; }

du_human() {
    du -sh "$1" 2>/dev/null | cut -f1
}

print_usage() {
    local label=$1
    echo
    hr
    printf "%-30s %s\n" "Directory" "Size"
    hr
    for d in "$EXPERIMENTS_DIR"/*/; do
        [[ -d "$d" ]] && printf "%-30s %s\n" "$(basename "$d")" "$(du_human "$d")"
    done
    printf "%-30s %s\n" "[wandb]" "$(du_human "$WANDB_DIR")"
    hr
    printf "%-30s %s\n" "TOTAL (experiments)" "$(du_human "$EXPERIMENTS_DIR")"
    echo
}

delete_or_show() {
    local target=$1
    local size
    size=$(du -sh "$target" 2>/dev/null | cut -f1)
    if $DRY_RUN; then
        printf "  [dry-run] would delete %-6s  %s\n" "$size" "$target"
    else
        printf "  deleting  %-6s  %s\n" "$size" "$target"
        rm -rf "$target"
    fi
}

# ── before ────────────────────────────────────────────────────────────────────

echo
echo "━━━  DISK USAGE BEFORE  ━━━"
print_usage

# ── find targets ──────────────────────────────────────────────────────────────

echo "━━━  TARGETS  ━━━"
echo

# 1. Final/periodic checkpoints (anything matching checkpoint_<N>_final.pth)
echo "▸ Final checkpoints (checkpoint_*_final.pth):"
found_final=false
while IFS= read -r -d '' f; do
    delete_or_show "$f"
    found_final=true
done < <(find "$EXPERIMENTS_DIR" -name "checkpoint_*_final.pth" -print0 2>/dev/null)
$found_final || echo "  none found"

echo

# 2. Wandb local file caches (the files/ subdir inside each run, keeps metadata)
echo "▸ Wandb run file caches (wandb/run-*/files/):"
found_wandb=false
while IFS= read -r -d '' d; do
    delete_or_show "$d"
    found_wandb=true
done < <(find "$WANDB_DIR" -mindepth 2 -maxdepth 2 -type d -name "files" -print0 2>/dev/null)
$found_wandb || echo "  none found"

echo

# ── after ─────────────────────────────────────────────────────────────────────

if $DRY_RUN; then
    echo "━━━  DRY RUN — nothing deleted  ━━━"
    echo "      Run with --delete to apply changes."
else
    echo "━━━  DISK USAGE AFTER  ━━━"
    print_usage
fi
echo
