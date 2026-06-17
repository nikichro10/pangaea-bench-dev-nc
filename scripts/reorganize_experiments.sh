#!/bin/bash
# Reorganize experiments folder into clean structure.
# Run with --dry-run first to preview, then without to apply.

set -euo pipefail

BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments"
DRY_RUN=true
[[ "${1:-}" == "--apply" ]] && DRY_RUN=false

run() {
    if $DRY_RUN; then
        echo "[DRY] $*"
    else
        echo "$*"
        eval "$*"
    fi
}

echo "=== Mode: $( $DRY_RUN && echo DRY-RUN || echo APPLY ) ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Create new directory tree
# ---------------------------------------------------------------------------
echo "--- Creating directory structure ---"
run "mkdir -p '$BASE/pangaea_full_seg/upernet'"
run "mkdir -p '$BASE/pangaea_full_seg/mlp_fusion'"
run "mkdir -p '$BASE/pangaea_rgb/upernet'"
run "mkdir -p '$BASE/pangaea_rgb/mlp_fusion'"
run "mkdir -p '$BASE/linear_probes'"
run "mkdir -p '$BASE/ablations'"

# ---------------------------------------------------------------------------
# 2. Merge wbce ai4smallfarms runs into main seg folders, delete old ones
# ---------------------------------------------------------------------------
echo ""
echo "--- Merging wbce ai4smallfarms into main seg folders ---"

for enc in carl dofa panopticon copernicus_fm_base smarties_vitb; do
    wbce_dir="$BASE/${enc}_seg_wbce"
    main_dir="$BASE/${enc}_seg"
    if [ -d "$wbce_dir" ]; then
        # Move correct wbce run in
        for run_dir in "$wbce_dir"/*/; do
            [ -d "$run_dir" ] || continue
            run "mv '$run_dir' '$main_dir/'"
        done
        # Delete old wrong ai4smallfarms run(s) from main folder
        for old in "$main_dir"/*ai4smallfarms*/; do
            # Skip the one we just moved in (it came from wbce_dir)
            src=$(basename "$wbce_dir"/*ai4smallfarms*/ 2>/dev/null || true)
            [ -d "$old" ] || continue
            name=$(basename "$old")
            if [[ "$name" != "$src" ]]; then
                run "rm -rf '$old'"
            fi
        done
        # Remove empty wbce folder
        run "rm -rf '$wbce_dir'"
    fi
done

# ---------------------------------------------------------------------------
# 3. Merge wbce ai4smallfarms runs into main mlp_fusion folders
# ---------------------------------------------------------------------------
echo ""
echo "--- Merging wbce ai4smallfarms into main mlp_fusion folders ---"

for enc in carl dofa panopticon smarties_vitb; do
    wbce_dir="$BASE/${enc}_mlp_fusion_wbce"
    main_dir="$BASE/${enc}_mlp_fusion"
    if [ -d "$wbce_dir" ]; then
        # Keep only the best (last) wbce run if there are multiple
        best=""
        for run_dir in "$wbce_dir"/*/; do
            [ -d "$run_dir" ] && best="$run_dir"
        done
        if [ -n "$best" ]; then
            # Delete duplicates in wbce (keep only best)
            for run_dir in "$wbce_dir"/*/; do
                [ "$run_dir" != "$best" ] && run "rm -rf '$run_dir'"
            done
            run "mv '$best' '$main_dir/'"
            # Delete old wrong ai4smallfarms run from main folder
            src=$(basename "$best")
            for old in "$main_dir"/*ai4smallfarms*/; do
                [ -d "$old" ] || continue
                name=$(basename "$old")
                [[ "$name" != "$src" ]] && run "rm -rf '$old'"
            done
        fi
        run "rm -rf '$wbce_dir'"
    fi
done

# copernicus mlp_fusion_wbce is empty — just remove it
run "rm -rf '$BASE/copernicus_fm_base_mlp_fusion_wbce'"

# ---------------------------------------------------------------------------
# 4. Move full-band seg runs → pangaea_full_seg/upernet/
# ---------------------------------------------------------------------------
echo ""
echo "--- Moving full-band UPerNet runs ---"
for enc in carl dofa dofav2 panopticon smarties_vitb copernicus_fm_base dinov2 unet; do
    src="$BASE/${enc}_seg"
    [ -d "$src" ] && run "mv '$src' '$BASE/pangaea_full_seg/upernet/'"
done
run "mv '$BASE/vit_scratch_seg'                   '$BASE/pangaea_full_seg/upernet/'"
run "mv '$BASE/vit_scratch_frozen_encoder_seg'    '$BASE/pangaea_full_seg/upernet/'"

# ---------------------------------------------------------------------------
# 5. Move full-band mlp_fusion runs → pangaea_full_seg/mlp_fusion/
# ---------------------------------------------------------------------------
echo ""
echo "--- Moving full-band MLP fusion runs ---"
for enc in carl dofa dofav2 panopticon smarties_vitb copernicus_fm_base dinov2; do
    src="$BASE/${enc}_mlp_fusion"
    [ -d "$src" ] && run "mv '$src' '$BASE/pangaea_full_seg/mlp_fusion/'"
done
run "mv '$BASE/vit_scratch_frozen_encoder_mlp_fusion' '$BASE/pangaea_full_seg/mlp_fusion/'"

# ---------------------------------------------------------------------------
# 6. Move RGB runs → pangaea_rgb/
# ---------------------------------------------------------------------------
echo ""
echo "--- Moving RGB runs ---"
for enc in carl dofa dofav2 panopticon smarties_vitb copernicus_fm_base; do
    [ -d "$BASE/${enc}_rgb_seg" ]        && run "mv '$BASE/${enc}_rgb_seg'        '$BASE/pangaea_rgb/upernet/'"
    [ -d "$BASE/${enc}_rgb_mlp_fusion" ] && run "mv '$BASE/${enc}_rgb_mlp_fusion' '$BASE/pangaea_rgb/mlp_fusion/'"
done

# ---------------------------------------------------------------------------
# 7. Move linear probes
# ---------------------------------------------------------------------------
echo ""
echo "--- Moving linear probes ---"
for enc in carl dofa dofav2 panopticon smarties_vitb copernicus_fm_base; do
    src="$BASE/${enc}_linear_probe"
    [ -d "$src" ] && run "mv '$src' '$BASE/linear_probes/'"
done

# ---------------------------------------------------------------------------
# 8. Move ablations
# ---------------------------------------------------------------------------
echo ""
echo "--- Moving ablations ---"
run "mv '$BASE/dofav2_dofalayers' '$BASE/ablations/'"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Done ==="
if $DRY_RUN; then
    echo "Re-run with --apply to execute."
fi
