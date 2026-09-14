#!/bin/bash

# -----------------------------------------------------------------------------
# CropTypeMapping seed-stability ablation
#
# CropTypeMapping (multi-temporal, multi-modal S1+S2 segmentation) results
# have looked noisy across encoders. This runs 3 seeds (0, 1, 2) for every
# encoder in the standard 13-encoder benchmark roster so run-to-run variance
# can be quantified, in addition to the encoder-to-encoder differences.
#
# Mirrors the exact dataset=croptypemapping invocation each encoder's own
# run_<encoder>_seg_benchmark.sh uses (same encoder config, decoder,
# preprocessing, and extra_args) — only `seed` and `work_dir` differ here.
# CARL and CARL-Imagewise are both included (2 CARL variants).
# ViT-Scratch and ViT-Frozen share the same vit_scratch_mi encoder config and
# are distinguished only by finetune=True/False, matching their own scripts.
#
# Output: ablations/croptypemapping_seeds/<encoder_key>/ (3 runs per encoder,
# one per seed, each in its own timestamped subfolder — same convention as
# the regular benchmark scripts).
#
# Control via env vars:
#   RUN_ENCODERS="all"   or space-separated keys (e.g. "carl carl_imagewise")
#   SEEDS="0 1 2"         or any space-separated list of seeds
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/croptypemapping_seed_ablation"
readonly DATASET_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/croptypemapping"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/croptypemapping_seeds"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

# -----------------------------------------------------------------------------
# ENCODER DEFINITIONS
# Format: "encoder_cfg|decoder|preprocessing|time_limit|finetune|extra_args"
#   finetune: "true", "false", or "" (omit the flag — pretrained encoders that
#             don't expose it, i.e. everything except the UNet/ViT-scratch family)
#   extra_args: space-separated, appended verbatim (may be empty)
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole"
    ["dofav2"]="dofav2|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole"
    ["panopticon"]="panopticon|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole"
    ["copernicus_fm_base"]="copernicus_fm_base|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole"
    ["smarties_vitb"]="smarties_vitb|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole"
    ["carl"]="carl|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole task.trainer.ckpt_interval=999"
    ["carl_imagewise"]="carl_imagewise|seg_upernet_mt_ltae|seg_resize_no_norm|12:00:00||task.evaluator.inference_mode=whole task.trainer.ckpt_interval=999"
    ["dinov2"]="dinov2_vitb14|seg_upernet_mt_ltae|seg_resize|12:00:00||task.evaluator.inference_mode=whole 'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["dinov3_vitl"]="dinov3_vitl16_sat493m|seg_upernet_mt_ltae|seg_resize|16:00:00||task.evaluator.inference_mode=whole 'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["unet"]="unet_encoder_mi|seg_unet|seg_default|12:00:00|true|task.evaluator.inference_mode=whole wandb_project=geofm-bench"
    ["vit_scratch"]="vit_scratch_mi|seg_upernet_mt_ltae|seg_resize|12:00:00|true|task.evaluator.inference_mode=whole wandb_project=geofm-bench"
    ["vit_frozen"]="vit_scratch_mi|seg_upernet_mt_ltae|seg_resize|12:00:00|false|task.evaluator.inference_mode=whole"
    ["vit_frozen_p8"]="vit_scratch_mi_p8|seg_upernet_mt_ltae|seg_resize|12:00:00|false|task.evaluator.inference_mode=whole"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r encoder_key=$1
    local -r seed=$2

    IFS='|' read -r encoder decoder preprocessing time_limit finetune extra_str <<< "${encoders[$encoder_key]}"
    local -ra extra_args=($extra_str)

    local -r work_dir="${WORK_BASE}/${encoder_key}"
    mkdir -p "${work_dir}"

    local -r job_name="ctm_seed_${encoder_key}_s${seed}"
    local -r sbatch_script="croptypemapping_${encoder_key}_seed${seed}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=croptypemapping"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_ROOT}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" seed=${seed}"

    [ -n "$finetune" ] && cmd+=" finetune=${finetune}"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s seed=%s] encoder=%s preprocessing=%s:\n%s\n' \
        "$encoder_key" "$seed" "$encoder" "$preprocessing" "$cmd"

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${encoder_key}_seed${seed}_output.%j
#SBATCH --error=${LOG_DIR}/${encoder_key}_seed${seed}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SELECT WHAT TO RUN
# -----------------------------------------------------------------------------
RUN_ENCODERS="${RUN_ENCODERS:-all}"
SEEDS="${SEEDS:-0 1 2}"

for encoder_key in "${!encoders[@]}"; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $encoder_key "* ]] && continue
    fi

    for seed in $SEEDS; do
        launch_job "$encoder_key" "$seed"
    done
done

echo "All jobs queued."
