#!/bin/bash
#
# Band-padding ablation -- CARL, imagewise-normalized variant
#
# The original band-padding ablation's CARL leg (run_band_padding_ablation.sh)
# uses carl_no_internal_norm, but CARL's baseline (w/o padding, as reported in
# the main benchmark and every other notebook) uses carl_imagewise instead
# (the CARL paper's own default: per-image normalization, image_norm=true).
# That mismatch meant analyze_band_padding.ipynb's Δ column for CARL mixed a
# padding effect with a normalization-scheme difference, unlike every other
# encoder (same encoder config on both sides). This reruns the padding leg
# with carl_imagewise across all 9 ablation datasets so CARL's Δ is a clean,
# isolated padding effect too.
#
# Preprocessing: new *_no_norm_pad configs (BandFilter -> BandPadding, no
# NormalizeMeanStd) -- mirrors carl_imagewise's own no-external-norm
# preprocessing (it does its own internal per-image _standardize()), just with
# BandPadding added on top, same as the *_pad configs add BandPadding on top
# of the normal *_pad preprocessing for the other 5 encoders.
#
# SpaceNet7-CD and DynamicEarthNet use the fixes already applied to the rest
# of the band-padding ablation (weighted_bce_sqrt + best_metric_key=IoU +
# data_replicate=2 for the dice-collapse fix; dataset=dynamicen_true_wl for
# the B5->B8A relabeling fix) from the start, rather than needing a second fix
# pass.
#
# Output: ablations/band_padding/carl_imagewise_band_pad/ (separate from the
# existing carl_band_pad/, which stays as the carl_no_internal_norm data).
#
# Control via env vars:
#   RUN_DATASETS="all"   or space-separated dataset keys
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/band_padding_carl_imagewise"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/band_padding/carl_imagewise_band_pad"
readonly ENCODER_IMSIZE=224
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2
    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize_no_norm_pad"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop_no_norm_pad"
    else
        echo "seg_default_no_norm_pad"
    fi
}

launch_job() {
    local -r dataset=$1
    local -r dataset_cfg=$2
    local -r root_path=$3
    local -r task=$4
    local -r decoder=$5
    local -r criterion=$6
    local -r time_limit=$7
    local -ra extra_args=("${@:8}")

    local -r dataset_size="${dataset_sizes[$dataset]}"
    local -r preprocessing=$(choose_preprocessing "$dataset" "$dataset_size")
    local -r job_name="padiw_${dataset:0:10}"
    local -r sbatch_script="carl_imagewise_${dataset}_pad.sh"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%s]\n' "$dataset"
        return 0
    fi

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset_cfg}"
    cmd+=" encoder=carl_imagewise"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${root_path}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"
    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s] preprocessing=%s:\n%s\n' "$dataset" "$preprocessing" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${dataset}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

${cmd}
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# Dataset definitions
# Format: "dataset_cfg|root_path|task|decoder|criterion|time_limit|extra_args"
# dynamicen uses dynamicen_true_wl (B8A fix); spacenet7cd uses weighted_bce_sqrt
# (dice-collapse fix) -- both fixes applied from the start, matching the rest
# of this ablation after run_band_padding_fix.sh.
# -----------------------------------------------------------------------------
declare -A dataset_sizes=(
    ["hlsburnscars"]=512
    ["sen1floods11"]=512
    ["mados"]=240
    ["pastis"]=128
    ["dynamicen"]=1024
    ["croptypemapping"]=64
    ["spacenet7cd"]=256
    ["ai4smallfarms"]=496
    ["fivebillionpixels"]=520
)

declare -A datasets=(
    ["hlsburnscars"]="hlsburnscars|${DATASET_PATH}/hlsburnscars|segmentation|seg_upernet|cross_entropy|6:00:00|"
    ["sen1floods11"]="sen1floods11|${DATASET_PATH}/sen1floods11|segmentation|seg_upernet|cross_entropy|6:00:00|"
    ["mados"]="mados|${DATASET_PATH}/mados|segmentation|seg_upernet|cross_entropy|8:00:00|"
    ["pastis"]="pastis|${DATASET_PATH}/pastis|segmentation|seg_upernet_mt_ltae|cross_entropy|12:00:00|task.evaluator.inference_mode=whole"
    ["dynamicen"]="dynamicen_true_wl|${DATASET_PATH}/dynamicen|segmentation|seg_upernet_mt_ltae|cross_entropy|24:00:00|"
    ["croptypemapping"]="croptypemapping|${DATASET_PATH}/croptypemapping|segmentation|seg_upernet_mt_ltae|cross_entropy|12:00:00|task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="spacenet7cd|${DATASET_PATH}/spacenet7cd|change_detection|seg_siamupernet_conc|weighted_bce_sqrt|8:00:00|task.trainer.best_metric_key=IoU data_replicate=2"
    ["ai4smallfarms"]="ai4smallfarms|${DATASET_PATH}/ai4smallfarms|segmentation|seg_upernet|weighted_bce|10:00:00|task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="fivebillionpixels|${DATASET_PATH}/fivebillionpixels|segmentation|seg_upernet|cross_entropy|24:00:00|"
)

RUN_DATASETS="${RUN_DATASETS:-all}"
for ds in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $ds "* ]] && continue
    fi
    IFS='|' read -r dataset_cfg root_path task decoder criterion time_limit extra_str <<< "${datasets[$ds]}"
    read -ra extra_args <<< "$extra_str"
    launch_job "$ds" "$dataset_cfg" "$root_path" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo ""
echo "CARL (imagewise) band-padding jobs queued (up to 9 total)."
