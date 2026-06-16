#!/bin/bash

# -----------------------------------------------------------------------------
# UNet supervised baseline benchmark (all downstream tasks, all dataset bands)
#
# Single-temporal datasets use unet_encoder + seg_unet.
# Multi-temporal datasets (pastis, dynamicen, croptypemapping) use unet_encoder_mi
# which concatenates all time steps before the encoder.
# Change detection (spacenet7cd) uses unet_encoder + seg_siamunet_conc.
#
# No pretrained weights — trained from scratch on each downstream task.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/unet_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/unet_seg"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

choose_preprocessing() {
    local -r dataset=$1
    if [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r encoder=$3
    local -r decoder=$4
    local -r criterion=$5
    local -r time_limit=$6
    local -ra extra_args=("${@:7}")

    local -r preprocessing=$(choose_preprocessing "$dataset")
    local -r job_name="unet_${dataset}"
    local -r sbatch_script="${dataset}_unet.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s] preprocessing=%s encoder=%s decoder=%s:\n%s\n' \
        "$dataset" "$preprocessing" "$encoder" "$decoder" "$cmd"

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

# Format: "task encoder decoder criterion time_limit [extra_args...]"
declare -A datasets=(
    ["hlsburnscars"]="segmentation    unet_encoder    seg_unet            cross_entropy  6:00:00"
    ["sen1floods11"]="segmentation    unet_encoder    seg_unet            cross_entropy  6:00:00"
    ["mados"]="segmentation          unet_encoder    seg_unet            cross_entropy  8:00:00"
    ["pastis"]="segmentation         unet_encoder_mi seg_unet            cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation      unet_encoder_mi seg_unet            cross_entropy  24:00:00"
    ["croptypemapping"]="segmentation unet_encoder_mi seg_unet           cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection unet_encoder   seg_siamunet_conc   dice           8:00:00"
    ["ai4smallfarms"]="segmentation   unet_encoder    seg_unet            weighted_bce   10:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation unet_encoder  seg_unet            cross_entropy  10:00:00"
)

RUN_DATASETS="all"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task encoder decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"
    launch_job "$dataset" "$task" "$encoder" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All jobs queued."
