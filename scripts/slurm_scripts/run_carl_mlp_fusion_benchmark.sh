#!/bin/bash

# -----------------------------------------------------------------------------
# CARL MLP-fusion benchmark — two normalization variants
#
# datasetwise: carl_no_internal_norm + seg_default* (dataset-wise NormalizeMeanStd,
#              matches all other encoders in the benchmark)
# imagewise:   carl (image_norm=true) + seg_default_no_norm* (CARL's internal
#              per-image normalization only, no external NormalizeMeanStd)
#
# Control via env vars:
#   RUN_DATASETS="all"          or space-separated keys (e.g. "mados pastis")
#   RUN_VARIANTS="all"          or "datasetwise" / "imagewise"
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_mlp_fusion"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR_DATASETWISE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/mlp_fusion/carl_mlp_fusion"
readonly WORK_DIR_IMAGEWISE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/mlp_fusion/carl_imagewise_mlp_fusion"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR_DATASETWISE}" "${WORK_DIR_IMAGEWISE}"

# -----------------------------------------------------------------------------
# PREPROCESSING SELECTION
# -----------------------------------------------------------------------------
choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2

    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

choose_preprocessing_no_norm() {
    local -r dataset=$1
    local -r dataset_size=$2

    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize_no_norm"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop_no_norm"
    else
        echo "seg_default_no_norm"
    fi
}

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# Args: dataset encoder work_dir preprocessing task decoder criterion time_limit [extra_args...]
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r encoder=$2
    local -r work_dir=$3
    local -r preprocessing=$4
    local -r task=$5
    local -r decoder=$6
    local -r criterion=$7
    local -r time_limit=$8
    local -ra extra_args=("${@:9}")

    local -r base_dataset="${dataset%_optical}"
    local -r job_name="carl_mlp_${encoder##carl}_${dataset}"
    local -r sbatch_script="${dataset}_carl_mlp_${encoder}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s] encoder=%s preprocessing=%s:\n%s\n' \
        "$dataset" "$encoder" "$preprocessing" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${encoder}_${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${encoder}_${dataset}_error.%j
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

# -----------------------------------------------------------------------------
# IMAGE SIZES (for preprocessing selection)
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

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder criterion time_limit [extra_args...]"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["hlsburnscars"]="segmentation    seg_mlp_fusion      cross_entropy  3:00:00"
    ["sen1floods11"]="segmentation    seg_mlp_fusion      cross_entropy  3:00:00"
    ["mados"]="segmentation          seg_mlp_fusion      cross_entropy  4:00:00"
    ["pastis"]="segmentation         seg_mlp_fusion      cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation      seg_mlp_fusion      cross_entropy  12:00:00"
    ["croptypemapping"]="segmentation seg_mlp_fusion     cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection seg_siam_mlp_fusion dice           4:00:00"
    ["ai4smallfarms"]="segmentation   seg_mlp_fusion      weighted_bce   6:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation seg_mlp_fusion   cross_entropy  12:00:00"
)

# -----------------------------------------------------------------------------
# SELECT WHAT TO RUN
# -----------------------------------------------------------------------------
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_VARIANTS="${RUN_VARIANTS:-all}"   # "all", "datasetwise" or "imagewise"

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi

    read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"

    base_dataset="${dataset%_optical}"
    dataset_size="${dataset_sizes[$base_dataset]}"

    if [[ "$RUN_VARIANTS" == "all" || "$RUN_VARIANTS" == "datasetwise" ]]; then
        preprocessing=$(choose_preprocessing "$base_dataset" "$dataset_size")
        launch_job "$dataset" "carl" "$WORK_DIR_DATASETWISE" \
            "$preprocessing" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
    fi

    if [[ "$RUN_VARIANTS" == "all" || "$RUN_VARIANTS" == "imagewise" ]]; then
        preprocessing=$(choose_preprocessing_no_norm "$base_dataset" "$dataset_size")
        launch_job "$dataset" "carl_imagewise" "$WORK_DIR_IMAGEWISE" \
            "$preprocessing" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
    fi
done

echo "All jobs queued."
