#!/bin/bash

# CARL full UPerNet segmentation benchmark — dataset-wise normalization
# encoder.image_norm=false: per-image standardization is disabled; NormalizeMeanStd
# in the preprocessing pipeline provides dataset-level mean/std instead.

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_dataset_norm_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/carl_dataset_norm_seg"
readonly ENCODER="carl"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

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

# Format: "task decoder criterion time_limit [extra_args...]"
declare -A datasets=(
    ["hlsburnscars"]="segmentation    seg_upernet             cross_entropy  6:00:00"
    ["sen1floods11"]="segmentation    seg_upernet             cross_entropy  6:00:00"
    ["mados"]="segmentation          seg_upernet             cross_entropy  8:00:00"
    ["pastis"]="segmentation         seg_upernet_mt_ltae     cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation      seg_upernet_mt_ltae     cross_entropy  24:00:00"
    ["croptypemapping"]="segmentation seg_upernet_mt_ltae    cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection seg_siamupernet_conc   dice           8:00:00"
    ["ai4smallfarms"]="segmentation   seg_upernet             weighted_bce   10:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation seg_upernet           cross_entropy  10:00:00"
)

choose_preprocessing() {
    local -r dataset=$1
    local -r size=$2
    if [ "$size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r decoder=$3
    local -r criterion=$4
    local -r time_limit=$5
    local -ra extra_args=("${@:6}")

    local -r base_dataset="${dataset%_cd}"
    local -r size="${dataset_sizes[$dataset]}"
    local -r preprocessing=$(choose_preprocessing "$dataset" "$size")
    local -r job_name="carl_datasetnorm_${dataset}"
    local -r sbatch_script="/tmp/${job_name}.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" encoder.image_norm=false"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=carl-norm-ablation"
    cmd+=" task.trainer.ckpt_interval=999"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf 'Launching [%s] preprocessing=%s\n' "$dataset" "$preprocessing"

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
}

RUN_DATASETS="all"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"
    launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All 9 jobs queued."
