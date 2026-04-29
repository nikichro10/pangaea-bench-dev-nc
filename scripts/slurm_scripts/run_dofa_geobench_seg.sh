#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_geobench_seg"
readonly GEO_BENCH_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_geobench_seg"
readonly ENCODER="dofa"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# PREPROCESSING SELECTION
# All GeoBench seg datasets have img_size > 224 -> seg_default (random crop)
# -----------------------------------------------------------------------------
choose_preprocessing() {
    local -r dataset_size=$1

    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    else
        echo "seg_default"
    fi
}

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r decoder=$3
    local -r criterion=$4
    local -r time_limit=$5
    local -ra extra_args=("${@:6}")

    local -r dataset_size="${dataset_sizes[$dataset]}"
    local -r preprocessing=$(choose_preprocessing "$dataset_size")
    local -r job_name="dofa_${dataset}"
    local -r sbatch_script="${dataset}_dofa_geobench.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"

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
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"

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
# IMAGE SIZES (needed for preprocessing selection)
# -----------------------------------------------------------------------------
declare -A dataset_sizes=(
    ["mpv4ger-seg"]=320
    ["mchesapeake-landcover"]=256
    ["msa-crop-type"]=256
    ["mcashew-plantation"]=256
    ["mneontree"]=400
    ["mnz-cattle"]=500
)

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder criterion time_limit [extra_args...]"
# All GeoBench seg datasets: single-temporal, optical only, seg_upernet
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mpv4ger-seg"]="segmentation        seg_upernet  cross_entropy  4:00:00"
    ["mchesapeake-landcover"]="segmentation seg_upernet cross_entropy 4:00:00"
    ["msa-crop-type"]="segmentation       seg_upernet  cross_entropy  4:00:00"
    ["mcashew-plantation"]="segmentation  seg_upernet  cross_entropy  4:00:00"
    ["mneontree"]="segmentation           seg_upernet  cross_entropy  4:00:00"
    ["mnz-cattle"]="segmentation          seg_upernet  cross_entropy  4:00:00"
)

# -----------------------------------------------------------------------------
# SELECT DATASETS TO RUN
# Use "all" to run everything, or list specific dataset names.
# -----------------------------------------------------------------------------
RUN_DATASETS="all"  # all, or space-separated list of dataset keys

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi

    read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"

    launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All jobs queued."
