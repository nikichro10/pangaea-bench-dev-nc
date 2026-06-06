#!/bin/bash

# -----------------------------------------------------------------------------
# SMARTIES planet-rgb segmentation benchmark
# Uses RGB-sensor spectrum projectors (red_1/green_1/blue_3) instead of the
# default S2 projectors, to match the sensor characteristics of Planet and
# aerial RGB data. Tests whether this closes the gap to the 62.2 mIoU
# reported in the SMARTIES paper for SpaceNet7.
#
# Datasets: SpaceNet7 (Planet RGB), xView2 (aerial RGB), Potsdam (aerial RGB)
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/smarties_planet_rgb_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/smarties_planet_rgb_seg"
readonly ENCODER="smarties_planet_rgb"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# PREPROCESSING SELECTION
# SMARTIES input_size=224:
#   img_size <= 224 -> seg_resize (upsample to 224)
#   img_size >  224 -> seg_default (random crop to 224)
# -----------------------------------------------------------------------------
choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2

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

    local -r base_dataset="${dataset%_optical}"
    local -r dataset_size="${dataset_sizes[$base_dataset]}"
    local -r preprocessing=$(choose_preprocessing "$base_dataset" "$dataset_size")
    local -r job_name="smarties_rgb_${dataset}"
    local -r sbatch_script="${dataset}_smarties_rgb.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

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
    ["spacenet7"]=256
    #["xview2"]=1024
    #["potsdam"]=512
)

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder criterion time_limit [extra_args...]"
#
# spacenet7: Planet RGB, binary building footprint, dice due to class imbalance
# xview2:    aerial RGB, binary building damage segmentation
# potsdam:   aerial RGB, multi-class urban segmentation
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["spacenet7"]="segmentation  seg_upernet  dice           6:00:00"
    #["xview2"]="segmentation     seg_upernet  cross_entropy  6:00:00"
    #["potsdam"]="segmentation    seg_upernet  cross_entropy  6:00:00"
)

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"
    launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All jobs queued."
