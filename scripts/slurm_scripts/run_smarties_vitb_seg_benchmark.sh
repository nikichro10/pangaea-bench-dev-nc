#!/bin/bash

# -----------------------------------------------------------------------------
# SMARTIES ViT-B segmentation benchmark
# Datasets: HLS BurnScars, DynamicEarthNet, SpaceNet7
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/smarties_vitb_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/smarties_vitb_seg"
readonly ENCODER="smarties_vitb"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# PREPROCESSING SELECTION
# SMARTIES input_size=224:
#   img_size <= 224 -> seg_resize (upsample to 224)
#   img_size >  224 -> seg_default (random crop to 224)
# All three datasets are larger than 224, so all use seg_default.
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
    local -r job_name="smarties_${dataset}"
    local -r sbatch_script="${dataset}_smarties.sh"

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
    ["hlsburnscars"]=512
    ["dynamicen"]=1024
    ["spacenet7"]=256
)

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder criterion time_limit [extra_args...]"
#
# hlsburnscars: standard single-temporal binary segmentation (burn scar vs. not)
# dynamicen:    multi-temporal (6 timesteps), LTAE fuses per-timestep SMARTIES features
# spacenet7:    binary building footprint segmentation, dice due to class imbalance
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["hlsburnscars"]="segmentation  seg_upernet          cross_entropy  6:00:00"
    ["dynamicen"]="segmentation     seg_upernet_mt_ltae  cross_entropy  24:00:00"
    ["spacenet7"]="segmentation     seg_upernet          dice           6:00:00"
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
