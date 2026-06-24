#!/bin/bash

# -----------------------------------------------------------------------------
# Normalization ablation — Condition 2: image-wise normalization
#
# Per-image zero-mean / unit-variance normalization (NormalizeImageWise).
# CARL's internal _standardize() is disabled (image_norm=false) to avoid
# double-normalization; the preprocessing pipeline provides the same operation.
#
# Datasets: hlsburnscars, mados, spacenet7cd
# Encoders: dofa, dofav2, carl, panopticon, smarties_vitb, copernicus_fm_base
# Output:   ablations/norm_ablation/<encoder>_imagewise/
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/norm_ablation_imagewise"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/norm_ablation"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

# mados uses FocusRandomCrop; hlsburnscars and spacenet7cd use RandomCrop
choose_preprocessing() {
    local -r dataset=$1
    if [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop_imagewise_norm"
    else
        echo "seg_default_imagewise_norm"
    fi
}

launch_job() {
    local -r encoder=$1
    local -r encoder_cfg=$2
    local -r dataset=$3
    local -r task=$4
    local -r decoder=$5
    local -r criterion=$6
    local -r time_limit=$7
    local -ra extra_args=("${@:8}")

    local -r preprocessing=$(choose_preprocessing "$dataset")
    local -r work_dir="${WORK_BASE}/${encoder}_imagewise"
    local -r job_name="iw_${encoder:0:6}_${dataset:0:8}"
    local -r sbatch_script="${encoder}_${dataset}_imagewise.sh"

    mkdir -p "${work_dir}"
    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder_cfg}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s / %s] preprocessing=%s:\n%s\n' \
        "$encoder" "$dataset" "$preprocessing" "$cmd"

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
# Dataset definitions: "task decoder criterion time_limit"
# Only the 3 ablation datasets.
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["hlsburnscars"]="segmentation    seg_upernet          cross_entropy  6:00:00"
    ["mados"]="segmentation          seg_upernet          cross_entropy  8:00:00"
    ["spacenet7cd"]="change_detection seg_siamupernet_conc dice           8:00:00"
)

# -----------------------------------------------------------------------------
# Encoder definitions: "encoder_name  config_name"
# CARL uses carl_no_internal_norm to avoid double-normalisation.
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa"
    ["dofav2"]="dofav2"
    ["carl"]="carl_no_internal_norm"
    ["panopticon"]="panopticon"
    ["smarties_vitb"]="smarties_vitb"
    ["copernicus_fm_base"]="copernicus_fm_base"
)

RUN_ENCODERS="all"
RUN_DATASETS="all"

for encoder in "${!encoders[@]}"; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $encoder "* ]] && continue
    fi
    encoder_cfg="${encoders[$encoder]}"

    for dataset in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
        fi

        read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
        read -ra extra_args <<< "$remainder"
        launch_job "$encoder" "$encoder_cfg" "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
    done
done

echo "All jobs queued."
