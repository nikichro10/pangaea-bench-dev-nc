#!/bin/bash

# -----------------------------------------------------------------------------
# Band-padding ablation: every encoder receives the full 13-band S2 optical
# input; bands missing from the dataset are zero-filled after normalisation.
#
# Comparison baseline: regular seg benchmarks that use BandFilter only.
#
# Encoders with adaptive input_bands (DOFA, DOFAv2, Panopticon, SMARTIES,
# Copernicus-FM) use *_s2full encoder configs with hardcoded S2 bands.
# CARL already has fixed S2 input_bands — no special config needed.
#
# Preprocessing: seg_default_pad / seg_resize_pad / seg_focus_crop_pad
#   (BandFilter → NormalizeMeanStd → BandPadding)
#
# Output: ablations/band_padding/<encoder>_band_pad/
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/band_padding_ablation"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/band_padding"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2
    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize_pad"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop_pad"
    else
        echo "seg_default_pad"
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

    local -r dataset_size="${dataset_sizes[$dataset]}"
    local -r preprocessing=$(choose_preprocessing "$dataset" "$dataset_size")
    local -r work_dir="${WORK_BASE}/${encoder}_band_pad"
    local -r job_name="pad_${encoder:0:6}_${dataset:0:8}"
    local -r sbatch_script="${encoder}_${dataset}_pad.sh"

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
# Dataset definitions
# Format: "task decoder criterion time_limit [extra_args...]"
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
    ["hlsburnscars"]="segmentation    seg_upernet          cross_entropy  6:00:00"
    ["sen1floods11"]="segmentation    seg_upernet          cross_entropy  6:00:00"
    ["mados"]="segmentation          seg_upernet          cross_entropy  8:00:00"
    ["pastis"]="segmentation         seg_upernet_mt_ltae  cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation      seg_upernet_mt_ltae  cross_entropy  24:00:00"
    ["croptypemapping"]="segmentation seg_upernet_mt_ltae cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection seg_siamupernet_conc dice           8:00:00"
    ["ai4smallfarms"]="segmentation   seg_upernet          weighted_bce   10:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation seg_upernet        cross_entropy  24:00:00"
)

# -----------------------------------------------------------------------------
# Encoder definitions: "encoder_name  config_name"
# *_s2full configs have input_bands hardcoded to the full 13-band S2 set.
# CARL already has fixed S2 input_bands — uses the standard config.
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa_s2full"
    ["dofav2"]="dofav2_s2full"
    ["carl"]="carl_no_internal_norm"
    ["panopticon"]="panopticon_s2full"
    ["smarties_vitb"]="smarties_vitb_s2full"
    ["copernicus_fm_base"]="copernicus_fm_base_s2full"
)

# -----------------------------------------------------------------------------
# SELECT WHAT TO RUN
# Edit to restrict encoders or datasets.
# Use "all" to run everything.
# Example: RUN_ENCODERS="dofa carl"  RUN_DATASETS="hlsburnscars sen1floods11"
# -----------------------------------------------------------------------------
RUN_ENCODERS="carl"
RUN_DATASETS="fivebillionpixels"  # rerunning fivebillionpixels with extended time limit

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
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
