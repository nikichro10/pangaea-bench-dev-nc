#!/bin/bash

# -----------------------------------------------------------------------------
# GeoBench classification linear-probe benchmark — hpda2
#
# Linear probe (finetune=False) on all 6 GeoBench classification datasets.
# Preprocessing: cls_geobench_no_pad (RandomResizedCropToEncoder + BandFilter +
# NormalizeMeanStd, no padding). Band selection is handled by BandFilter using
# each encoder's own input_bands config — no per-dataset overrides needed.
#
# mBigEarthNet is multi-label (43 classes) → linear_classification_multi_label
# task and binary_cross_entropy criterion.
# All others → linear_classification + cross_entropy.
#
# carl_imagewise uses cls_geobench_imagewise_no_pad (no external NormalizeMeanStd)
# so that carl's internal normalization (image_norm: true) handles it.
# vit_scratch runs as fully finetuned (finetune=True) for comparison with the
# frozen vit_frozen_p16 linear-probe baseline.
#
# Override RUN_ENCODERS / RUN_DATASETS at call time for partial runs:
#   RUN_ENCODERS="dofa panopticon" RUN_DATASETS="meurosat mso2sat" bash run_geobench_cls_benchmark.sh
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/geobench_cls"
readonly GEO_BENCH_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly BASE_WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/geobench/cls"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${BASE_WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task criterion time_limit"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mbigearthnet"]="linear_classification_multi_label  binary_cross_entropy  9:00:00"
    ["mbrickkiln"]="linear_classification                cross_entropy         6:00:00"
    ["meurosat"]="linear_classification                  cross_entropy         2:00:00"
    ["mforestnet"]="linear_classification                cross_entropy         5:00:00"
    ["mpv4ger"]="linear_classification                   cross_entropy         3:00:00"
    ["mso2sat"]="linear_classification                   cross_entropy         8:00:00"
)

# -----------------------------------------------------------------------------
# ENCODER DEFINITIONS
# Format: "encoder_config [extra_hydra_args...]"
# All encoders use finetune=False (frozen linear probe).
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa                    cls_geobench_no_pad          finetune=False"
    ["dofav2"]="dofav2               cls_geobench_no_pad          finetune=False"
    ["carl"]="carl                    cls_geobench_no_pad          finetune=False"
    ["carl_imagewise"]="carl_imagewise cls_geobench_imagewise_no_pad finetune=False"
    ["panopticon"]="panopticon        cls_geobench_no_pad          finetune=False"
    ["smarties_vitb"]="smarties_vitb  cls_geobench_no_pad          finetune=False"
    ["copernicus_fm_base"]="copernicus_fm_base  cls_geobench_no_pad  finetune=False"
    ["dinov2"]="dinov2_vitb14         cls_geobench_no_pad          finetune=False"
    ["dinov3_vitl"]="dinov3_vitl16_sat493m  cls_geobench_no_pad    finetune=False"
    ["vit_scratch"]="vit_scratch      cls_geobench_no_pad          finetune=True"
    ["vit_frozen_p16"]="vit_scratch   cls_geobench_no_pad          finetune=False"
    ["vit_frozen_p8"]="vit_scratch_p8 cls_geobench_no_pad          finetune=False"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r enc_key=$1
    local -r enc_config=$2
    local -r preprocessing=$3
    local -r dataset=$4
    local -r task=$5
    local -r criterion=$6
    local -r time_limit=$7
    local -ra extra_args=("${@:8}")

    local -r work_dir="${BASE_WORK_DIR}/${enc_key}"
    local -r job_name="${enc_key}_cls_${dataset}"
    local -r sbatch_script="${dataset}_${enc_key}_geobench_cls.sh"

    mkdir -p "${work_dir}"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${enc_config}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" work_dir=${work_dir}"
    cmd+=" batch_size=32"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geobench"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s | %s] enc=%s prep=%s task=%s:\n%s\n' \
        "$enc_key" "$dataset" "$enc_config" "$preprocessing" "$task" "$cmd"

    trap 'rm -f "${sbatch_script}"' EXIT

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
#SBATCH --output=${LOG_DIR}/${enc_key}_${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${enc_key}_${dataset}_error.%j
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
    rm -f "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SELECT ENCODERS AND DATASETS TO RUN
# Set to "all" or a space-separated list of keys from the arrays above.
# Examples:
#   RUN_ENCODERS="dofa panopticon dinov3_vitl"
#   RUN_DATASETS="meurosat mso2sat mbrickkiln"
# -----------------------------------------------------------------------------
RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for enc_key in "${!encoders[@]}"; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $enc_key "* ]] && continue
    fi

    read -r enc_config preprocessing remainder <<< "${encoders[$enc_key]}"
    read -ra enc_extra_args <<< "$remainder"

    for dataset in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
        fi

        read -r task criterion time_limit <<< "${datasets[$dataset]}"

        launch_job "$enc_key" "$enc_config" "$preprocessing" "$dataset" \
            "$task" "$criterion" "$time_limit" "${enc_extra_args[@]}"
    done
done

echo "All jobs queued."
