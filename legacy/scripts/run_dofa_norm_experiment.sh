#!/bin/bash

# -----------------------------------------------------------------------------
# NORMALIZATION EXPERIMENT
# Tests 3 alternative normalizations vs. the default dataset statistics:
#   1. dofa_norm     — DOFA pretraining stats (SatlasPretrain, /255)
#   2. custom_global — single global scalar (mean=0.204, std=0.144)
#   3. custom_bandwise — per-band custom stats
#
# Each normalization is tested on all datasets listed in RUN_DATASETS.
# Results land in separate work_dirs per normalization for easy comparison.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_norm_exp"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_norm_exp"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05"

mkdir -p "${LOG_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder base_preprocessing criterion time_limit [extra_args...]"
# base_preprocessing is the standard config — the script appends the norm suffix.
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["hlsburnscars"]="segmentation     seg_upernet          seg_default      cross_entropy  6:00:00"
    ["sen1floods11"]="segmentation     seg_upernet          seg_default      cross_entropy  6:00:00"
    ["ai4smallfarms"]="segmentation    seg_upernet          seg_default      dice           10:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["spacenet7cd"]="change_detection  seg_siamupernet_conc seg_default      dice           8:00:00"
    ["mados"]="segmentation            seg_upernet          seg_focus_crop   cross_entropy  8:00:00"
    ["croptypemapping_optical"]="segmentation seg_upernet_mt_ltae seg_resize cross_entropy 12:00:00 task.evaluator.inference_mode=whole"
    ["pastis_optical"]="segmentation   seg_upernet_mt_ltae  seg_resize       cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
)

# Map base preprocessing → 3 normalization variants
declare -A norm_configs=(
    ["seg_default,dofa_norm"]="seg_default_dofa_norm"
    ["seg_default,custom_global"]="seg_default_custom_global"
    ["seg_default,custom_bandwise"]="seg_default_custom_bandwise"
    ["seg_focus_crop,dofa_norm"]="seg_focus_crop_dofa_norm"
    ["seg_focus_crop,custom_global"]="seg_focus_crop_customglobalnorm"
    ["seg_focus_crop,custom_bandwise"]="seg_focus_crop_custombandwisenorm"
    ["seg_resize,dofa_norm"]="seg_resize_dofa_norm"
    ["seg_resize,custom_global"]="seg_resize_customglobalnorm"
    ["seg_resize,custom_bandwise"]="seg_resize_custombandwisenorm"
)

# -----------------------------------------------------------------------------
# SELECT DATASETS AND NORMALIZATIONS TO RUN
# -----------------------------------------------------------------------------
RUN_DATASETS="ai4smallfarms spacenet7cd mados croptypemapping_optical pastis_optical"
RUN_NORMS="dofa_norm custom_global custom_bandwise"

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r norm=$2
    local -r task=$3
    local -r decoder=$4
    local -r base_preprocessing=$5
    local -r criterion=$6
    local -r time_limit=$7
    local -ra extra_args=("${@:8}")

    local -r preprocessing="${norm_configs[${base_preprocessing},${norm}]}"
    local -r base_dataset="${dataset%_optical}"
    local -r job_name="dofa_${dataset}_${norm}"
    local -r sbatch_script="/tmp/${job_name}.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=dofa"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${WORK_DIR}/${norm}"
    cmd+=" use_wandb=True"
    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf 'Launching [%s] norm=%s preprocessing=%s\n' "$dataset" "$norm" "$preprocessing"

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
#SBATCH --output=${LOG_DIR}/${dataset}_${norm}_output.%j
#SBATCH --error=${LOG_DIR}/${dataset}_${norm}_error.%j
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

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi

    read -r task decoder base_preprocessing criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"

    for norm in $RUN_NORMS; do
        launch_job "$dataset" "$norm" "$task" "$decoder" "$base_preprocessing" \
                   "$criterion" "$time_limit" "${extra_args[@]}"
    done
done

echo "All normalization experiment jobs queued."
