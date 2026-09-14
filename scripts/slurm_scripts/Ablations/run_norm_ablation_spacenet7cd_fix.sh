#!/bin/bash
#
# SpaceNet7-CD dice-collapse fix -- normalization ablation rerun
#
# The normalization-condition scripts (customglobal, imagewise,
# custombandwise, no_norm) all ran SpaceNet7-CD with criterion=dice, the same
# collapse-prone setup fixed for the main SpaceNet7-CD benchmark. This
# reruns just the SpaceNet7-CD leg of all four conditions with
# criterion=weighted_bce_sqrt + best_metric_key=IoU + data_replicate=2.
# hlsburnscars/mados legs are untouched (dice was never broken for them --
# no severe class imbalance there).
#
# no_norm was added later (2026-09-14) -- its SpaceNet7-CD leg was found still
# on the original dice runs (20260618) with no fixed rerun, unlike
# customglobal/imagewise/custombandwise which were already fixed. Submit just
# that condition with RUN_JOBS="<enc>_no_norm ..." to avoid resubmitting the
# other three, already-fixed conditions.
#
# Encoders: DOFA, DOFAv2, CARL (carl_no_internal_norm), Panopticon,
#           SMARTIES, Copernicus-FM (frozen, finetune=false, matches originals)
# Conditions: customglobal, imagewise, custombandwise, no_norm (preprocessing only differs)
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated "<encoder>_<condition>" keys
#                     (e.g. "dofa_customglobal carl_imagewise")
# -----------------------------------------------------------------------------

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/norm_ablation_sn7_fix"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/norm_ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

declare -A encoders=(
    ["dofa"]="dofa"
    ["dofav2"]="dofav2"
    ["carl"]="carl_no_internal_norm"
    ["panopticon"]="panopticon"
    ["smarties_vitb"]="smarties_vitb"
    ["copernicus_fm_base"]="copernicus_fm_base"
)

# condition -> "preprocessing|work_dir_suffix|job_prefix"
declare -A conditions=(
    ["customglobal"]="seg_default_customglobal_nopad|customglobal|cg"
    ["imagewise"]="seg_default_imagewise_norm|imagewise|iw"
    ["custombandwise"]="seg_default_custombandwise_nopad|custombandwise|cb"
    ["no_norm"]="seg_default_no_norm|nonorm|nn"
)

launch_job() {
    local -r encoder=$1
    local -r condition=$2

    local -r encoder_cfg="${encoders[$encoder]}"
    IFS='|' read -r preprocessing work_suffix job_prefix <<< "${conditions[$condition]}"

    local -r work_dir="${WORK_BASE}/${encoder}_${work_suffix}"
    mkdir -p "${work_dir}"

    local -r job_name="${job_prefix}_${encoder:0:6}_sn7fix"
    local -r sbatch_script="${encoder}_${condition}_sn7fix.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=spacenet7cd"
    cmd+=" encoder=${encoder_cfg}"
    cmd+=" decoder=seg_siamupernet_conc"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=weighted_bce_sqrt"
    cmd+=" task=change_detection"
    cmd+=" dataset.root_path='${DATASET_PATH}/spacenet7cd'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"
    cmd+=" task.trainer.best_metric_key=IoU"
    cmd+=" data_replicate=2"

    printf '\nLaunching [%s / %s] encoder=%s preprocessing=%s:\n%s\n' \
        "$encoder" "$condition" "$encoder_cfg" "$preprocessing" "$cmd"

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
#SBATCH --time=8:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${encoder}_${condition}_output.%j
#SBATCH --error=${LOG_DIR}/${encoder}_${condition}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

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

RUN_JOBS="${RUN_JOBS:-all}"
for encoder in "${!encoders[@]}"; do
    for condition in "${!conditions[@]}"; do
        combo_key="${encoder}_${condition}"
        if [ "$RUN_JOBS" != "all" ]; then
            [[ " $RUN_JOBS " != *" $combo_key "* ]] && continue
        fi
        launch_job "$encoder" "$condition"
    done
done

echo "All jobs queued."
