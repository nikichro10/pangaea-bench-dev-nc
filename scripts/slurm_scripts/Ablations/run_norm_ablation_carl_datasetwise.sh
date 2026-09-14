#!/bin/bash
#
# Normalization ablation — missing CARL "dataset_stats" (datasetwise-norm) baseline
#
# analyze_norm_ablation.ipynb's "dataset_stats" condition reuses
# pangaea_full_seg/upernet/ as its baseline source for all 6 encoders. That
# folder never actually contains a carl_no_internal_norm run (external
# dataset-wise NormalizeMeanStd only, CARL's own internal _standardize()
# disabled) for hlsburnscars/mados/spacenet7cd with UPerNet -- only plain
# "carl" (internal norm behavior undetermined/inconsistent) exists there for
# those 3 datasets. Every other condition in this ablation (imagewise,
# no_norm, custombandwise, customglobal) already uses carl_no_internal_norm
# consistently (see run_norm_ablation_imagewise.sh etc.) -- this fills the
# missing 4th leg so CARL's baseline is actually comparable to its own other
# rows, using the plain/standard preprocessing (no special norm override).
#
# Output goes to its own folder (ablations/norm_ablation/carl_datasetwise/),
# NOT into pangaea_full_seg/upernet/carl_seg/ -- that folder is shared with
# the main benchmark table (analyze_results.ipynb), which expects plain
# "carl" there; dropping a differently-normalized carl_no_internal_norm run
# into it would silently corrupt that table's CARL row via aggfunc="last".
#
# SpaceNet7-CD uses weighted_bce_sqrt + best_metric_key=IoU + data_replicate=2,
# matching the dice-collapse fix already applied to every other SpaceNet7-CD
# leg of this ablation.
# -----------------------------------------------------------------------------

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/norm_ablation_carl_datasetwise"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/norm_ablation/carl_datasetwise"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# dataset -> "task|decoder|preprocessing|criterion|extra_args"
declare -A datasets=(
    ["hlsburnscars"]="segmentation|seg_upernet|seg_default|cross_entropy|"
    ["mados"]="segmentation|seg_upernet|seg_focus_crop|cross_entropy|"
    ["spacenet7cd"]="change_detection|seg_siamupernet_conc|seg_default|weighted_bce_sqrt|task.trainer.best_metric_key=IoU data_replicate=2"
)

launch_job() {
    local -r ds=$1
    IFS='|' read -r task decoder preprocessing criterion extra_str <<< "${datasets[$ds]}"
    local -ra extra_args=($extra_str)

    local -r job_name="carl_dw_${ds:0:10}"
    local -r sbatch_script="carl_datasetwise_${ds}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${ds}"
    cmd+=" encoder=carl_no_internal_norm"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" finetune=false"
    cmd+=" dataset.root_path='${DATASET_PATH}/${ds}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"
    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s]:\n%s\n' "$ds" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${ds}_output.%j
#SBATCH --error=${LOG_DIR}/${ds}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

${cmd}
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
}

RUN_DATASETS="${RUN_DATASETS:-all}"
for ds in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $ds "* ]] && continue
    fi
    launch_job "$ds"
done

echo ""
echo "CARL dataset-wise baseline jobs queued (up to 3 total)."
