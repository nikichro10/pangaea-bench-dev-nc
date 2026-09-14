#!/bin/bash

# -----------------------------------------------------------------------------
# SpaceNet7-CD dice-collapse fix -- validation run
#
# The regular MLP-Fusion benchmark on SpaceNet7-CD (criterion=dice) collapses
# to predicting "No Change" everywhere after ~10-20 epochs, for every encoder
# checked (DOFA, DOFAv2, Panopticon, Copernicus-FM, SMARTIES, CARL-Imagewise,
# DINOv3): val mIoU jumps to a constant 49.684 and never recovers. Root cause:
# DICELoss (pangaea/utils/losses.py) rewards predicting nothing on the many
# "Change"-free patches (empty-vs-empty dice ~= 1.0), which dominates under
# SpaceNet7-CD's extreme class imbalance (measured: only 0.63% of pixels are
# "Change", not the stale 7.47% previously in configs/dataset/spacenet7cd.yaml).
#
# Same collapse class already solved for AI4SmallFarms via weighted_bce +
# best_metric_key=IoU + data_replicate=2 (see run_carl_seg_benchmark.sh).
# This script applies the identical recipe to SpaceNet7-CD, now that
# spacenet7cd.yaml's `distribution` has been corrected to the measured
# [0.99366, 0.00634] (was previously an unverified/wrong [0.9253, 0.0747]
# placeholder, flagged "TODO: update for CD").
#
# Validates: 3 encoders x MLP-Fusion (where collapse happened) + DOFA x UPerNet
# (sanity check the fix doesn't hurt the decoder that was already stable).
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated keys (e.g. "dofa_mlp panopticon_mlp")
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/spacenet7cd_weighted_bce_fix"
readonly DATASET_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/spacenet7cd"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/spacenet7cd_weighted_bce_fix"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

# job_key -> "encoder|decoder|time_limit"
declare -A jobs=(
    ["dofa_mlp"]="dofa|seg_siam_mlp_fusion|4:00:00"
    ["panopticon_mlp"]="panopticon|seg_siam_mlp_fusion|4:00:00"
    ["smarties_mlp"]="smarties_vitb|seg_siam_mlp_fusion|4:00:00"
    ["dofa_upernet"]="dofa|seg_siamupernet_conc|8:00:00"
)

launch_job() {
    local -r job_key=$1
    IFS='|' read -r encoder decoder time_limit <<< "${jobs[$job_key]}"

    local -r work_dir="${WORK_BASE}/${job_key}"
    mkdir -p "${work_dir}"

    local -r job_name="sn7fix_${job_key}"
    local -r sbatch_script="spacenet7cd_fix_${job_key}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=spacenet7cd"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=seg_default"
    cmd+=" criterion=weighted_bce"
    cmd+=" task=change_detection"
    cmd+=" dataset.root_path='${DATASET_ROOT}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.best_metric_key=IoU"
    cmd+=" data_replicate=2"

    printf '\nLaunching [%s] encoder=%s decoder=%s:\n%s\n' "$job_key" "$encoder" "$decoder" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${job_key}_output.%j
#SBATCH --error=${LOG_DIR}/${job_key}_error.%j
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
for job_key in "${!jobs[@]}"; do
    if [ "$RUN_JOBS" != "all" ]; then
        [[ " $RUN_JOBS " != *" $job_key "* ]] && continue
    fi
    launch_job "$job_key"
done

echo "All jobs queued."
