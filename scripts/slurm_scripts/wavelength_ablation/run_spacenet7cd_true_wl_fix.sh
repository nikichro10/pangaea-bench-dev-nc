#!/bin/bash
#
# SpaceNet7-CD True-Wavelength rerun with the dice-collapse fix
#
# Companion to run_wavelength_ablation.sh / run_wavelength_ablation_mlpfusion.sh,
# scoped to SpaceNet7-CD only. The original true-WL runs used
# criterion=binary_cross_entropy against a dataset config whose `distribution`
# was a stale/wrong placeholder ([0.9253, 0.0747] instead of the measured
# [0.99366, 0.00634]) -- the same issue that caused the regular SpaceNet7-CD
# S2-default benchmark to collapse under `dice`. Both spacenet7cd.yaml and
# spacenet7cd_true_wl.yaml have since been corrected; this reruns the
# True-WL side with criterion=weighted_bce_sqrt + best_metric_key=IoU +
# data_replicate=2, matching the fix already validated and rolled out for the
# S2-default SpaceNet7-CD benchmark. S2-default baselines do NOT need
# rerunning here -- those already got the fix in the regular benchmark rerun.
#
# Encoders: DOFA, DOFAv2, Panopticon, SMARTIES, Copernicus-FM,
#           CARL (dataset-wise norm), CARL (imagewise norm)
# Decoders: UPerNet (seg_siamupernet_conc) + MLP-Fusion (seg_siam_mlp_fusion)
# finetune=false (frozen encoder), matching the original ablation.
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated keys (e.g. "dofa_up carl_dw_mlp")
# -----------------------------------------------------------------------------

readonly SN7_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/spacenet7"
readonly LOG_BASE="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/wavelength_ablation_sn7_fix"
readonly WORK_BASE_UP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/wavelength_ablation"
readonly WORK_BASE_MLP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/wavelength_ablation_mlpfusion"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_BASE}"

# job_key -> "enc_short|enc_cfg|preprocessing|extra_args"
declare -A jobs=(
    ["dofa"]="dofa|dofa|seg_default|"
    ["dofav2"]="dofav2|dofav2|seg_default|"
    ["panopticon"]="panopticon|panopticon|seg_default|"
    ["sm"]="smarties_vitb|smarties_vitb|seg_default|"
    ["cfm"]="copernicus_fm_base|copernicus_fm_base|seg_default|"
    ["carl_dw"]="carl_dw|carl|seg_default|"
    ["carl_iw"]="carl_iw|carl|seg_default_no_norm|encoder.image_norm=true"
)

launch_job() {
    local -r job_key=$1
    local -r decoder_family=$2   # "upernet" or "mlp_fusion"

    IFS='|' read -r enc_short enc_cfg preprocessing extra_str <<< "${jobs[$job_key]}"
    local -ra extra_args=($extra_str)

    local decoder time_limit work_base
    if [ "$decoder_family" = "upernet" ]; then
        decoder="seg_siamupernet_conc"; time_limit="8:00:00"; work_base="$WORK_BASE_UP"
    else
        decoder="seg_siam_mlp_fusion"; time_limit="8:00:00"; work_base="$WORK_BASE_MLP"
    fi

    local -r work_dir="${work_base}/${enc_short}_wl_ablation"
    mkdir -p "${work_dir}"

    local -r job_name="${job_key}_sn7fix_${decoder_family}"
    local -r sbatch_script="sn7_true_wl_fix_${job_key}_${decoder_family}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=spacenet7cd_true_wl"
    cmd+=" encoder=${enc_cfg}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=weighted_bce_sqrt"
    cmd+=" task=change_detection"
    cmd+=" finetune=false"
    cmd+=" dataset.root_path='${SN7_PATH}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench-wavelength-ablation"
    cmd+=" task.trainer.best_metric_key=IoU"
    cmd+=" data_replicate=2"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s / %s] encoder=%s preprocessing=%s:\n%s\n' \
        "$job_key" "$decoder_family" "$enc_cfg" "$preprocessing" "$cmd"

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
#SBATCH --output=${LOG_BASE}/${job_key}_${decoder_family}_output.%j
#SBATCH --error=${LOG_BASE}/${job_key}_${decoder_family}_error.%j
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
    for decoder_family in upernet mlp_fusion; do
        combo_key="${job_key}_${decoder_family}"
        if [ "$RUN_JOBS" != "all" ]; then
            [[ " $RUN_JOBS " != *" $combo_key "* ]] && continue
        fi
        launch_job "$job_key" "$decoder_family"
    done
done

echo "All jobs queued."
