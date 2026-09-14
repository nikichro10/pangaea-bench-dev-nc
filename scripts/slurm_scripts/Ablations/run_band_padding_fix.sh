#!/bin/bash
#
# Band-padding ablation fix -- SpaceNet7-CD (dice collapse) + DynamicEarthNet (B5/B8A)
#
# Two independent fixes bundled into one rerun of the band-padding ablation:
#   - SpaceNet7-CD: criterion=dice collapsed under the padded-band setup just
#     like the main benchmark did. Switched to weighted_bce_sqrt +
#     best_metric_key=IoU + data_replicate=2.
#   - DynamicEarthNet: used dataset=dynamicen (mislabeled 4th band as B5,
#     Vegetation Red Edge) instead of dynamicen_true_wl (correct B8A, true
#     NIR). All 6 encoders here are wavelength-aware *_s2full configs, so all
#     are affected. criterion stays cross_entropy (never broken).
#
# Encoders: DOFA, DOFAv2, CARL (carl_no_internal_norm), Panopticon,
#           SMARTIES, Copernicus-FM -- same *_s2full / no-special-config set
#           as the original band-padding ablation.
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated "<encoder>_<dataset>" keys
#                     (e.g. "dofa_sn7 carl_dynamicen")
# -----------------------------------------------------------------------------

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/band_padding_fix"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/ablations/band_padding"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

declare -A encoders=(
    ["dofa"]="dofa_s2full"
    ["dofav2"]="dofav2_s2full"
    ["carl"]="carl_no_internal_norm"
    ["panopticon"]="panopticon_s2full"
    ["smarties_vitb"]="smarties_vitb_s2full"
    ["copernicus_fm_base"]="copernicus_fm_base_s2full"
)

launch_job() {
    local -r encoder=$1
    local -r ds_key=$2   # "sn7" or "dynamicen"

    local -r encoder_cfg="${encoders[$encoder]}"
    local -r work_dir="${WORK_BASE}/${encoder}_band_pad"
    mkdir -p "${work_dir}"

    local dataset_cfg root_path decoder task criterion time_limit extra_args=""
    if [ "$ds_key" = "sn7" ]; then
        dataset_cfg="spacenet7cd"
        root_path="${DATASET_PATH}/spacenet7cd"
        decoder="seg_siamupernet_conc"
        task="change_detection"
        criterion="weighted_bce_sqrt"
        time_limit="8:00:00"
        extra_args="task.trainer.best_metric_key=IoU data_replicate=2"
    else
        dataset_cfg="dynamicen_true_wl"
        root_path="${DATASET_PATH}/dynamicen"
        decoder="seg_upernet_mt_ltae"
        task="segmentation"
        criterion="cross_entropy"
        time_limit="24:00:00"
    fi

    local -r job_name="pad_${encoder:0:6}_${ds_key}fix"
    local -r sbatch_script="${encoder}_${ds_key}_padfix.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset_cfg}"
    cmd+=" encoder=${encoder_cfg}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=seg_default_pad"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${root_path}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    [ -n "$extra_args" ] && cmd+=" ${extra_args}"

    printf '\nLaunching [%s / %s] encoder=%s dataset=%s:\n%s\n' \
        "$encoder" "$ds_key" "$encoder_cfg" "$dataset_cfg" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${encoder}_${ds_key}_output.%j
#SBATCH --error=${LOG_DIR}/${encoder}_${ds_key}_error.%j
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
    for ds_key in sn7 dynamicen; do
        combo_key="${encoder}_${ds_key}"
        if [ "$RUN_JOBS" != "all" ]; then
            [[ " $RUN_JOBS " != *" $combo_key "* ]] && continue
        fi
        launch_job "$encoder" "$ds_key"
    done
done

echo "All jobs queued."
