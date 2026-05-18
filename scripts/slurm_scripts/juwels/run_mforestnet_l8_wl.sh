#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# mforestnet classification experiment: CARL vs DOFA with true Landsat-8
# wavelengths. Uses mforestnet_l8 dataset (bands named L8_B2…L8_B7) so each
# encoder receives the actual L8 center wavelengths instead of S2 defaults.
# No BandPadding — encoders receive only the 6 bands the dataset provides.
#
# Compare against run_mforestnet_real_wavelengths.sh which uses the same
# dataset but with S2-labeled bands (mforestnet, B2…B12 naming).
#
# CARL:  carl_mforestnet_l8 encoder config supplies L8 wavelengths via
#        band_wavelength_overrides; set_wv_list_from_data_bands() restricts
#        wv_list to the 6 L8 bands at runtime.
#        Preprocessing: cls_resize_no_norm_no_pad
#
# DOFA:  wave_list already contains L8_B* entries; set_wv_list_from_data_bands()
#        restricts wv_list to these 6 bands at runtime.
#        Preprocessing: cls_resize_norm_no_pad
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/mforestnet_l8_wl"
readonly WORK_DIR="/p/project1/hai_1280/experiments/mforestnet_l8_wl"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r encoder=$1
    local -r preprocessing=$2

    local -r job_name="mforestnet_l8_${encoder}"
    local -r sbatch_script="${encoder}_mforestnet_l8.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=mforestnet_l8"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=linear_classification"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=16"
    cmd+=" finetune=false"
    cmd+=" work_dir=${WORK_DIR}/${encoder}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s | preprocessing=%s]:\n%s\n' "$encoder" "$preprocessing" "$cmd"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=hai_1280
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=${LOG_DIR}/${encoder}_%j.out
#SBATCH --error=${LOG_DIR}/${encoder}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"
export MAMBA_ROOT_PREFIX="/p/project1/hai_1280/micromamba"
export WANDB_MODE=offline
export WANDB_PROJECT="Juwels"
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH}"
export HF_HOME="/p/project1/hai_1280/hf_cache"

eval "\$(${MICROMAMBA} shell hook --shell bash)"
micromamba activate ${ENV_PATH}

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm -f "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
launch_job "carl_mforestnet_l8" "cls_resize_no_norm_no_pad"
launch_job "dofa" "cls_resize_norm_no_pad"

echo "All jobs queued."
