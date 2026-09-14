#!/bin/bash

# -----------------------------------------------------------------------------
# DynamicEarthNet true-wavelength rerun
#
# dynamicen.yaml mislabeled its 4th band B5 (Vegetation Red Edge, 0.705um)
# when the underlying Planet Fusion NIR band (0.85-0.88um, centre ~0.865um)
# actually matches Sentinel-2 B8A (0.865um). dataset=dynamicen_true_wl fixes
# this (same imagery, only the band label/wavelength changes).
#
# Only wavelength-conditioned encoders are affected — DINOv2/v3, UNet and
# ViT-scratch/frozen ignore per-band wavelength metadata and don't need a
# rerun. Reruns both UPerNet (seg_upernet_mt_ltae) and MLP-Fusion
# (seg_mlp_fusion) decoders for: dofa, dofav2, panopticon, copernicus_fm_base,
# smarties_vitb, carl, carl_imagewise.
#
# Uses the same WORK_DIRs as the original benchmark scripts — run.py creates
# a fresh timestamped subfolder per run, so these land alongside (not over)
# the original B5 results for direct before/after comparison.
#
# Control via env vars:
#   RUN_ENCODERS="all"    or space-separated keys (e.g. "dofa carl_imagewise")
#   RUN_DECODERS="all"    or "upernet" / "mlp_fusion"
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dynamicen_true_wl_rerun"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly DATASET_ROOT="${DATASET_PATH}/dynamicen"   # real data dir on disk; unrelated to the dataset= config name
readonly DATASET="dynamicen_true_wl"
readonly EXPERIMENTS_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

# -----------------------------------------------------------------------------
# ENCODER DEFINITIONS
# Format: "encoder preprocessing"
# carl_imagewise uses seg_default_no_norm (CARL's internal per-image
# normalization only, no external NormalizeMeanStd) — see run_carl_seg_benchmark.sh.
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa seg_default"
    ["dofav2"]="dofav2 seg_default"
    ["panopticon"]="panopticon seg_default"
    ["copernicus_fm_base"]="copernicus_fm_base seg_default"
    ["smarties_vitb"]="smarties_vitb seg_default"
    ["carl"]="carl seg_default"
    ["carl_imagewise"]="carl_imagewise seg_default_no_norm"
)

# work_dir subdirectory name per encoder key, for both decoder families
declare -A work_dir_name=(
    ["dofa"]="dofa_seg"
    ["dofav2"]="dofav2_seg"
    ["panopticon"]="panopticon_seg"
    ["copernicus_fm_base"]="copernicus_fm_base_seg"
    ["smarties_vitb"]="smarties_vitb_seg"
    ["carl"]="carl_seg"
    ["carl_imagewise"]="carl_imagewise_seg"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r encoder_key=$1
    local -r decoder_family=$2   # "upernet" or "mlp_fusion"
    local -r decoder=$3
    local -r time_limit=$4

    read -r encoder preprocessing <<< "${encoders[$encoder_key]}"

    # upernet dirs keep the "_seg" suffix (e.g. dofa_seg); mlp_fusion dirs use
    # "_mlp_fusion" instead (e.g. dofa_mlp_fusion) — matches the existing
    # per-encoder benchmark scripts' WORK_DIR naming.
    local -r encoder_base="${work_dir_name[$encoder_key]%_seg}"
    local work_dir_leaf="${work_dir_name[$encoder_key]}"
    [ "$decoder_family" = "mlp_fusion" ] && work_dir_leaf="${encoder_base}_mlp_fusion"
    local -r work_dir="${EXPERIMENTS_ROOT}/${decoder_family}/${work_dir_leaf}"
    mkdir -p "${work_dir}"

    local -r job_name="truewl_${encoder_key}_${decoder_family}"
    local -r sbatch_script="dynamicen_true_wl_${encoder_key}_${decoder_family}.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${DATASET}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_ROOT}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"

    if [[ "$encoder_key" == carl* ]]; then
        cmd+=" task.trainer.ckpt_interval=999"
    fi

    printf '\nLaunching [%s / %s] encoder=%s preprocessing=%s:\n%s\n' \
        "$encoder_key" "$decoder_family" "$encoder" "$preprocessing" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${encoder_key}_${decoder_family}_output.%j
#SBATCH --error=${LOG_DIR}/${encoder_key}_${decoder_family}_error.%j
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

# -----------------------------------------------------------------------------
# SELECT WHAT TO RUN
# -----------------------------------------------------------------------------
RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DECODERS="${RUN_DECODERS:-all}"   # "all", "upernet", or "mlp_fusion"

for encoder_key in "${!encoders[@]}"; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $encoder_key "* ]] && continue
    fi

    if [[ "$RUN_DECODERS" == "all" || "$RUN_DECODERS" == "upernet" ]]; then
        launch_job "$encoder_key" "upernet" "seg_upernet_mt_ltae" "24:00:00"
    fi

    if [[ "$RUN_DECODERS" == "all" || "$RUN_DECODERS" == "mlp_fusion" ]]; then
        launch_job "$encoder_key" "mlp_fusion" "seg_mlp_fusion" "12:00:00"
    fi
done

echo "All jobs queued."
