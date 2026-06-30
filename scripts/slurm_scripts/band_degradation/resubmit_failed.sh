#!/bin/bash
#
# Resubmit failed / never-submitted band degradation jobs.
#
# Root cause of failures: node hpdar04c01s01 had a GPU initialization error
# (RuntimeError: CUDA unknown error). All scripts now exclude this node.
#
# Strategy:
#   - dofa (1 missing): targeted single job — sen1floods11 rgb_rededge.
#   - copernicus_fm_base (4 missing): targeted MADOS-only jobs.
#   - carl / dofav2 / panopticon / smarties / vit_frozen_p16: rerun full scripts.
#     Already-completed runs will get duplicate dirs; notebook picks the best result.
#   - unet / vit_finetuned: never submitted — first-time run.

set -euo pipefail

readonly SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_BASE="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/band_degradation"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

# Snapshot of active jobs — encoder scripts also fetch their own copy at invocation time
_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
_ACTIVE_JOB_IDS=$(squeue --clusters=hpda2 -u "$USER" -h --format="%i" 2>/dev/null | tr '\n' '|')

# Check if job name is in the active queue
job_in_queue() {
    echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"
}

# Check if work_dir has a truly completed (non-active) run for the given dataset+band count
has_done_run() {
    local wd=$1 ds_display=$2 n_bands=$3
    local _d _cfg _jid _dn _nb
    for _d in "${wd}"/*/; do
        [ -f "${_d}checkpoint__best.pth" ] || continue
        _cfg="${_d}configs/config.yaml"; [ -f "$_cfg" ] || continue
        _jid=$(basename "$_d" | cut -d_ -f2)
        echo "|${_ACTIVE_JOB_IDS}|" | grep -qF "|${_jid}|" && continue
        _dn=$(awk '/^  dataset_name:/{print $2; exit}' "$_cfg")
        _nb=$(grep -A 50 "input_bands:" "$_cfg" | grep -c "^    - B" || true)
        [ "$_dn" = "$ds_display" ] && [ "$_nb" = "$n_bands" ] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Helper: submit a single sbatch job inline
# Usage: submit_job <job_name> <time> <log_prefix> <cmd_string>
# ---------------------------------------------------------------------------
submit_job() {
    local -r job_name=$1
    local -r time_limit=$2
    local -r log_prefix=$3
    local -r cmd=$4
    local -r sbatch_tmp="/tmp/${job_name}_$$.sh"

    cat > "$sbatch_tmp" << EOF
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
#SBATCH --output=${log_prefix}_output.%j
#SBATCH --error=${log_prefix}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

${cmd}
EOF

    chmod +x "$sbatch_tmp"
    sbatch "$sbatch_tmp"
    rm "$sbatch_tmp"
}

# ---------------------------------------------------------------------------
# 1. TARGETED: dofa — only sen1floods11 rgb_rededge (8 bands) missing
# ---------------------------------------------------------------------------
echo "=== [1/9] DOFA — sen1floods11 rgb_rededge (1 job) ==="

readonly DOFA_LOG="${LOG_BASE}/dofa"
readonly DOFA_WORK="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/band_subset_degradation/upernet/dofa_band_degradation"
mkdir -p "${DOFA_LOG}" "${DOFA_WORK}"

if job_in_queue "dofa_sen1floods11_rgb_rededge"; then
    echo "SKIP (queued): dofa sen1floods11 rgb_rededge"
elif has_done_run "${DOFA_WORK}" "Sen1Floods11" 8; then
    echo "SKIP (done): dofa sen1floods11 rgb_rededge"
else
submit_job \
    "dofa_sen1floods11_rgb_rededge" \
    "6:00:00" \
    "${DOFA_LOG}/sen1floods11_rgb_rededge" \
    "torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=sen1floods11 \
  encoder=dofa \
  decoder=seg_upernet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=segmentation \
  dataset.root_path='${DATASET_PATH}/sen1floods11' \
  work_dir=${DOFA_WORK} \
  use_wandb=True \
  wandb_project=geofm-bench-band-degradation \
  'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A]}'"
fi

# ---------------------------------------------------------------------------
# 2. TARGETED: copernicus_fm_base — all 4 MADOS band levels missing
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/9] Copernicus FM Base — MADOS (4 jobs) ==="

readonly CFMB_LOG="${LOG_BASE}/copernicus_fm_base"
readonly CFMB_WORK="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/band_subset_degradation/upernet/copernicus_fm_base_band_degradation"
mkdir -p "${CFMB_LOG}" "${CFMB_WORK}"

declare -A cfmb_mados=(
    ["rgb_nir"]="'encoder.input_bands={optical:[B2,B3,B4,B8]}'"
    ["rgb_rededge"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A]}'"
    ["rgb_swir"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
    ["full"]="'encoder.input_bands={optical:[B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
)

for band_level in "${!cfmb_mados[@]}"; do
    _n=$(echo "${cfmb_mados[$band_level]}" | grep -oE 'B[0-9]+[A-Za-z]*' | wc -l | tr -d ' ')
    if job_in_queue "cop_fm_mados_${band_level}"; then
        echo "SKIP (queued): copernicus_fm_base mados ${band_level}"
        continue
    elif has_done_run "${CFMB_WORK}" "MADOS" "${_n}"; then
        echo "SKIP (done): copernicus_fm_base mados ${band_level}"
        continue
    fi
    submit_job \
        "cop_fm_mados_${band_level}" \
        "8:00:00" \
        "${CFMB_LOG}/mados_${band_level}" \
        "torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=mados \
  encoder=copernicus_fm_base \
  decoder=seg_upernet \
  preprocessing=seg_focus_crop \
  criterion=cross_entropy \
  task=segmentation \
  dataset.root_path='${DATASET_PATH}/mados' \
  work_dir=${CFMB_WORK} \
  use_wandb=True \
  wandb_project=geofm-bench-band-degradation \
  ${cfmb_mados[$band_level]}"
done

# ---------------------------------------------------------------------------
# 3-7. FULL RESCHEDULE: heavily-failed encoders.
#   Each encoder script now has an idempotency guard — already-queued or
#   already-completed jobs will be skipped automatically.
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/9] CARL (10/11 failed) — rerunning full script ==="
bash "${SCRIPTS_DIR}/run_carl_band_degradation.sh"

echo ""
echo "=== [4/9] DOFAv2 (9/11 failed) — rerunning full script ==="
bash "${SCRIPTS_DIR}/run_dofav2_band_degradation.sh"

echo ""
echo "=== [5/9] Panopticon (11/11 failed) — rerunning full script ==="
bash "${SCRIPTS_DIR}/run_panopticon_band_degradation.sh"

echo ""
echo "=== [6/9] SMARTIES ViT-B (10/11 failed) — rerunning full script ==="
bash "${SCRIPTS_DIR}/run_smarties_vitb_band_degradation.sh"

echo ""
echo "=== [7/9] ViT-Frozen-p16 (7/14 failed) — rerunning full script ==="
bash "${SCRIPTS_DIR}/run_vit_frozen_p16_band_degradation.sh"

# ---------------------------------------------------------------------------
# 8-9. FIRST-TIME SUBMISSION: never submitted
# ---------------------------------------------------------------------------
echo ""
echo "=== [8/9] UNet (never submitted) — first run ==="
bash "${SCRIPTS_DIR}/run_unet_band_degradation.sh"

echo ""
echo "=== [9/9] ViT-Finetuned (never submitted) — first run ==="
bash "${SCRIPTS_DIR}/run_vit_finetuned_band_degradation.sh"

echo ""
echo "All resubmit jobs queued."
echo "Monitor with: squeue -u \$USER --clusters=hpda2"
