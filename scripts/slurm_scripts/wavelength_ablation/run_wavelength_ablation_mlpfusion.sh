#!/bin/bash
#
# Wavelength Ablation (MLP-Fusion decoder): S2 band names vs. true sensor wavelengths
#
# Identical setup to run_wavelength_ablation.sh but uses MLP-Fusion decoders:
#   seg_siam_mlp_fusion  (SN7-CD change detection)
#   seg_mlp_fusion       (FiveBillionPixels segmentation)
#
# Encoders: DOFA · DOFAv2 · Panopticon · SMARTIES · Copernicus-FM
#           CARL (dataset-wise norm) · CARL (imagewise norm)
#
# Datasets / true sensor:
#   SpaceNet7-CD      — PlanetScope PS2  (RGB 3-band), binary change detection
#   FiveBillionPixels — GaoFen-2 PMS     (NIR+RGB 4-band), 25-class segmentation
#
# Per encoder × dataset: tw (true-wavelength) runs only.
# S2 baselines come from the regular benchmark.
#
# True-WL: 5 encoders × 2 datasets   = 10
#        + 2 CARL norms × 2 datasets = 4
#                                      ──
#                                      14

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_BASE="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/wavelength_ablation_mlpfusion"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/wavelength_ablation_mlpfusion"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_BASE}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# ---------------------------------------------------------------------------
# launch_job <enc_short> <enc_cfg> <ds_short> <ds_cfg>
#            <wl_variant: s2|tw> <task> <decoder> <preprocessing>
#            <criterion> <root_path> <time_limit> [extra hydra args...]
# ---------------------------------------------------------------------------
launch_job() {
    local -r enc_short=$1   # used in job name + work_dir (e.g. carl_dw)
    local -r enc_cfg=$2     # hydra encoder key            (e.g. carl)
    local -r ds_short=$3    # used in job name             (e.g. sn7cd)
    local -r ds_cfg=$4      # hydra dataset key            (e.g. spacenet7cd_true_wl)
    local -r wl_variant=$5  # s2 or tw
    local -r task=$6
    local -r decoder=$7
    local -r preprocessing=$8
    local -r criterion=$9
    local -r root_path="${10}"
    local -r time_limit="${11}"
    local -ra extra_args=("${@:12}")

    local -r job_name="${enc_short}_${ds_short}_${wl_variant}_mlp"
    local -r work_dir="${WORK_BASE}/${enc_short}_wl_ablation"
    local -r log_prefix="${LOG_BASE}/${enc_short}_${ds_short}_${wl_variant}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "${work_dir}"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-10s | %-3s | %s]\n' "$enc_short" "$ds_short" "$wl_variant"
        return 0
    fi
    # Check for completed run (requires [test] block in log, not just a checkpoint)
    local _d _cfg
    for _d in "${work_dir}"/*/; do
        _cfg="${_d}configs/config.yaml"
        [ -f "${_d}checkpoint__best.pth" ] && [ -f "$_cfg" ] || continue
        grep -q "\[test\]" "${_d}train.log-0" 2>/dev/null || continue
        local _ds_name
        _ds_name=$(awk '/^dataset_name:/{print $2; exit}' \
            "/dss/dsshome1/08/di54zop/pangaea-bench_dev/configs/dataset/${ds_cfg}.yaml" 2>/dev/null)
        grep -q "dataset_name: ${_ds_name}" "$_cfg" || continue
        # true-wl configs contain PS_/GF2_/L8_B band names; s2 configs do not
        local _has_true_wl
        _has_true_wl=$(grep -cE "PS_|GF2_|L8_B" "$_cfg" || true)
        [ "$wl_variant" = "tw" ] && [ "$_has_true_wl" -gt 0 ] && {
            printf 'SKIP (done)    [%-10s | %-3s | %s]\n' "$enc_short" "$ds_short" "$wl_variant"
            return 0
        }
        [ "$wl_variant" = "s2" ] && [ "$_has_true_wl" -eq 0 ] && {
            printf 'SKIP (done)    [%-10s | %-3s | %s]\n' "$enc_short" "$ds_short" "$wl_variant"
            return 0
        }
    done

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${ds_cfg}"
    cmd+=" encoder=${enc_cfg}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" finetune=false"
    cmd+=" dataset.root_path='${root_path}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench-wavelength-ablation-mlpfusion"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%-10s | %-3s | %s]:\n%s\n' "$enc_short" "$ds_short" "$wl_variant" "$cmd"

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
#SBATCH --output=${log_prefix}_output.%j
#SBATCH --error=${log_prefix}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR=${GEO_BENCH_DATA}

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

# ===========================================================================
# Helper: submit true-wl variant for one encoder × dataset combo
# Usage: submit_pair <enc_short> <enc_cfg> <ds_short> <ds_s2_cfg> <ds_tw_cfg>
#                    <task> <decoder> <preprocessing> <criterion>
#                    <root_path> <time_limit> [extra hydra args...]
# ===========================================================================
submit_pair() {
    local -r enc_short=$1 enc_cfg=$2 ds_short=$3 ds_s2=$4 ds_tw=$5
    local -r task=$6 decoder=$7 prep=$8 criterion=$9 root="${10}" tlimit="${11}"
    local -ra xargs=("${@:12}")

    launch_job "$enc_short" "$enc_cfg" "$ds_short" "$ds_tw" tw \
        "$task" "$decoder" "$prep" "$criterion" "$root" "$tlimit" "${xargs[@]}"
}

readonly SN7_PATH="${DATASET_PATH}/spacenet7"
readonly FBP_PATH="${DATASET_PATH}/fivebillionpixels"

# ===========================================================================
# Standard wavelength-aware encoders (no normalization variants)
# ===========================================================================
for ENC in dofa dofav2 panopticon smarties_vitb copernicus_fm_base; do
    # shorten copernicus and smarties for job names / path
    case "$ENC" in
        copernicus_fm_base) SHORT="cfm"      ;;
        smarties_vitb)      SHORT="sm"       ;;
        *)                  SHORT="$ENC"     ;;
    esac

    echo ""
    echo "=== ${ENC} ==="

    # SpaceNet7-CD — binary change detection
    submit_pair "$SHORT" "$ENC" sn7cd \
        spacenet7cd spacenet7cd_true_wl \
        change_detection seg_siam_mlp_fusion seg_default binary_cross_entropy \
        "$SN7_PATH" 8:00:00

    # FiveBillionPixels — 25-class segmentation
    submit_pair "$SHORT" "$ENC" fbp \
        fivebillionpixels fivebillionpixels_true_wl \
        segmentation seg_mlp_fusion seg_default cross_entropy \
        "$FBP_PATH" 12:00:00
done

# ===========================================================================
# CARL — dataset-wise normalization (image_norm=false, default)
# Preprocessing: seg_default (global NormalizeMeanStd) / cls_geobench_no_pad
# ===========================================================================
echo ""
echo "=== CARL — dataset-wise normalization ==="

submit_pair carl_dw carl sn7cd \
    spacenet7cd spacenet7cd_true_wl \
    change_detection seg_siam_mlp_fusion seg_default binary_cross_entropy \
    "$SN7_PATH" 8:00:00

submit_pair carl_dw carl fbp \
    fivebillionpixels fivebillionpixels_true_wl \
    segmentation seg_mlp_fusion seg_default cross_entropy \
    "$FBP_PATH" 16:00:00

# ===========================================================================
# CARL — imagewise normalization (image_norm=true, no external normalization)
# Preprocessing: seg_default_no_norm / cls_geobench_imagewise_no_pad
# ===========================================================================
echo ""
echo "=== CARL — imagewise normalization ==="

submit_pair carl_iw carl sn7cd \
    spacenet7cd spacenet7cd_true_wl \
    change_detection seg_siam_mlp_fusion seg_default_no_norm binary_cross_entropy \
    "$SN7_PATH" 8:00:00 \
    encoder.image_norm=true

submit_pair carl_iw carl fbp \
    fivebillionpixels fivebillionpixels_true_wl \
    segmentation seg_mlp_fusion seg_default_no_norm cross_entropy \
    "$FBP_PATH" 16:00:00 \
    encoder.image_norm=true

echo ""
echo "All wavelength ablation (MLP-Fusion) jobs queued (14 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
