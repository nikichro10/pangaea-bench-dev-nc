#!/bin/bash
#
# Constant Wavelength Ablation: all S2 bands set to 550 nm (visible green)
#
# Tests whether wavelength conditioning actually matters by replacing true
# S2 band wavelengths with a single uninformative constant.
#
# Encoders: DOFA · DOFAv2 · Panopticon · SMARTIES · Copernicus-FM
#           CARL (dataset-wise norm) · CARL (imagewise norm)
#
# Datasets (S2 or S2-named bands, single-temporal):
#   HLSBurnScars        — 6 S2 bands (B2,B3,B4,B8A,B11,B12), binary fire-scar seg
#   MADOS               — 11 S2 bands (B1–B8,B8A,B11,B12), 15-class marine debris seg
#   GeoBench:
#   m-cashew-plantation — 12 S2 bands (B1–B9,B11,B12), 7-class seg
#   m-SA-crop-type      — 12 S2 bands (B1–B9,B11,B12), 10-class seg
#   m-forestnet         — 6 S2-named bands (B2,B3,B4,B8,B11,B12), 12-class linear cls
#
# Decoders: UPerNet + MLP-Fusion for seg; cls_linear for m-forestnet.
# Encoder frozen (finetune=false) — only the wavelength vector changes.
#
# Per-encoder const-WL implementation:
#   DOFA/DOFAv2   — encoder.wave_list.optical.{band}=0.55  (µm)
#   Panopticon    — encoder.wave_list.optical.{band}=550    (nm)
#   Copernicus-FM — encoder.wave_list.optical.{band}=550    (nm)
#   SMARTIES      — encoder/smarties_vitb_const_wl (proj_idx 4 for all bands)
#   CARL          — encoder/carl_const_wl or carl_imagewise_const_wl
#
# Total: 7 encoders × (4 seg datasets × 2 decoders + 1 cls dataset) = 63 runs
# S2 baselines come from the regular benchmark.

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_BASE="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/const_wl_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/wavelength_ablation_const_wl"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_BASE}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# ---------------------------------------------------------------------------
# Per-encoder constant-wavelength override arrays
# All 13 S2 optical bands overridden; only bands present in the dataset are
# actually used by the encoder — the rest are looked up but ignored.
# ---------------------------------------------------------------------------

# DOFA / DOFAv2: wavelengths in µm → 0.55
CWL_UM=(
    encoder.wave_list.optical.B1=0.55
    encoder.wave_list.optical.B2=0.55
    encoder.wave_list.optical.B3=0.55
    encoder.wave_list.optical.B4=0.55
    encoder.wave_list.optical.B5=0.55
    encoder.wave_list.optical.B6=0.55
    encoder.wave_list.optical.B7=0.55
    encoder.wave_list.optical.B8=0.55
    encoder.wave_list.optical.B8A=0.55
    encoder.wave_list.optical.B9=0.55
    encoder.wave_list.optical.B10=0.55
    encoder.wave_list.optical.B11=0.55
    encoder.wave_list.optical.B12=0.55
)

# Panopticon / Copernicus-FM: wavelengths in nm → 550
CWL_NM=(
    encoder.wave_list.optical.B1=550
    encoder.wave_list.optical.B2=550
    encoder.wave_list.optical.B3=550
    encoder.wave_list.optical.B4=550
    encoder.wave_list.optical.B5=550
    encoder.wave_list.optical.B6=550
    encoder.wave_list.optical.B7=550
    encoder.wave_list.optical.B8=550
    encoder.wave_list.optical.B8A=550
    encoder.wave_list.optical.B9=550
    encoder.wave_list.optical.B10=550
    encoder.wave_list.optical.B11=550
    encoder.wave_list.optical.B12=550
)

# ---------------------------------------------------------------------------
# launch_job <enc_short> <enc_cfg> <ds_short> <ds_cfg>
#            <task> <decoder> <preprocessing> <criterion>
#            <root_path> <time_limit> <dec_type: up|mlp>
#            [extra hydra args...]
# ---------------------------------------------------------------------------
launch_job() {
    local -r enc_short=$1
    local -r enc_cfg=$2
    local -r ds_short=$3
    local -r ds_cfg=$4
    local -r task=$5
    local -r decoder=$6
    local -r preprocessing=$7
    local -r criterion=$8
    local -r root_path=$9
    local -r time_limit="${10}"
    local -r dec_type="${11}"   # up or mlp (used in job name / work_dir suffix)
    local -ra extra_args=("${@:12}")

    local -r job_name="${enc_short}_${ds_short}_cwl_${dec_type}"
    local -r work_dir="${WORK_BASE}/${enc_short}_const_wl_${dec_type}"
    local -r log_prefix="${LOG_BASE}/${enc_short}_${ds_short}_cwl_${dec_type}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "${work_dir}"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-10s | %-5s | cwl_%s]\n' "$enc_short" "$ds_short" "$dec_type"
        return 0
    fi

    # Check for a completed run (checkpoint + [test] in log + dataset match)
    local _d _cfg
    for _d in "${work_dir}"/*/; do
        _cfg="${_d}configs/config.yaml"
        [ -f "${_d}checkpoint__best.pth" ] && [ -f "$_cfg" ] || continue
        grep -q "\[test\]" "${_d}train.log-0" 2>/dev/null || continue
        local _ds_name
        _ds_name=$(awk '/^dataset_name:/{print $2; exit}' \
            "/dss/dsshome1/08/di54zop/pangaea-bench_dev/configs/dataset/${ds_cfg}.yaml" 2>/dev/null)
        grep -q "dataset_name: ${_ds_name}" "$_cfg" && {
            printf 'SKIP (done)    [%-10s | %-5s | cwl_%s]\n' "$enc_short" "$ds_short" "$dec_type"
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
    cmd+=" wandb_project=wavelength_augmentations"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%-10s | %-5s | cwl_%s]:\n%s\n' "$enc_short" "$ds_short" "$dec_type" "$cmd"

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

# ---------------------------------------------------------------------------
# submit_pair: submit both UPerNet and MLP-Fusion for one encoder × dataset
# Usage: submit_pair <enc_short> <enc_cfg> <ds_short> <ds_cfg>
#                    <task> <prep_default> <prep_no_norm> <criterion>
#                    <root_path> <time_up> <time_mlp> [const_wl_args...]
# ---------------------------------------------------------------------------
submit_pair() {
    local -r enc_short=$1 enc_cfg=$2 ds_short=$3 ds_cfg=$4
    local -r task=$5 prep=$6 criterion=$7 root=$8
    local -r time_up=$9 time_mlp="${10}"
    local -ra cwl_args=("${@:11}")

    launch_job "$enc_short" "$enc_cfg" "$ds_short" "$ds_cfg" \
        "$task" seg_upernet "$prep" "$criterion" "$root" "$time_up" up \
        "${cwl_args[@]}"

    launch_job "$enc_short" "$enc_cfg" "$ds_short" "$ds_cfg" \
        "$task" seg_mlp_fusion "$prep" "$criterion" "$root" "$time_mlp" mlp \
        "${cwl_args[@]}"
}

# ---------------------------------------------------------------------------
# submit_cls: submit cls_linear for one encoder × dataset (linear classification)
# Usage: submit_cls <enc_short> <enc_cfg> <ds_short> <ds_cfg>
#                   <prep> <root_path> <time_limit> [const_wl_args...]
# ---------------------------------------------------------------------------
submit_cls() {
    local -r enc_short=$1 enc_cfg=$2 ds_short=$3 ds_cfg=$4
    local -r prep=$5 root=$6 tlimit=$7
    local -ra cwl_args=("${@:8}")

    launch_job "$enc_short" "$enc_cfg" "$ds_short" "$ds_cfg" \
        linear_classification cls_linear "$prep" cross_entropy "$root" "$tlimit" cls \
        batch_size=32 "${cwl_args[@]}"
}

readonly HLS_PATH="${DATASET_PATH}/hlsburnscars"
readonly MADOS_PATH="${DATASET_PATH}/mados"
readonly CASHEW_PATH="${GEO_BENCH_DATA}/segmentation_v1.0/m-cashew-plant"
readonly SACROP_PATH="${GEO_BENCH_DATA}/segmentation_v1.0/m-SA-crop-type"
readonly MFN_PATH="${GEO_BENCH_DATA}/classification_v1.0/m-forestnet"

# ===========================================================================
# DOFA — wave_list overrides in µm
# ===========================================================================
echo ""
echo "=== DOFA ==="

submit_pair dofa dofa hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair dofa dofa mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 10:00:00 10:00:00 \
    "${CWL_UM[@]}"

# ===========================================================================
# DOFAv2 — same µm convention
# ===========================================================================
echo ""
echo "=== DOFAv2 ==="

submit_pair dofav2 dofav2 hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair dofav2 dofav2 mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 10:00:00 10:00:00 \
    "${CWL_UM[@]}"

# ===========================================================================
# Panopticon — nm convention
# ===========================================================================
echo ""
echo "=== Panopticon ==="

submit_pair panopticon panopticon hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair panopticon panopticon mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 10:00:00 10:00:00 \
    "${CWL_NM[@]}"

# ===========================================================================
# SMARTIES — all S2 bands → projector 4 (green_2 ≈ 550 nm)
#            handled via smarties_vitb_const_wl.yaml (no wave_list CLI args)
# ===========================================================================
echo ""
echo "=== SMARTIES ==="

submit_pair sm smarties_vitb_const_wl hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 6:00:00 6:00:00

submit_pair sm smarties_vitb_const_wl mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 10:00:00 10:00:00

# ===========================================================================
# Copernicus-FM — nm convention
# ===========================================================================
echo ""
echo "=== Copernicus-FM ==="

submit_pair cfm copernicus_fm_base hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair cfm copernicus_fm_base mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 10:00:00 10:00:00 \
    "${CWL_NM[@]}"

# ===========================================================================
# CARL — dataset-wise normalization (image_norm=false)
#        const-WL via carl_const_wl.yaml; preprocessing includes NormalizeMeanStd
# ===========================================================================
echo ""
echo "=== CARL — dataset-wise normalization ==="

submit_pair carl_dw carl_const_wl hls hlsburnscars \
    segmentation seg_default cross_entropy "$HLS_PATH" 8:00:00 8:00:00

submit_pair carl_dw carl_const_wl mados mados \
    segmentation seg_focus_crop cross_entropy "$MADOS_PATH" 12:00:00 12:00:00

# ===========================================================================
# CARL — imagewise normalization (image_norm=true, no external NormalizeMeanStd)
#        const-WL via carl_imagewise_const_wl.yaml
# ===========================================================================
echo ""
echo "=== CARL — imagewise normalization ==="

submit_pair carl_iw carl_imagewise_const_wl hls hlsburnscars \
    segmentation seg_default_no_norm cross_entropy "$HLS_PATH" 8:00:00 8:00:00

submit_pair carl_iw carl_imagewise_const_wl mados mados \
    segmentation seg_focus_crop_no_norm cross_entropy "$MADOS_PATH" 12:00:00 12:00:00

# ===========================================================================
# GeoBench segmentation: m-cashew-plantation  (12 S2 bands, 7-class, seg_default)
# ===========================================================================
echo ""
echo "=== m-cashew-plantation ==="

submit_pair dofa dofa csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair dofav2 dofav2 csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair panopticon panopticon csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair sm smarties_vitb_const_wl csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 6:00:00 6:00:00

submit_pair cfm copernicus_fm_base csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair carl_dw carl_const_wl csh mcashew-plantation \
    segmentation seg_default cross_entropy "$CASHEW_PATH" 8:00:00 8:00:00

submit_pair carl_iw carl_imagewise_const_wl csh mcashew-plantation \
    segmentation seg_default_no_norm cross_entropy "$CASHEW_PATH" 8:00:00 8:00:00

# ===========================================================================
# GeoBench segmentation: m-SA-crop-type  (12 S2 bands, 10-class, seg_default)
# ===========================================================================
echo ""
echo "=== m-SA-crop-type ==="

submit_pair dofa dofa sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair dofav2 dofav2 sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 6:00:00 6:00:00 \
    "${CWL_UM[@]}"

submit_pair panopticon panopticon sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair sm smarties_vitb_const_wl sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 6:00:00 6:00:00

submit_pair cfm copernicus_fm_base sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 6:00:00 6:00:00 \
    "${CWL_NM[@]}"

submit_pair carl_dw carl_const_wl sact msa-crop-type \
    segmentation seg_default cross_entropy "$SACROP_PATH" 8:00:00 8:00:00

submit_pair carl_iw carl_imagewise_const_wl sact msa-crop-type \
    segmentation seg_default_no_norm cross_entropy "$SACROP_PATH" 8:00:00 8:00:00

# ===========================================================================
# GeoBench classification: m-forestnet  (6 S2-named bands, 12-class, cls_linear)
# ===========================================================================
echo ""
echo "=== m-forestnet ==="

submit_cls dofa dofa mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00 \
    "${CWL_UM[@]}"

submit_cls dofav2 dofav2 mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00 \
    "${CWL_UM[@]}"

submit_cls panopticon panopticon mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00 \
    "${CWL_NM[@]}"

submit_cls sm smarties_vitb_const_wl mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00

submit_cls cfm copernicus_fm_base mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00 \
    "${CWL_NM[@]}"

submit_cls carl_dw carl_const_wl mfn mforestnet \
    cls_geobench_no_pad "$MFN_PATH" 6:00:00

submit_cls carl_iw carl_imagewise_const_wl mfn mforestnet \
    cls_geobench_imagewise_no_pad "$MFN_PATH" 6:00:00

echo ""
echo "All constant-wavelength ablation jobs queued (63 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
