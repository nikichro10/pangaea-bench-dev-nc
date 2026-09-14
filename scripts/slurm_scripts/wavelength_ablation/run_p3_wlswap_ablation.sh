#!/bin/bash
#
# P3 -- Full wavelength-vector reversal ablation.
#
# The most severe wavelength-label perturbation in the P1/P2/P3 ladder: the
# first band's channel gets the last band's wavelength label, the second gets
# the second-to-last, and so on inward. No randomness, no seeds -- one fixed
# config per dataset. As with P1/P2, pixel data and normalization stay
# completely untouched at their original channel position; only the
# wavelength/proj_idx label fed to each encoder's weight generator moves.
#
# Reversal pairs (index-reversal of each dataset's *actual* band vector --
# see note below on the 12- vs 13-band discrepancy):
#
#   MADOS (11 bands, no B9/B10):
#     B1(443)<->B12(2190)  B2(490)<->B11(1610)  B3(560)<->B8A(865)
#     B4(665)<->B8(842)    B5(705)<->B7(783)    B6(740) fixed (middle of 11)
#
#   m-Cashew-Plantation / m-SA-Crop-Type (12 bands: B1-B9,B11,B12 -- NO B10):
#     NOTE: the originally proposed pairing assumed a 13-band vector
#     (B1<->B12 ... B7 fixed) including B10, which these datasets do not
#     have. Reversing the *true* 12-band vector instead (confirmed against
#     configs/dataset/mcashew-plantation.yaml and msa-crop-type.yaml) gives
#     an even-length vector, so every band pairs -- none held fixed:
#     B1(443)<->B12(2190)  B2(490)<->B11(1610)  B3(560)<->B9(945)
#     B4(665)<->B8A(865)   B5(705)<->B8(842)    B6(740)<->B7(783)
#
#   m-Forestnet (6 bands, S2-aliased Landsat-8: B2,B3,B4,B8,B11,B12):
#     blue(B2)<->SWIR2(B12)  green(B3)<->SWIR1(B11)  red(B4)<->NIR(B8)
#     (6 is even -- all-cross-region, no fixed band; this is the "cleaner"
#     P3 variant since MADOS/S2 12-band still has 2 within-region pairs:
#     B5<->B8/B8A(mados)/B5<->B8(12-band) sit inside the red-edge/NIR block,
#     Delta ~100-160nm, so S2-P3 severity is concentrated in the VIS<->SWIR
#     swaps rather than being purely cross-region for every pair)
#
# If this reversal result is ambiguous, the documented fallback is a
# block-swap variant (VIS<->SWIR labels swapped wholesale, red-edge internal
# order kept) -- not implemented here; only the primary reversal.
#
# Per-encoder implementation (same mechanism as P1/P2):
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on
#     encoder.wave_list.optical.{band} (keys already exist, no '+').
#   CARL     -- encoder.band_wavelength_overrides.optical.{band} (µm), '+' for new keys.
#   SMARTIES -- encoder.proj_idx_override.optical.{band} (discrete projector index), '+' for new keys.
#
# All encoders frozen (finetune=false). W&B project: wavelength-ablation
# (same project as P1/P2; runs distinguished by job/run name, prefix p3wl_).
#
# 6 encoders x 4 datasets = 24 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" bash run_p3_wlswap_ablation.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p3_wlswap_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p3_wlswap_ablation"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <dsgroup: mados|s2_12b|mfn>
# Echoes the Hydra CLI overrides for one encoder's full vector reversal.
# Values are hardcoded (generated offline from each encoder's own native
# wave_list/proj_idx table -- see header for the reversal pairing).
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 dsgroup=$2
    local args=""

    case "$enc" in
        dofa|dofav2)  # micrometres, keys already exist in base config
            case "$dsgroup" in
                mados)
                    args="encoder.wave_list.optical.B1=2.2 encoder.wave_list.optical.B2=1.61 encoder.wave_list.optical.B3=0.864 encoder.wave_list.optical.B4=0.832 encoder.wave_list.optical.B5=0.783 encoder.wave_list.optical.B7=0.705 encoder.wave_list.optical.B8=0.665 encoder.wave_list.optical.B8A=0.56 encoder.wave_list.optical.B11=0.49 encoder.wave_list.optical.B12=0.44"
                    ;;
                s2_12b)
                    args="encoder.wave_list.optical.B1=2.2 encoder.wave_list.optical.B2=1.61 encoder.wave_list.optical.B3=0.945 encoder.wave_list.optical.B4=0.864 encoder.wave_list.optical.B5=0.832 encoder.wave_list.optical.B6=0.783 encoder.wave_list.optical.B7=0.74 encoder.wave_list.optical.B8=0.705 encoder.wave_list.optical.B8A=0.665 encoder.wave_list.optical.B9=0.56 encoder.wave_list.optical.B11=0.49 encoder.wave_list.optical.B12=0.44"
                    ;;
                mfn)
                    args="encoder.wave_list.optical.B2=2.2 encoder.wave_list.optical.B3=1.61 encoder.wave_list.optical.B4=0.832 encoder.wave_list.optical.B8=0.665 encoder.wave_list.optical.B11=0.56 encoder.wave_list.optical.B12=0.49"
                    ;;
            esac
            ;;
        panopticon)  # nanometres, keys already exist in base config
            case "$dsgroup" in
                mados)
                    args="encoder.wave_list.optical.B1=2203.618 encoder.wave_list.optical.B2=1613.862 encoder.wave_list.optical.B3=864.78 encoder.wave_list.optical.B4=827.539 encoder.wave_list.optical.B5=782.419 encoder.wave_list.optical.B7=704.006 encoder.wave_list.optical.B8=664.63 encoder.wave_list.optical.B8A=559.599 encoder.wave_list.optical.B11=492.997 encoder.wave_list.optical.B12=442.923"
                    ;;
                s2_12b)
                    args="encoder.wave_list.optical.B1=2203.618 encoder.wave_list.optical.B2=1613.862 encoder.wave_list.optical.B3=945.029 encoder.wave_list.optical.B4=864.78 encoder.wave_list.optical.B5=827.539 encoder.wave_list.optical.B6=782.419 encoder.wave_list.optical.B7=740.552 encoder.wave_list.optical.B8=704.006 encoder.wave_list.optical.B8A=664.63 encoder.wave_list.optical.B9=559.599 encoder.wave_list.optical.B11=492.997 encoder.wave_list.optical.B12=442.923"
                    ;;
                mfn)
                    args="encoder.wave_list.optical.B2=2203.618 encoder.wave_list.optical.B3=1613.862 encoder.wave_list.optical.B4=827.539 encoder.wave_list.optical.B8=664.63 encoder.wave_list.optical.B11=559.599 encoder.wave_list.optical.B12=492.997"
                    ;;
            esac
            ;;
        copernicus_fm_base)  # nanometres, keys already exist in base config
            case "$dsgroup" in
                mados)
                    args="encoder.wave_list.optical.B1=2190 encoder.wave_list.optical.B2=1610 encoder.wave_list.optical.B3=865 encoder.wave_list.optical.B4=842 encoder.wave_list.optical.B5=783 encoder.wave_list.optical.B7=705 encoder.wave_list.optical.B8=665 encoder.wave_list.optical.B8A=560 encoder.wave_list.optical.B11=490 encoder.wave_list.optical.B12=443"
                    ;;
                s2_12b)
                    args="encoder.wave_list.optical.B1=2190 encoder.wave_list.optical.B2=1610 encoder.wave_list.optical.B3=945 encoder.wave_list.optical.B4=865 encoder.wave_list.optical.B5=842 encoder.wave_list.optical.B6=783 encoder.wave_list.optical.B7=740 encoder.wave_list.optical.B8=705 encoder.wave_list.optical.B8A=665 encoder.wave_list.optical.B9=560 encoder.wave_list.optical.B11=490 encoder.wave_list.optical.B12=443"
                    ;;
                mfn)
                    args="encoder.wave_list.optical.B2=2190 encoder.wave_list.optical.B3=1610 encoder.wave_list.optical.B4=842 encoder.wave_list.optical.B8=665 encoder.wave_list.optical.B11=560 encoder.wave_list.optical.B12=490"
                    ;;
            esac
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            case "$dsgroup" in
                mados)
                    args="+encoder.band_wavelength_overrides.optical.B1=2.19 +encoder.band_wavelength_overrides.optical.B2=1.61 +encoder.band_wavelength_overrides.optical.B3=0.865 +encoder.band_wavelength_overrides.optical.B4=0.842 +encoder.band_wavelength_overrides.optical.B5=0.783 +encoder.band_wavelength_overrides.optical.B7=0.705 +encoder.band_wavelength_overrides.optical.B8=0.665 +encoder.band_wavelength_overrides.optical.B8A=0.56 +encoder.band_wavelength_overrides.optical.B11=0.49 +encoder.band_wavelength_overrides.optical.B12=0.443"
                    ;;
                s2_12b)
                    args="+encoder.band_wavelength_overrides.optical.B1=2.19 +encoder.band_wavelength_overrides.optical.B2=1.61 +encoder.band_wavelength_overrides.optical.B3=0.945 +encoder.band_wavelength_overrides.optical.B4=0.865 +encoder.band_wavelength_overrides.optical.B5=0.842 +encoder.band_wavelength_overrides.optical.B6=0.783 +encoder.band_wavelength_overrides.optical.B7=0.74 +encoder.band_wavelength_overrides.optical.B8=0.705 +encoder.band_wavelength_overrides.optical.B8A=0.665 +encoder.band_wavelength_overrides.optical.B9=0.56 +encoder.band_wavelength_overrides.optical.B11=0.49 +encoder.band_wavelength_overrides.optical.B12=0.443"
                    ;;
                mfn)
                    args="+encoder.band_wavelength_overrides.optical.B2=2.19 +encoder.band_wavelength_overrides.optical.B3=1.61 +encoder.band_wavelength_overrides.optical.B4=0.842 +encoder.band_wavelength_overrides.optical.B8=0.665 +encoder.band_wavelength_overrides.optical.B11=0.56 +encoder.band_wavelength_overrides.optical.B12=0.49"
                    ;;
            esac
            ;;
        smarties_vitb)  # discrete projection index, override dict
            case "$dsgroup" in
                mados)
                    args="+encoder.proj_idx_override.optical.B1=14 +encoder.proj_idx_override.optical.B2=13 +encoder.proj_idx_override.optical.B3=11 +encoder.proj_idx_override.optical.B4=9 +encoder.proj_idx_override.optical.B5=10 +encoder.proj_idx_override.optical.B7=7 +encoder.proj_idx_override.optical.B8=6 +encoder.proj_idx_override.optical.B8A=4 +encoder.proj_idx_override.optical.B11=1 +encoder.proj_idx_override.optical.B12=0"
                    ;;
                s2_12b)
                    args="+encoder.proj_idx_override.optical.B1=14 +encoder.proj_idx_override.optical.B2=13 +encoder.proj_idx_override.optical.B3=12 +encoder.proj_idx_override.optical.B4=11 +encoder.proj_idx_override.optical.B5=9 +encoder.proj_idx_override.optical.B6=10 +encoder.proj_idx_override.optical.B7=8 +encoder.proj_idx_override.optical.B8=7 +encoder.proj_idx_override.optical.B8A=6 +encoder.proj_idx_override.optical.B9=4 +encoder.proj_idx_override.optical.B11=1 +encoder.proj_idx_override.optical.B12=0"
                    ;;
                mfn)
                    args="+encoder.proj_idx_override.optical.B2=14 +encoder.proj_idx_override.optical.B3=13 +encoder.proj_idx_override.optical.B4=9 +encoder.proj_idx_override.optical.B8=6 +encoder.proj_idx_override.optical.B11=4 +encoder.proj_idx_override.optical.B12=1"
                    ;;
            esac
            ;;
        *)
            echo "ERROR: unknown encoder '$enc'" >&2
            exit 1
            ;;
    esac

    echo "$args"
}

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r enc=$1 ds=$2 task=$3 decoder=$4 preprocessing=$5
    local -r criterion=$6 root=$7 dsgroup=$8 time_limit=$9

    local -r job_name="p3wl_${enc}_${ds}"
    local -r work_dir="${WORK_BASE}/${enc}"
    local -r log_prefix="${LOG_DIR}/${enc}_${ds}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %s]\n' "$enc" "$ds"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(swap_args "$enc" "$dsgroup")"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${ds}"
    cmd+=" encoder=${enc}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" finetune=false"
    [ -n "$root" ] && cmd+=" dataset.root_path='${root}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=${WANDB_PROJECT}"
    [ "$task" = "linear_classification" ] && cmd+=" batch_size=32"
    cmd+=" ${swap[*]}"

    printf '\nLaunching [%-20s | %s]:\n%s\n' "$enc" "$ds" "$cmd"

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

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder preprocessing criterion root_path dsgroup time_limit"
# preprocessing here is the *default* (external NormalizeMeanStd) variant;
# preprocessing_for_encoder() below swaps in the "_no_norm"/imagewise
# equivalent for carl_imagewise, since it does its own internal per-image
# normalization and must not also get an external NormalizeMeanStd.
# root_path "-" means the dataset config resolves its own path (GeoBench, via
# GEO_BENCH_DIR).
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mados"]="segmentation seg_upernet seg_focus_crop cross_entropy ${DATASET_PATH}/mados mados 8:00:00"
    ["mcashew-plantation"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["msa-crop-type"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - mfn 5:00:00"
)

# -----------------------------------------------------------------------------
# preprocessing_for_encoder <enc_key> <base_preprocessing>
# carl_imagewise (image_norm=true, its own per-image normalization) must use
# the "_no_norm" / imagewise preprocessing variant instead of the default
# external-NormalizeMeanStd one. All other encoders keep the default
# preprocessing unchanged.
# -----------------------------------------------------------------------------
preprocessing_for_encoder() {
    local -r enc=$1 base=$2
    if [ "$enc" = "carl_imagewise" ]; then
        case "$base" in
            seg_focus_crop)      echo "seg_focus_crop_no_norm" ;;
            seg_default)         echo "seg_default_no_norm" ;;
            cls_geobench_no_pad) echo "cls_geobench_imagewise_no_pad" ;;
            *)                   echo "$base" ;;
        esac
    else
        echo "$base"
    fi
}

readonly ENCODERS="dofa dofav2 panopticon copernicus_fm_base carl_imagewise smarties_vitb"

RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"

for enc in $ENCODERS; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $enc "* ]] && continue
    fi

    for ds in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $ds "* ]] && continue
        fi

        read -r task decoder preprocessing criterion root dsgroup time_limit <<< "${datasets[$ds]}"
        [ "$root" = "-" ] && root=""
        preprocessing=$(preprocessing_for_encoder "$enc" "$preprocessing")

        launch_job "$enc" "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$dsgroup" "$time_limit"
    done
done

echo ""
echo "All P3 wavelength-reversal ablation jobs queued (up to 24 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
