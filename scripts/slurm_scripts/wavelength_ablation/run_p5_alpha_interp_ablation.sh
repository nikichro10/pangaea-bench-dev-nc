#!/bin/bash
#
# P5 -- Neighbor-band alpha-interpolation ablation (MLP-Fusion / cls_linear).
#
# lambda_i(alpha) = (1-alpha)*lambda_i + alpha*lambda_{i+1}
#
# Every band in a dataset's native wavelength vector (sorted ascending) is
# pulled toward its immediate next-higher-wavelength neighbor by a continuous
# factor alpha -- a uniform sequential shift across the *whole* vector, not
# P1's specific 3 pairs and not constrained to a spectral group like P2. The
# last band in each vector (highest lambda) has no next neighbor and stays
# fixed. Pixel data and normalization are untouched, as in P1-P4 -- only the
# wavelength label fed to the hypernetwork moves.
#
# alpha=0 (baseline) and alpha=1 (~ P1, though not an identical construction --
# P1 only touches 3 chosen pairs, this touches every band) already exist as
# separate conditions; this script runs only the 3 intermediate values:
#   alpha = 0.25, 0.50, 0.75
# fully deterministic (closed-form formula), no seeds/draws needed.
#
# Because native band spacing is uneven, the absolute nm displacement at a
# fixed alpha varies a lot band to band (e.g. MADOS has no B9/B10, so
# B8A->B11 is a ~745nm jump at alpha=1, vs. B6->B7's ~43nm) -- expected and
# fine, same caveat as P3's reversal.
#
# Datasets: MADOS, m-Cashew-Plantation, m-SA-Crop-Type (segmentation, decoder
# = seg_mlp_fusion) plus m-Forestnet (linear classification, decoder always
# cls_linear -- a linear probe on pooled features has no spatial decoder to
# vary, so this run is identical regardless of whether the segmentation
# datasets alongside it use UPerNet or MLP-Fusion).
#
# Encoders: DOFA, DOFAv2, Panopticon, Copernicus-FM, CARL only -- SMARTIES is
# excluded. SMARTIES has no continuous wavelength input at all (bands map to
# discrete projector indices), so "25% of the way from projector 6 to
# projector 7" isn't a meaningful interpolation -- there's no such projector,
# and rounding would just silently collapse back to one of the two discrete
# endpoints. SMARTIES's architecture structurally cannot be probed this way.
#
# Per-encoder implementation (same mechanism as P1-P4):
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on
#     encoder.wave_list.optical.{band} (keys already exist, no '+').
#   CARL -- encoder.band_wavelength_overrides.optical.{band} (µm), '+' for new keys.
#
# CARL variant: carl_imagewise (image_norm=true, CARL's own internal per-image
# normalization), not the dataset-wise carl config P1-P4 used -- performs
# better, so this is now the standard CARL setup going forward. Paired with
# the "_no_norm"/imagewise preprocessing variants (seg_focus_crop_no_norm,
# seg_default_no_norm, cls_geobench_imagewise_no_pad) instead of the default
# external-NormalizeMeanStd ones, via preprocessing_for_encoder() below --
# same encoder+preprocessing pairing as run_carl_seg_benchmark.sh /
# run_geobench_cls_benchmark.sh's carl_imagewise entries, which already
# produced the "Baseline (S2)" carl_imagewise data on disk. The
# band_wavelength_overrides CLI-override mechanism and wave_list values are
# unaffected by image_norm -- only pixel normalization changes, not what
# wavelength label the hypernetwork receives.
#
# All encoders frozen (finetune=false). W&B project: wavelength-ablation
# (same project as P1-P4; runs distinguished by job/run name, prefix p5wl_).
#
# 5 encoders x 4 datasets x 3 alphas = 60 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS / RUN_ALPHAS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" RUN_ALPHAS="050" bash run_p5_alpha_interp_ablation.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p5_alpha_interp_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p5_alpha_interp_ablation"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <dsgroup: mados|s2_12b|mfn> <alpha: 025|050|075>
# Echoes the Hydra CLI overrides for one encoder's alpha-interpolated vector.
# Values are hardcoded (generated offline as (1-alpha)*lambda_i + alpha*lambda_{i+1}
# from each encoder's own native wave_list table -- see header for the formula).
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 dsgroup=$2 alpha=$3
    local args=""

    case "$enc" in
        dofa|dofav2)  # micrometres, keys already exist in base config
            case "$alpha" in
                025)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.4525 encoder.wave_list.optical.B2=0.5075 encoder.wave_list.optical.B3=0.58625 encoder.wave_list.optical.B4=0.675 encoder.wave_list.optical.B5=0.71375 encoder.wave_list.optical.B6=0.75075 encoder.wave_list.optical.B7=0.79525 encoder.wave_list.optical.B8=0.84 encoder.wave_list.optical.B8A=1.0505 encoder.wave_list.optical.B11=1.7575"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=0.4525 encoder.wave_list.optical.B2=0.5075 encoder.wave_list.optical.B3=0.58625 encoder.wave_list.optical.B4=0.675 encoder.wave_list.optical.B5=0.71375 encoder.wave_list.optical.B6=0.75075 encoder.wave_list.optical.B7=0.79525 encoder.wave_list.optical.B8=0.84 encoder.wave_list.optical.B8A=0.88425 encoder.wave_list.optical.B9=1.11125 encoder.wave_list.optical.B11=1.7575"
                    else
                        args="encoder.wave_list.optical.B2=0.5075 encoder.wave_list.optical.B3=0.58625 encoder.wave_list.optical.B4=0.70675 encoder.wave_list.optical.B8=1.0265 encoder.wave_list.optical.B11=1.7575"
                    fi
                    ;;
                050)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.465 encoder.wave_list.optical.B2=0.525 encoder.wave_list.optical.B3=0.6125 encoder.wave_list.optical.B4=0.685 encoder.wave_list.optical.B5=0.7225 encoder.wave_list.optical.B6=0.7615 encoder.wave_list.optical.B7=0.8075 encoder.wave_list.optical.B8=0.848 encoder.wave_list.optical.B8A=1.237 encoder.wave_list.optical.B11=1.905"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=0.465 encoder.wave_list.optical.B2=0.525 encoder.wave_list.optical.B3=0.6125 encoder.wave_list.optical.B4=0.685 encoder.wave_list.optical.B5=0.7225 encoder.wave_list.optical.B6=0.7615 encoder.wave_list.optical.B7=0.8075 encoder.wave_list.optical.B8=0.848 encoder.wave_list.optical.B8A=0.9045 encoder.wave_list.optical.B9=1.2775 encoder.wave_list.optical.B11=1.905"
                    else
                        args="encoder.wave_list.optical.B2=0.525 encoder.wave_list.optical.B3=0.6125 encoder.wave_list.optical.B4=0.7485 encoder.wave_list.optical.B8=1.221 encoder.wave_list.optical.B11=1.905"
                    fi
                    ;;
                075)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.4775 encoder.wave_list.optical.B2=0.5425 encoder.wave_list.optical.B3=0.63875 encoder.wave_list.optical.B4=0.695 encoder.wave_list.optical.B5=0.73125 encoder.wave_list.optical.B6=0.77225 encoder.wave_list.optical.B7=0.81975 encoder.wave_list.optical.B8=0.856 encoder.wave_list.optical.B8A=1.4235 encoder.wave_list.optical.B11=2.0525"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=0.4775 encoder.wave_list.optical.B2=0.5425 encoder.wave_list.optical.B3=0.63875 encoder.wave_list.optical.B4=0.695 encoder.wave_list.optical.B5=0.73125 encoder.wave_list.optical.B6=0.77225 encoder.wave_list.optical.B7=0.81975 encoder.wave_list.optical.B8=0.856 encoder.wave_list.optical.B8A=0.92475 encoder.wave_list.optical.B9=1.44375 encoder.wave_list.optical.B11=2.0525"
                    else
                        args="encoder.wave_list.optical.B2=0.5425 encoder.wave_list.optical.B3=0.63875 encoder.wave_list.optical.B4=0.79025 encoder.wave_list.optical.B8=1.4155 encoder.wave_list.optical.B11=2.0525"
                    fi
                    ;;
            esac
            ;;
        panopticon)  # nanometres, keys already exist in base config
            case "$alpha" in
                025)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=455.4415 encoder.wave_list.optical.B2=509.6475 encoder.wave_list.optical.B3=585.85675 encoder.wave_list.optical.B4=674.474 encoder.wave_list.optical.B5=713.1425 encoder.wave_list.optical.B6=751.01875 encoder.wave_list.optical.B7=793.699 encoder.wave_list.optical.B8=836.84925 encoder.wave_list.optical.B8A=1052.0505 encoder.wave_list.optical.B11=1761.301"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=455.4415 encoder.wave_list.optical.B2=509.6475 encoder.wave_list.optical.B3=585.85675 encoder.wave_list.optical.B4=674.474 encoder.wave_list.optical.B5=713.1425 encoder.wave_list.optical.B6=751.01875 encoder.wave_list.optical.B7=793.699 encoder.wave_list.optical.B8=836.84925 encoder.wave_list.optical.B8A=884.84225 encoder.wave_list.optical.B9=1112.23725 encoder.wave_list.optical.B11=1761.301"
                    else
                        args="encoder.wave_list.optical.B2=509.6475 encoder.wave_list.optical.B3=585.85675 encoder.wave_list.optical.B4=705.35725 encoder.wave_list.optical.B8=1024.11975 encoder.wave_list.optical.B11=1761.301"
                    fi
                    ;;
                050)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=467.96 encoder.wave_list.optical.B2=526.298 encoder.wave_list.optical.B3=612.1145 encoder.wave_list.optical.B4=684.318 encoder.wave_list.optical.B5=722.279 encoder.wave_list.optical.B6=761.4855 encoder.wave_list.optical.B7=804.979 encoder.wave_list.optical.B8=846.1595 encoder.wave_list.optical.B8A=1239.321 encoder.wave_list.optical.B11=1908.74"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=467.96 encoder.wave_list.optical.B2=526.298 encoder.wave_list.optical.B3=612.1145 encoder.wave_list.optical.B4=684.318 encoder.wave_list.optical.B5=722.279 encoder.wave_list.optical.B6=761.4855 encoder.wave_list.optical.B7=804.979 encoder.wave_list.optical.B8=846.1595 encoder.wave_list.optical.B8A=904.9045 encoder.wave_list.optical.B9=1279.4455 encoder.wave_list.optical.B11=1908.74"
                    else
                        args="encoder.wave_list.optical.B2=526.298 encoder.wave_list.optical.B3=612.1145 encoder.wave_list.optical.B4=746.0845 encoder.wave_list.optical.B8=1220.7005 encoder.wave_list.optical.B11=1908.74"
                    fi
                    ;;
                075)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=480.4785 encoder.wave_list.optical.B2=542.9485 encoder.wave_list.optical.B3=638.37225 encoder.wave_list.optical.B4=694.162 encoder.wave_list.optical.B5=731.4155 encoder.wave_list.optical.B6=771.95225 encoder.wave_list.optical.B7=816.259 encoder.wave_list.optical.B8=855.46975 encoder.wave_list.optical.B8A=1426.5915 encoder.wave_list.optical.B11=2056.179"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=480.4785 encoder.wave_list.optical.B2=542.9485 encoder.wave_list.optical.B3=638.37225 encoder.wave_list.optical.B4=694.162 encoder.wave_list.optical.B5=731.4155 encoder.wave_list.optical.B6=771.95225 encoder.wave_list.optical.B7=816.259 encoder.wave_list.optical.B8=855.46975 encoder.wave_list.optical.B8A=924.96675 encoder.wave_list.optical.B9=1446.65375 encoder.wave_list.optical.B11=2056.179"
                    else
                        args="encoder.wave_list.optical.B2=542.9485 encoder.wave_list.optical.B3=638.37225 encoder.wave_list.optical.B4=786.81175 encoder.wave_list.optical.B8=1417.28125 encoder.wave_list.optical.B11=2056.179"
                    fi
                    ;;
            esac
            ;;
        copernicus_fm_base)  # nanometres, keys already exist in base config
            case "$alpha" in
                025)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=454.75 encoder.wave_list.optical.B2=507.5 encoder.wave_list.optical.B3=586.25 encoder.wave_list.optical.B4=675 encoder.wave_list.optical.B5=713.75 encoder.wave_list.optical.B6=750.75 encoder.wave_list.optical.B7=797.75 encoder.wave_list.optical.B8=847.75 encoder.wave_list.optical.B8A=1051.25 encoder.wave_list.optical.B11=1755"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=454.75 encoder.wave_list.optical.B2=507.5 encoder.wave_list.optical.B3=586.25 encoder.wave_list.optical.B4=675 encoder.wave_list.optical.B5=713.75 encoder.wave_list.optical.B6=750.75 encoder.wave_list.optical.B7=797.75 encoder.wave_list.optical.B8=847.75 encoder.wave_list.optical.B8A=885 encoder.wave_list.optical.B9=1111.25 encoder.wave_list.optical.B11=1755"
                    else
                        args="encoder.wave_list.optical.B2=507.5 encoder.wave_list.optical.B3=586.25 encoder.wave_list.optical.B4=709.25 encoder.wave_list.optical.B8=1034 encoder.wave_list.optical.B11=1755"
                    fi
                    ;;
                050)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=466.5 encoder.wave_list.optical.B2=525 encoder.wave_list.optical.B3=612.5 encoder.wave_list.optical.B4=685 encoder.wave_list.optical.B5=722.5 encoder.wave_list.optical.B6=761.5 encoder.wave_list.optical.B7=812.5 encoder.wave_list.optical.B8=853.5 encoder.wave_list.optical.B8A=1237.5 encoder.wave_list.optical.B11=1900"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=466.5 encoder.wave_list.optical.B2=525 encoder.wave_list.optical.B3=612.5 encoder.wave_list.optical.B4=685 encoder.wave_list.optical.B5=722.5 encoder.wave_list.optical.B6=761.5 encoder.wave_list.optical.B7=812.5 encoder.wave_list.optical.B8=853.5 encoder.wave_list.optical.B8A=905 encoder.wave_list.optical.B9=1277.5 encoder.wave_list.optical.B11=1900"
                    else
                        args="encoder.wave_list.optical.B2=525 encoder.wave_list.optical.B3=612.5 encoder.wave_list.optical.B4=753.5 encoder.wave_list.optical.B8=1226 encoder.wave_list.optical.B11=1900"
                    fi
                    ;;
                075)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=478.25 encoder.wave_list.optical.B2=542.5 encoder.wave_list.optical.B3=638.75 encoder.wave_list.optical.B4=695 encoder.wave_list.optical.B5=731.25 encoder.wave_list.optical.B6=772.25 encoder.wave_list.optical.B7=827.25 encoder.wave_list.optical.B8=859.25 encoder.wave_list.optical.B8A=1423.75 encoder.wave_list.optical.B11=2045"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=478.25 encoder.wave_list.optical.B2=542.5 encoder.wave_list.optical.B3=638.75 encoder.wave_list.optical.B4=695 encoder.wave_list.optical.B5=731.25 encoder.wave_list.optical.B6=772.25 encoder.wave_list.optical.B7=827.25 encoder.wave_list.optical.B8=859.25 encoder.wave_list.optical.B8A=925 encoder.wave_list.optical.B9=1443.75 encoder.wave_list.optical.B11=2045"
                    else
                        args="encoder.wave_list.optical.B2=542.5 encoder.wave_list.optical.B3=638.75 encoder.wave_list.optical.B4=797.75 encoder.wave_list.optical.B8=1418 encoder.wave_list.optical.B11=2045"
                    fi
                    ;;
            esac
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            case "$alpha" in
                025)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.45475 +encoder.band_wavelength_overrides.optical.B2=0.5075 +encoder.band_wavelength_overrides.optical.B3=0.58625 +encoder.band_wavelength_overrides.optical.B4=0.675 +encoder.band_wavelength_overrides.optical.B5=0.71375 +encoder.band_wavelength_overrides.optical.B6=0.75075 +encoder.band_wavelength_overrides.optical.B7=0.79775 +encoder.band_wavelength_overrides.optical.B8=0.84775 +encoder.band_wavelength_overrides.optical.B8A=1.05125 +encoder.band_wavelength_overrides.optical.B11=1.755"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.45475 +encoder.band_wavelength_overrides.optical.B2=0.5075 +encoder.band_wavelength_overrides.optical.B3=0.58625 +encoder.band_wavelength_overrides.optical.B4=0.675 +encoder.band_wavelength_overrides.optical.B5=0.71375 +encoder.band_wavelength_overrides.optical.B6=0.75075 +encoder.band_wavelength_overrides.optical.B7=0.79775 +encoder.band_wavelength_overrides.optical.B8=0.84775 +encoder.band_wavelength_overrides.optical.B8A=0.885 +encoder.band_wavelength_overrides.optical.B9=1.11125 +encoder.band_wavelength_overrides.optical.B11=1.755"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=0.5075 +encoder.band_wavelength_overrides.optical.B3=0.58625 +encoder.band_wavelength_overrides.optical.B4=0.70925 +encoder.band_wavelength_overrides.optical.B8=1.034 +encoder.band_wavelength_overrides.optical.B11=1.755"
                    fi
                    ;;
                050)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.4665 +encoder.band_wavelength_overrides.optical.B2=0.525 +encoder.band_wavelength_overrides.optical.B3=0.6125 +encoder.band_wavelength_overrides.optical.B4=0.685 +encoder.band_wavelength_overrides.optical.B5=0.7225 +encoder.band_wavelength_overrides.optical.B6=0.7615 +encoder.band_wavelength_overrides.optical.B7=0.8125 +encoder.band_wavelength_overrides.optical.B8=0.8535 +encoder.band_wavelength_overrides.optical.B8A=1.2375 +encoder.band_wavelength_overrides.optical.B11=1.9"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.4665 +encoder.band_wavelength_overrides.optical.B2=0.525 +encoder.band_wavelength_overrides.optical.B3=0.6125 +encoder.band_wavelength_overrides.optical.B4=0.685 +encoder.band_wavelength_overrides.optical.B5=0.7225 +encoder.band_wavelength_overrides.optical.B6=0.7615 +encoder.band_wavelength_overrides.optical.B7=0.8125 +encoder.band_wavelength_overrides.optical.B8=0.8535 +encoder.band_wavelength_overrides.optical.B8A=0.905 +encoder.band_wavelength_overrides.optical.B9=1.2775 +encoder.band_wavelength_overrides.optical.B11=1.9"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=0.525 +encoder.band_wavelength_overrides.optical.B3=0.6125 +encoder.band_wavelength_overrides.optical.B4=0.7535 +encoder.band_wavelength_overrides.optical.B8=1.226 +encoder.band_wavelength_overrides.optical.B11=1.9"
                    fi
                    ;;
                075)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.47825 +encoder.band_wavelength_overrides.optical.B2=0.5425 +encoder.band_wavelength_overrides.optical.B3=0.63875 +encoder.band_wavelength_overrides.optical.B4=0.695 +encoder.band_wavelength_overrides.optical.B5=0.73125 +encoder.band_wavelength_overrides.optical.B6=0.77225 +encoder.band_wavelength_overrides.optical.B7=0.82725 +encoder.band_wavelength_overrides.optical.B8=0.85925 +encoder.band_wavelength_overrides.optical.B8A=1.42375 +encoder.band_wavelength_overrides.optical.B11=2.045"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.47825 +encoder.band_wavelength_overrides.optical.B2=0.5425 +encoder.band_wavelength_overrides.optical.B3=0.63875 +encoder.band_wavelength_overrides.optical.B4=0.695 +encoder.band_wavelength_overrides.optical.B5=0.73125 +encoder.band_wavelength_overrides.optical.B6=0.77225 +encoder.band_wavelength_overrides.optical.B7=0.82725 +encoder.band_wavelength_overrides.optical.B8=0.85925 +encoder.band_wavelength_overrides.optical.B8A=0.925 +encoder.band_wavelength_overrides.optical.B9=1.44375 +encoder.band_wavelength_overrides.optical.B11=2.045"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=0.5425 +encoder.band_wavelength_overrides.optical.B3=0.63875 +encoder.band_wavelength_overrides.optical.B4=0.79775 +encoder.band_wavelength_overrides.optical.B8=1.418 +encoder.band_wavelength_overrides.optical.B11=2.045"
                    fi
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
    local -r criterion=$6 root=$7 dsgroup=$8 alpha=$9 time_limit=${10}

    local -r job_name="p5wl_${enc}_${ds}_a${alpha}"
    local -r work_dir="${WORK_BASE}/${enc}_a${alpha}"
    local -r log_prefix="${LOG_DIR}/${enc}_${ds}_a${alpha}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %-20s | alpha 0.%s]\n' "$enc" "$ds" "$alpha"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(swap_args "$enc" "$dsgroup" "$alpha")"

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

    printf '\nLaunching [%-20s | %-20s | alpha 0.%s]:\n%s\n' "$enc" "$ds" "$alpha" "$cmd"

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
    ["mados"]="segmentation seg_mlp_fusion seg_focus_crop cross_entropy ${DATASET_PATH}/mados mados 8:00:00"
    ["mcashew-plantation"]="segmentation seg_mlp_fusion seg_default cross_entropy - s2_12b 6:00:00"
    ["msa-crop-type"]="segmentation seg_mlp_fusion seg_default cross_entropy - s2_12b 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - mfn 5:00:00"
)

# -----------------------------------------------------------------------------
# preprocessing_for_encoder <enc_key> <base_preprocessing>
# carl_imagewise (image_norm=true, its own per-image normalization) must use
# the "_no_norm" / imagewise preprocessing variant instead of the default
# external-NormalizeMeanStd one -- same pairing as run_carl_seg_benchmark.sh /
# run_geobench_cls_benchmark.sh's carl_imagewise entries. All other encoders
# keep the default preprocessing unchanged.
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

readonly ENCODERS="dofa dofav2 panopticon copernicus_fm_base carl_imagewise"
readonly ALPHAS="025 050 075"

RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_ALPHAS="${RUN_ALPHAS:-all}"

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

        for alpha in $ALPHAS; do
            if [ "$RUN_ALPHAS" != "all" ]; then
                [[ " $RUN_ALPHAS " != *" $alpha "* ]] && continue
            fi
            launch_job "$enc" "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$dsgroup" "$alpha" "$time_limit"
        done
    done
done

echo ""
echo "All P5 alpha-interpolation ablation jobs queued (up to 60 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
