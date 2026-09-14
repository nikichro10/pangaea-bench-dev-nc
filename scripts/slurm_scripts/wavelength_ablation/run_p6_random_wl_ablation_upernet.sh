#!/bin/bash
#
# P6 (UPerNet variant) -- Fully random wavelength-label ablation.
#
# Identical to run_p6_random_wl_ablation.sh (same encoders, same datasets,
# same 2 seeded random-wavelength draws from p6_random_wl_draws.json) but
# with the UPerNet decoder instead of MLP-Fusion, for a decoder comparison at
# a fixed wavelength perturbation.
#
# Each band gets an *independent* draw from Uniform(400, 2500) nm -- no
# ordering, no relationship to that band's true wavelength, no relationship
# to any real Sentinel-2/Landsat/PlanetScope/GaoFen-2 band center at all.
# Unlike P4 (which permutes the *true* wavelength set among bands, so every
# band still gets some real spectral value), P6 tests whether the encoder
# cares whether the wavelength it's told is physically real in the first
# place -- values can land in gaps no real band ever occupies (e.g. the
# ~1000-1600nm gap between S2 B9 and B11), and nothing stops two bands from
# landing close together by chance. Pixel data and normalization are
# untouched, as in P1-P5 -- only the wavelength label fed to the
# hypernetwork moves.
#
# Construction: 2 independent draws, each a fixed vector of n_bands
# Uniform(400,2500) samples, rejection-sampled so no two bands in the same
# draw land within 5nm of each other (purely to avoid a degenerate
# near-duplicate-band edge case in a single draw -- not a physical
# constraint). Same sharing rule as P2/P4: each draw applies identically
# (the literal same nm value) across all encoders, and across datasets
# wherever the band count *and* band names match:
#   - MADOS (11 bands)                              -> its own draw per seed
#   - m-Cashew-Plantation / m-SA-Crop-Type (12 bands, identical vectors)
#                                                     -> share one draw per seed
#   - m-Forestnet (6 bands)                          -> its own draw per seed
#   - SpaceNet7-CD (3 bands: B4,B3,B2 -- PlanetScope RGB, change detection)
#                                                     -> its own draw per seed
#   - FiveBillionPixels (4 bands: B8,B4,B3,B2 -- GaoFen-2 NIR+RGB, NOT pure
#     RGB despite the name -- segmentation)          -> its own draw per seed
#
# Seeds: SEED_1=501, SEED_2=502 (offsets +5000/+9000/+13000/+17000 on the
# 12-band/m-Forestnet/SpaceNet7-CD/FiveBillionPixels draws respectively, only
# to decorrelate the different vector-length draws within the same seed).
# Algorithm: rejection-sampled np.random.default_rng(seed).uniform(400, 2500,
# size=n) resampled until the sorted values are all >=5nm apart.
#
# Unlike P1-P4, these values cannot be derived from any known band-center
# formula -- the full sampled table (per encoder, per draw, per dataset) is
# documented in p6_random_wl_draws.json for reproducibility.
#
# Encoders: DOFA, DOFAv2, Panopticon, Copernicus-FM, CARL (imagewise) only --
# SMARTIES excluded, same reasoning as P5: it has no continuous wavelength
# input (bands map to discrete projector indices), so "a random float from
# Uniform(400,2500)" has no meaningful mapping into its architecture.
#
# CARL uses carl_imagewise (image_norm=true, its own internal per-image
# normalization), matching the standard set in P1-P5 -- paired with the
# "_no_norm"/imagewise preprocessing variants via preprocessing_for_encoder()
# below, not the default external-NormalizeMeanStd ones.
#
# Decoder: UPerNet for all segmentation datasets -- seg_upernet for
# MADOS/cashew/crop-type/FiveBillionPixels, seg_siamupernet_conc for
# SpaceNet7-CD (siamese, change detection). cls_linear for m-Forestnet
# (always decoder-agnostic, linear probe on pooled features).
#
# Per-encoder implementation (same mechanism as P1-P5):
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on
#     encoder.wave_list.optical.{band} (keys already exist, no '+').
#   CARL (imagewise) -- encoder.band_wavelength_overrides.optical.{band} (µm),
#     '+' for new keys.
#
# All encoders frozen (finetune=false). W&B project: wavelength-ablation
# (same project as P1-P5; runs distinguished by job/run name, prefix p6up_).
#
# 5 encoders x 6 datasets x 2 draws = 60 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS / RUN_DRAWS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" RUN_DRAWS="1" bash run_p6_random_wl_ablation_upernet.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p6_random_wl_ablation_upernet"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p6_random_wl_ablation_upernet"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <dsgroup: mados|s2_12b|mfn|sn7cd|fbp> <draw: 1|2>
# Echoes the Hydra CLI overrides for one encoder's fully-random wavelength
# vector. Values are hardcoded (generated offline via rejection-sampled
# Uniform(400,2500) draws, converted to each encoder's own unit convention --
# see header and p6_random_wl_draws.json for the literal sampled values).
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 dsgroup=$2 draw=$3
    local args=""

    case "$enc" in
        dofa|dofav2)  # micrometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2.492495 encoder.wave_list.optical.B2=1.74309 encoder.wave_list.optical.B3=1.279714 encoder.wave_list.optical.B4=1.574607 encoder.wave_list.optical.B5=2.20877 encoder.wave_list.optical.B6=2.48611 encoder.wave_list.optical.B7=0.553906 encoder.wave_list.optical.B8=2.140654 encoder.wave_list.optical.B8A=0.719441 encoder.wave_list.optical.B11=1.923719 encoder.wave_list.optical.B12=1.36536"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1.881357 encoder.wave_list.optical.B2=1.71089 encoder.wave_list.optical.B3=0.574267 encoder.wave_list.optical.B4=1.472651 encoder.wave_list.optical.B5=0.48473 encoder.wave_list.optical.B6=1.558498 encoder.wave_list.optical.B7=0.718972 encoder.wave_list.optical.B8=1.258879 encoder.wave_list.optical.B8A=0.782082 encoder.wave_list.optical.B9=2.334465 encoder.wave_list.optical.B11=1.868171 encoder.wave_list.optical.B12=1.682292"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1.542973 encoder.wave_list.optical.B3=0.481902 encoder.wave_list.optical.B4=1.720891 encoder.wave_list.optical.B8=1.244888 encoder.wave_list.optical.B11=1.91343 encoder.wave_list.optical.B12=1.385045"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=1.334215 encoder.wave_list.optical.B3=2.135323 encoder.wave_list.optical.B2=0.462891"
                    else
                        args="encoder.wave_list.optical.B8=1.521339 encoder.wave_list.optical.B4=1.037867 encoder.wave_list.optical.B3=1.793272 encoder.wave_list.optical.B2=1.588468"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.693149 encoder.wave_list.optical.B2=1.193755 encoder.wave_list.optical.B3=2.111195 encoder.wave_list.optical.B4=1.378592 encoder.wave_list.optical.B5=1.445826 encoder.wave_list.optical.B6=0.540555 encoder.wave_list.optical.B7=1.931067 encoder.wave_list.optical.B8=2.36185 encoder.wave_list.optical.B8A=1.677627 encoder.wave_list.optical.B11=1.291454 encoder.wave_list.optical.B12=0.955193"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1.396741 encoder.wave_list.optical.B2=1.639602 encoder.wave_list.optical.B3=2.029846 encoder.wave_list.optical.B4=1.375738 encoder.wave_list.optical.B5=0.917061 encoder.wave_list.optical.B6=0.837527 encoder.wave_list.optical.B7=1.422631 encoder.wave_list.optical.B8=2.264879 encoder.wave_list.optical.B8A=2.383487 encoder.wave_list.optical.B9=0.909054 encoder.wave_list.optical.B11=1.769199 encoder.wave_list.optical.B12=2.227005"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1.347486 encoder.wave_list.optical.B3=2.027374 encoder.wave_list.optical.B4=2.455442 encoder.wave_list.optical.B8=2.420121 encoder.wave_list.optical.B11=1.622063 encoder.wave_list.optical.B12=1.791035"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=0.621842 encoder.wave_list.optical.B3=0.674262 encoder.wave_list.optical.B2=1.424243"
                    else
                        args="encoder.wave_list.optical.B8=2.079129 encoder.wave_list.optical.B4=1.055904 encoder.wave_list.optical.B3=2.205496 encoder.wave_list.optical.B2=1.108139"
                    fi
                    ;;
            esac
            ;;
        panopticon)  # nanometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2492.495 encoder.wave_list.optical.B2=1743.09 encoder.wave_list.optical.B3=1279.714 encoder.wave_list.optical.B4=1574.607 encoder.wave_list.optical.B5=2208.77 encoder.wave_list.optical.B6=2486.11 encoder.wave_list.optical.B7=553.906 encoder.wave_list.optical.B8=2140.654 encoder.wave_list.optical.B8A=719.441 encoder.wave_list.optical.B11=1923.719 encoder.wave_list.optical.B12=1365.36"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1881.357 encoder.wave_list.optical.B2=1710.89 encoder.wave_list.optical.B3=574.267 encoder.wave_list.optical.B4=1472.651 encoder.wave_list.optical.B5=484.73 encoder.wave_list.optical.B6=1558.498 encoder.wave_list.optical.B7=718.972 encoder.wave_list.optical.B8=1258.879 encoder.wave_list.optical.B8A=782.082 encoder.wave_list.optical.B9=2334.465 encoder.wave_list.optical.B11=1868.171 encoder.wave_list.optical.B12=1682.292"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1542.973 encoder.wave_list.optical.B3=481.902 encoder.wave_list.optical.B4=1720.891 encoder.wave_list.optical.B8=1244.888 encoder.wave_list.optical.B11=1913.43 encoder.wave_list.optical.B12=1385.045"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=1334.215 encoder.wave_list.optical.B3=2135.323 encoder.wave_list.optical.B2=462.891"
                    else
                        args="encoder.wave_list.optical.B8=1521.339 encoder.wave_list.optical.B4=1037.867 encoder.wave_list.optical.B3=1793.272 encoder.wave_list.optical.B2=1588.468"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=693.149 encoder.wave_list.optical.B2=1193.755 encoder.wave_list.optical.B3=2111.195 encoder.wave_list.optical.B4=1378.592 encoder.wave_list.optical.B5=1445.826 encoder.wave_list.optical.B6=540.555 encoder.wave_list.optical.B7=1931.067 encoder.wave_list.optical.B8=2361.85 encoder.wave_list.optical.B8A=1677.627 encoder.wave_list.optical.B11=1291.454 encoder.wave_list.optical.B12=955.193"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1396.741 encoder.wave_list.optical.B2=1639.602 encoder.wave_list.optical.B3=2029.846 encoder.wave_list.optical.B4=1375.738 encoder.wave_list.optical.B5=917.061 encoder.wave_list.optical.B6=837.527 encoder.wave_list.optical.B7=1422.631 encoder.wave_list.optical.B8=2264.879 encoder.wave_list.optical.B8A=2383.487 encoder.wave_list.optical.B9=909.054 encoder.wave_list.optical.B11=1769.199 encoder.wave_list.optical.B12=2227.005"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1347.486 encoder.wave_list.optical.B3=2027.374 encoder.wave_list.optical.B4=2455.442 encoder.wave_list.optical.B8=2420.121 encoder.wave_list.optical.B11=1622.063 encoder.wave_list.optical.B12=1791.035"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=621.842 encoder.wave_list.optical.B3=674.262 encoder.wave_list.optical.B2=1424.243"
                    else
                        args="encoder.wave_list.optical.B8=2079.129 encoder.wave_list.optical.B4=1055.904 encoder.wave_list.optical.B3=2205.496 encoder.wave_list.optical.B2=1108.139"
                    fi
                    ;;
            esac
            ;;
        copernicus_fm_base)  # nanometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2492.495 encoder.wave_list.optical.B2=1743.09 encoder.wave_list.optical.B3=1279.714 encoder.wave_list.optical.B4=1574.607 encoder.wave_list.optical.B5=2208.77 encoder.wave_list.optical.B6=2486.11 encoder.wave_list.optical.B7=553.906 encoder.wave_list.optical.B8=2140.654 encoder.wave_list.optical.B8A=719.441 encoder.wave_list.optical.B11=1923.719 encoder.wave_list.optical.B12=1365.36"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1881.357 encoder.wave_list.optical.B2=1710.89 encoder.wave_list.optical.B3=574.267 encoder.wave_list.optical.B4=1472.651 encoder.wave_list.optical.B5=484.73 encoder.wave_list.optical.B6=1558.498 encoder.wave_list.optical.B7=718.972 encoder.wave_list.optical.B8=1258.879 encoder.wave_list.optical.B8A=782.082 encoder.wave_list.optical.B9=2334.465 encoder.wave_list.optical.B11=1868.171 encoder.wave_list.optical.B12=1682.292"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1542.973 encoder.wave_list.optical.B3=481.902 encoder.wave_list.optical.B4=1720.891 encoder.wave_list.optical.B8=1244.888 encoder.wave_list.optical.B11=1913.43 encoder.wave_list.optical.B12=1385.045"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=1334.215 encoder.wave_list.optical.B3=2135.323 encoder.wave_list.optical.B2=462.891"
                    else
                        args="encoder.wave_list.optical.B8=1521.339 encoder.wave_list.optical.B4=1037.867 encoder.wave_list.optical.B3=1793.272 encoder.wave_list.optical.B2=1588.468"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=693.149 encoder.wave_list.optical.B2=1193.755 encoder.wave_list.optical.B3=2111.195 encoder.wave_list.optical.B4=1378.592 encoder.wave_list.optical.B5=1445.826 encoder.wave_list.optical.B6=540.555 encoder.wave_list.optical.B7=1931.067 encoder.wave_list.optical.B8=2361.85 encoder.wave_list.optical.B8A=1677.627 encoder.wave_list.optical.B11=1291.454 encoder.wave_list.optical.B12=955.193"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1396.741 encoder.wave_list.optical.B2=1639.602 encoder.wave_list.optical.B3=2029.846 encoder.wave_list.optical.B4=1375.738 encoder.wave_list.optical.B5=917.061 encoder.wave_list.optical.B6=837.527 encoder.wave_list.optical.B7=1422.631 encoder.wave_list.optical.B8=2264.879 encoder.wave_list.optical.B8A=2383.487 encoder.wave_list.optical.B9=909.054 encoder.wave_list.optical.B11=1769.199 encoder.wave_list.optical.B12=2227.005"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="encoder.wave_list.optical.B2=1347.486 encoder.wave_list.optical.B3=2027.374 encoder.wave_list.optical.B4=2455.442 encoder.wave_list.optical.B8=2420.121 encoder.wave_list.optical.B11=1622.063 encoder.wave_list.optical.B12=1791.035"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="encoder.wave_list.optical.B4=621.842 encoder.wave_list.optical.B3=674.262 encoder.wave_list.optical.B2=1424.243"
                    else
                        args="encoder.wave_list.optical.B8=2079.129 encoder.wave_list.optical.B4=1055.904 encoder.wave_list.optical.B3=2205.496 encoder.wave_list.optical.B2=1108.139"
                    fi
                    ;;
            esac
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=2.492495 +encoder.band_wavelength_overrides.optical.B2=1.74309 +encoder.band_wavelength_overrides.optical.B3=1.279714 +encoder.band_wavelength_overrides.optical.B4=1.574607 +encoder.band_wavelength_overrides.optical.B5=2.20877 +encoder.band_wavelength_overrides.optical.B6=2.48611 +encoder.band_wavelength_overrides.optical.B7=0.553906 +encoder.band_wavelength_overrides.optical.B8=2.140654 +encoder.band_wavelength_overrides.optical.B8A=0.719441 +encoder.band_wavelength_overrides.optical.B11=1.923719 +encoder.band_wavelength_overrides.optical.B12=1.36536"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=1.881357 +encoder.band_wavelength_overrides.optical.B2=1.71089 +encoder.band_wavelength_overrides.optical.B3=0.574267 +encoder.band_wavelength_overrides.optical.B4=1.472651 +encoder.band_wavelength_overrides.optical.B5=0.48473 +encoder.band_wavelength_overrides.optical.B6=1.558498 +encoder.band_wavelength_overrides.optical.B7=0.718972 +encoder.band_wavelength_overrides.optical.B8=1.258879 +encoder.band_wavelength_overrides.optical.B8A=0.782082 +encoder.band_wavelength_overrides.optical.B9=2.334465 +encoder.band_wavelength_overrides.optical.B11=1.868171 +encoder.band_wavelength_overrides.optical.B12=1.682292"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B2=1.542973 +encoder.band_wavelength_overrides.optical.B3=0.481902 +encoder.band_wavelength_overrides.optical.B4=1.720891 +encoder.band_wavelength_overrides.optical.B8=1.244888 +encoder.band_wavelength_overrides.optical.B11=1.91343 +encoder.band_wavelength_overrides.optical.B12=1.385045"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B4=1.334215 +encoder.band_wavelength_overrides.optical.B3=2.135323 +encoder.band_wavelength_overrides.optical.B2=0.462891"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B8=1.521339 +encoder.band_wavelength_overrides.optical.B4=1.037867 +encoder.band_wavelength_overrides.optical.B3=1.793272 +encoder.band_wavelength_overrides.optical.B2=1.588468"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.693149 +encoder.band_wavelength_overrides.optical.B2=1.193755 +encoder.band_wavelength_overrides.optical.B3=2.111195 +encoder.band_wavelength_overrides.optical.B4=1.378592 +encoder.band_wavelength_overrides.optical.B5=1.445826 +encoder.band_wavelength_overrides.optical.B6=0.540555 +encoder.band_wavelength_overrides.optical.B7=1.931067 +encoder.band_wavelength_overrides.optical.B8=2.36185 +encoder.band_wavelength_overrides.optical.B8A=1.677627 +encoder.band_wavelength_overrides.optical.B11=1.291454 +encoder.band_wavelength_overrides.optical.B12=0.955193"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=1.396741 +encoder.band_wavelength_overrides.optical.B2=1.639602 +encoder.band_wavelength_overrides.optical.B3=2.029846 +encoder.band_wavelength_overrides.optical.B4=1.375738 +encoder.band_wavelength_overrides.optical.B5=0.917061 +encoder.band_wavelength_overrides.optical.B6=0.837527 +encoder.band_wavelength_overrides.optical.B7=1.422631 +encoder.band_wavelength_overrides.optical.B8=2.264879 +encoder.band_wavelength_overrides.optical.B8A=2.383487 +encoder.band_wavelength_overrides.optical.B9=0.909054 +encoder.band_wavelength_overrides.optical.B11=1.769199 +encoder.band_wavelength_overrides.optical.B12=2.227005"
                    elif [ "$dsgroup" = "mfn" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B2=1.347486 +encoder.band_wavelength_overrides.optical.B3=2.027374 +encoder.band_wavelength_overrides.optical.B4=2.455442 +encoder.band_wavelength_overrides.optical.B8=2.420121 +encoder.band_wavelength_overrides.optical.B11=1.622063 +encoder.band_wavelength_overrides.optical.B12=1.791035"
                    elif [ "$dsgroup" = "sn7cd" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B4=0.621842 +encoder.band_wavelength_overrides.optical.B3=0.674262 +encoder.band_wavelength_overrides.optical.B2=1.424243"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B8=2.079129 +encoder.band_wavelength_overrides.optical.B4=1.055904 +encoder.band_wavelength_overrides.optical.B3=2.205496 +encoder.band_wavelength_overrides.optical.B2=1.108139"
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
    local -r criterion=$6 root=$7 dsgroup=$8 draw=$9 time_limit=${10}

    local -r job_name="p6up_${enc}_${ds}_${draw}"
    local -r work_dir="${WORK_BASE}/${enc}_draw${draw}"
    local -r log_prefix="${LOG_DIR}/${enc}_${ds}_${draw}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %-20s | draw %s]\n' "$enc" "$ds" "$draw"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(swap_args "$enc" "$dsgroup" "$draw")"

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

    printf '\nLaunching [%-20s | %-20s | draw %s]:\n%s\n' "$enc" "$ds" "$draw" "$cmd"

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
# equivalent for carl_imagewise.
# root_path "-" means the dataset config resolves its own path (GeoBench, via
# GEO_BENCH_DIR).
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mados"]="segmentation seg_upernet seg_focus_crop cross_entropy ${DATASET_PATH}/mados mados 8:00:00"
    ["mcashew-plantation"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["msa-crop-type"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - mfn 5:00:00"
    ["spacenet7cd"]="change_detection seg_siamupernet_conc seg_default binary_cross_entropy ${DATASET_PATH}/spacenet7 sn7cd 8:00:00"
    ["fivebillionpixels"]="segmentation seg_upernet seg_default cross_entropy ${DATASET_PATH}/fivebillionpixels fbp 16:00:00"
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

readonly ENCODERS="dofa dofav2 panopticon copernicus_fm_base carl_imagewise"
readonly DRAWS="1 2"

RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_DRAWS="${RUN_DRAWS:-all}"

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

        for draw in $DRAWS; do
            if [ "$RUN_DRAWS" != "all" ]; then
                [[ " $RUN_DRAWS " != *" $draw "* ]] && continue
            fi
            launch_job "$enc" "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$dsgroup" "$draw" "$time_limit"
        done
    done
done

echo ""
echo "All P6 (UPerNet) random-wavelength ablation jobs queued (up to 60 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
