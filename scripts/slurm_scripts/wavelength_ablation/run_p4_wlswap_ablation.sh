#!/bin/bash
#
# P4 -- Full-vector random wavelength-label permutation ablation.
#
# Unlike P2 (random permutation *within* a spectral group), P4 fully shuffles
# the entire wavelength vector -- any band can end up labeled with any other
# band's wavelength, no group boundary respected. This is the least
# structured / most chaotic condition in the ladder (P1 < P2 < P3/P4 severity,
# with P3 being a specific deterministic reversal and P4 being unconstrained
# random chaos). As with P1/P2/P3, pixel data and normalization stay
# completely untouched at their original channel position; only the
# wavelength/proj_idx label fed to each encoder's weight generator moves.
#
# Scope: MADOS, m-Cashew-Plantation, m-SA-Crop-Type (S2) plus m-Forestnet
# (Landsat-8-aliased, 6 bands -- its own full 6-band shuffle, not just
# RGB+SWIR like P2, since P4 has no group constraint to respect).
#
# Construction: 3 random full-vector permutations, each a derangement (no band
# keeps its own label) from a fixed, documented seed -- same sharing rule as
# P2: each draw defines one permutation pattern applied identically across all
# encoders, and shared across datasets wherever the band vector actually
# matches:
#   - MADOS (11 bands: B1-B8,B8A,B11,B12, no B9)      -> its own permutation per draw
#   - m-Cashew-Plantation / m-SA-Crop-Type (12 bands: B1-B9,B11,B12, identical
#     vectors) -> share one permutation per draw
#   - m-Forestnet (6 bands: B2,B3,B4,B8,B11,B12)       -> its own permutation per draw
#
# Seeds: SEED_1=101, SEED_2=202, SEED_3=303 (offset +5000 on the 12-band draw,
# +9000 on the m-Forestnet draw, only to decorrelate the three vector-length
# draws within the same seed -- not for any statistical reason). Algorithm:
# rejection-sampled np.random.default_rng(seed).permutation(n) resampled
# until no fixed point.
#
#   Draw 1 (seed 101): MADOS      B1<-B12 B2<-B8A B3<-B4 B4<-B2 B5<-B8 B6<-B1
#                                 B7<-B6 B8<-B5 B8A<-B11 B11<-B7 B12<-B3
#                      12-band    B1<-B2 B2<-B5 B3<-B8 B4<-B1 B5<-B7 B6<-B11
#                                 B7<-B9 B8<-B12 B8A<-B3 B9<-B8A B11<-B4 B12<-B6
#                      mForestnet B2<-B12 B3<-B8 B4<-B2 B8<-B4 B11<-B3 B12<-B11
#   Draw 2 (seed 202): MADOS      B1<-B5 B2<-B6 B3<-B11 B4<-B8A B5<-B12 B6<-B2
#                                 B7<-B4 B8<-B1 B8A<-B7 B11<-B3 B12<-B8
#                      12-band    B1<-B11 B2<-B7 B3<-B6 B4<-B5 B5<-B8A B6<-B9
#                                 B7<-B12 B8<-B2 B8A<-B4 B9<-B8 B11<-B1 B12<-B3
#                      mForestnet B2<-B8 B3<-B12 B4<-B2 B8<-B11 B11<-B3 B12<-B4
#   Draw 3 (seed 303): MADOS      B1<-B3 B2<-B6 B3<-B12 B4<-B1 B5<-B4 B6<-B11
#                                 B7<-B2 B8<-B7 B8A<-B8 B11<-B5 B12<-B8A
#                      12-band    B1<-B8 B2<-B11 B3<-B7 B4<-B1 B5<-B12 B6<-B5
#                                 B7<-B3 B8<-B2 B8A<-B6 B9<-B4 B11<-B8A B12<-B9
#                      mForestnet B2<-B8 B3<-B4 B4<-B11 B8<-B12 B11<-B2 B12<-B3
#
# Full manifest (per-encoder resolved values): p4_wlswap_draws.json
#
# Per-encoder implementation (same mechanism as P1/P2/P3):
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on
#     encoder.wave_list.optical.{band} (keys already exist, no '+').
#   CARL     -- encoder.band_wavelength_overrides.optical.{band} (µm), '+' for new keys.
#   SMARTIES -- encoder.proj_idx_override.optical.{band} (discrete projector index), '+' for new keys.
#
# All encoders frozen (finetune=false). W&B project: wavelength-ablation
# (same project as P1/P2/P3; runs distinguished by job/run name, prefix p4wl_).
#
# 6 encoders x 4 datasets x 3 draws = 72 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS / RUN_DRAWS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" RUN_DRAWS="1" bash run_p4_wlswap_ablation.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p4_wlswap_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p4_wlswap_ablation"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <dsgroup: mados|s2_12b> <draw: 1|2|3>
# Echoes the Hydra CLI overrides for one encoder's full-vector permutation.
# Values are hardcoded (generated offline from each encoder's own native
# wave_list/proj_idx table -- see header for the permutation mapping).
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 dsgroup=$2 draw=$3
    local args=""

    case "$enc" in
        dofa|dofav2)  # micrometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2.2 encoder.wave_list.optical.B2=0.864 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=0.49 encoder.wave_list.optical.B5=0.832 encoder.wave_list.optical.B6=0.44 encoder.wave_list.optical.B7=0.74 encoder.wave_list.optical.B8=0.705 encoder.wave_list.optical.B8A=1.61 encoder.wave_list.optical.B11=0.783 encoder.wave_list.optical.B12=0.56"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=0.49 encoder.wave_list.optical.B2=0.705 encoder.wave_list.optical.B3=0.832 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.783 encoder.wave_list.optical.B6=1.61 encoder.wave_list.optical.B7=0.945 encoder.wave_list.optical.B8=2.2 encoder.wave_list.optical.B8A=0.56 encoder.wave_list.optical.B9=0.864 encoder.wave_list.optical.B11=0.665 encoder.wave_list.optical.B12=0.74"
                    else
                        args="encoder.wave_list.optical.B2=2.2 encoder.wave_list.optical.B3=0.832 encoder.wave_list.optical.B4=0.49 encoder.wave_list.optical.B8=0.665 encoder.wave_list.optical.B11=0.56 encoder.wave_list.optical.B12=1.61"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.705 encoder.wave_list.optical.B2=0.74 encoder.wave_list.optical.B3=1.61 encoder.wave_list.optical.B4=0.864 encoder.wave_list.optical.B5=2.2 encoder.wave_list.optical.B6=0.49 encoder.wave_list.optical.B7=0.665 encoder.wave_list.optical.B8=0.44 encoder.wave_list.optical.B8A=0.783 encoder.wave_list.optical.B11=0.56 encoder.wave_list.optical.B12=0.832"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1.61 encoder.wave_list.optical.B2=0.783 encoder.wave_list.optical.B3=0.74 encoder.wave_list.optical.B4=0.705 encoder.wave_list.optical.B5=0.864 encoder.wave_list.optical.B6=0.945 encoder.wave_list.optical.B7=2.2 encoder.wave_list.optical.B8=0.49 encoder.wave_list.optical.B8A=0.665 encoder.wave_list.optical.B9=0.832 encoder.wave_list.optical.B11=0.44 encoder.wave_list.optical.B12=0.56"
                    else
                        args="encoder.wave_list.optical.B2=0.832 encoder.wave_list.optical.B3=2.2 encoder.wave_list.optical.B4=0.49 encoder.wave_list.optical.B8=1.61 encoder.wave_list.optical.B11=0.56 encoder.wave_list.optical.B12=0.665"
                    fi
                    ;;
                3)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=0.56 encoder.wave_list.optical.B2=0.74 encoder.wave_list.optical.B3=2.2 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.665 encoder.wave_list.optical.B6=1.61 encoder.wave_list.optical.B7=0.49 encoder.wave_list.optical.B8=0.783 encoder.wave_list.optical.B8A=0.832 encoder.wave_list.optical.B11=0.705 encoder.wave_list.optical.B12=0.864"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=0.832 encoder.wave_list.optical.B2=1.61 encoder.wave_list.optical.B3=0.783 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=2.2 encoder.wave_list.optical.B6=0.705 encoder.wave_list.optical.B7=0.56 encoder.wave_list.optical.B8=0.49 encoder.wave_list.optical.B8A=0.74 encoder.wave_list.optical.B9=0.665 encoder.wave_list.optical.B11=0.864 encoder.wave_list.optical.B12=0.945"
                    else
                        args="encoder.wave_list.optical.B2=0.832 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=1.61 encoder.wave_list.optical.B8=2.2 encoder.wave_list.optical.B11=0.49 encoder.wave_list.optical.B12=0.56"
                    fi
                    ;;
            esac
            ;;
        panopticon)  # nanometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2203.618 encoder.wave_list.optical.B2=864.78 encoder.wave_list.optical.B3=664.63 encoder.wave_list.optical.B4=492.997 encoder.wave_list.optical.B5=827.539 encoder.wave_list.optical.B6=442.923 encoder.wave_list.optical.B7=740.552 encoder.wave_list.optical.B8=704.006 encoder.wave_list.optical.B8A=1613.862 encoder.wave_list.optical.B11=782.419 encoder.wave_list.optical.B12=559.599"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=492.997 encoder.wave_list.optical.B2=704.006 encoder.wave_list.optical.B3=827.539 encoder.wave_list.optical.B4=442.923 encoder.wave_list.optical.B5=782.419 encoder.wave_list.optical.B6=1613.862 encoder.wave_list.optical.B7=945.029 encoder.wave_list.optical.B8=2203.618 encoder.wave_list.optical.B8A=559.599 encoder.wave_list.optical.B9=864.78 encoder.wave_list.optical.B11=664.63 encoder.wave_list.optical.B12=740.552"
                    else
                        args="encoder.wave_list.optical.B2=2203.618 encoder.wave_list.optical.B3=827.539 encoder.wave_list.optical.B4=492.997 encoder.wave_list.optical.B8=664.63 encoder.wave_list.optical.B11=559.599 encoder.wave_list.optical.B12=1613.862"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=704.006 encoder.wave_list.optical.B2=740.552 encoder.wave_list.optical.B3=1613.862 encoder.wave_list.optical.B4=864.78 encoder.wave_list.optical.B5=2203.618 encoder.wave_list.optical.B6=492.997 encoder.wave_list.optical.B7=664.63 encoder.wave_list.optical.B8=442.923 encoder.wave_list.optical.B8A=782.419 encoder.wave_list.optical.B11=559.599 encoder.wave_list.optical.B12=827.539"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1613.862 encoder.wave_list.optical.B2=782.419 encoder.wave_list.optical.B3=740.552 encoder.wave_list.optical.B4=704.006 encoder.wave_list.optical.B5=864.78 encoder.wave_list.optical.B6=945.029 encoder.wave_list.optical.B7=2203.618 encoder.wave_list.optical.B8=492.997 encoder.wave_list.optical.B8A=664.63 encoder.wave_list.optical.B9=827.539 encoder.wave_list.optical.B11=442.923 encoder.wave_list.optical.B12=559.599"
                    else
                        args="encoder.wave_list.optical.B2=827.539 encoder.wave_list.optical.B3=2203.618 encoder.wave_list.optical.B4=492.997 encoder.wave_list.optical.B8=1613.862 encoder.wave_list.optical.B11=559.599 encoder.wave_list.optical.B12=664.63"
                    fi
                    ;;
                3)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=559.599 encoder.wave_list.optical.B2=740.552 encoder.wave_list.optical.B3=2203.618 encoder.wave_list.optical.B4=442.923 encoder.wave_list.optical.B5=664.63 encoder.wave_list.optical.B6=1613.862 encoder.wave_list.optical.B7=492.997 encoder.wave_list.optical.B8=782.419 encoder.wave_list.optical.B8A=827.539 encoder.wave_list.optical.B11=704.006 encoder.wave_list.optical.B12=864.78"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=827.539 encoder.wave_list.optical.B2=1613.862 encoder.wave_list.optical.B3=782.419 encoder.wave_list.optical.B4=442.923 encoder.wave_list.optical.B5=2203.618 encoder.wave_list.optical.B6=704.006 encoder.wave_list.optical.B7=559.599 encoder.wave_list.optical.B8=492.997 encoder.wave_list.optical.B8A=740.552 encoder.wave_list.optical.B9=664.63 encoder.wave_list.optical.B11=864.78 encoder.wave_list.optical.B12=945.029"
                    else
                        args="encoder.wave_list.optical.B2=827.539 encoder.wave_list.optical.B3=664.63 encoder.wave_list.optical.B4=1613.862 encoder.wave_list.optical.B8=2203.618 encoder.wave_list.optical.B11=492.997 encoder.wave_list.optical.B12=559.599"
                    fi
                    ;;
            esac
            ;;
        copernicus_fm_base)  # nanometres, keys already exist in base config
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=2190 encoder.wave_list.optical.B2=865 encoder.wave_list.optical.B3=665 encoder.wave_list.optical.B4=490 encoder.wave_list.optical.B5=842 encoder.wave_list.optical.B6=443 encoder.wave_list.optical.B7=740 encoder.wave_list.optical.B8=705 encoder.wave_list.optical.B8A=1610 encoder.wave_list.optical.B11=783 encoder.wave_list.optical.B12=560"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=490 encoder.wave_list.optical.B2=705 encoder.wave_list.optical.B3=842 encoder.wave_list.optical.B4=443 encoder.wave_list.optical.B5=783 encoder.wave_list.optical.B6=1610 encoder.wave_list.optical.B7=945 encoder.wave_list.optical.B8=2190 encoder.wave_list.optical.B8A=560 encoder.wave_list.optical.B9=865 encoder.wave_list.optical.B11=665 encoder.wave_list.optical.B12=740"
                    else
                        args="encoder.wave_list.optical.B2=2190 encoder.wave_list.optical.B3=842 encoder.wave_list.optical.B4=490 encoder.wave_list.optical.B8=665 encoder.wave_list.optical.B11=560 encoder.wave_list.optical.B12=1610"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=705 encoder.wave_list.optical.B2=740 encoder.wave_list.optical.B3=1610 encoder.wave_list.optical.B4=865 encoder.wave_list.optical.B5=2190 encoder.wave_list.optical.B6=490 encoder.wave_list.optical.B7=665 encoder.wave_list.optical.B8=443 encoder.wave_list.optical.B8A=783 encoder.wave_list.optical.B11=560 encoder.wave_list.optical.B12=842"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=1610 encoder.wave_list.optical.B2=783 encoder.wave_list.optical.B3=740 encoder.wave_list.optical.B4=705 encoder.wave_list.optical.B5=865 encoder.wave_list.optical.B6=945 encoder.wave_list.optical.B7=2190 encoder.wave_list.optical.B8=490 encoder.wave_list.optical.B8A=665 encoder.wave_list.optical.B9=842 encoder.wave_list.optical.B11=443 encoder.wave_list.optical.B12=560"
                    else
                        args="encoder.wave_list.optical.B2=842 encoder.wave_list.optical.B3=2190 encoder.wave_list.optical.B4=490 encoder.wave_list.optical.B8=1610 encoder.wave_list.optical.B11=560 encoder.wave_list.optical.B12=665"
                    fi
                    ;;
                3)
                    if [ "$dsgroup" = "mados" ]; then
                        args="encoder.wave_list.optical.B1=560 encoder.wave_list.optical.B2=740 encoder.wave_list.optical.B3=2190 encoder.wave_list.optical.B4=443 encoder.wave_list.optical.B5=665 encoder.wave_list.optical.B6=1610 encoder.wave_list.optical.B7=490 encoder.wave_list.optical.B8=783 encoder.wave_list.optical.B8A=842 encoder.wave_list.optical.B11=705 encoder.wave_list.optical.B12=865"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="encoder.wave_list.optical.B1=842 encoder.wave_list.optical.B2=1610 encoder.wave_list.optical.B3=783 encoder.wave_list.optical.B4=443 encoder.wave_list.optical.B5=2190 encoder.wave_list.optical.B6=705 encoder.wave_list.optical.B7=560 encoder.wave_list.optical.B8=490 encoder.wave_list.optical.B8A=740 encoder.wave_list.optical.B9=665 encoder.wave_list.optical.B11=865 encoder.wave_list.optical.B12=945"
                    else
                        args="encoder.wave_list.optical.B2=842 encoder.wave_list.optical.B3=665 encoder.wave_list.optical.B4=1610 encoder.wave_list.optical.B8=2190 encoder.wave_list.optical.B11=490 encoder.wave_list.optical.B12=560"
                    fi
                    ;;
            esac
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=2.19 +encoder.band_wavelength_overrides.optical.B2=0.865 +encoder.band_wavelength_overrides.optical.B3=0.665 +encoder.band_wavelength_overrides.optical.B4=0.49 +encoder.band_wavelength_overrides.optical.B5=0.842 +encoder.band_wavelength_overrides.optical.B6=0.443 +encoder.band_wavelength_overrides.optical.B7=0.74 +encoder.band_wavelength_overrides.optical.B8=0.705 +encoder.band_wavelength_overrides.optical.B8A=1.61 +encoder.band_wavelength_overrides.optical.B11=0.783 +encoder.band_wavelength_overrides.optical.B12=0.56"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.49 +encoder.band_wavelength_overrides.optical.B2=0.705 +encoder.band_wavelength_overrides.optical.B3=0.842 +encoder.band_wavelength_overrides.optical.B4=0.443 +encoder.band_wavelength_overrides.optical.B5=0.783 +encoder.band_wavelength_overrides.optical.B6=1.61 +encoder.band_wavelength_overrides.optical.B7=0.945 +encoder.band_wavelength_overrides.optical.B8=2.19 +encoder.band_wavelength_overrides.optical.B8A=0.56 +encoder.band_wavelength_overrides.optical.B9=0.865 +encoder.band_wavelength_overrides.optical.B11=0.665 +encoder.band_wavelength_overrides.optical.B12=0.74"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=2.19 +encoder.band_wavelength_overrides.optical.B3=0.842 +encoder.band_wavelength_overrides.optical.B4=0.49 +encoder.band_wavelength_overrides.optical.B8=0.665 +encoder.band_wavelength_overrides.optical.B11=0.56 +encoder.band_wavelength_overrides.optical.B12=1.61"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.705 +encoder.band_wavelength_overrides.optical.B2=0.74 +encoder.band_wavelength_overrides.optical.B3=1.61 +encoder.band_wavelength_overrides.optical.B4=0.865 +encoder.band_wavelength_overrides.optical.B5=2.19 +encoder.band_wavelength_overrides.optical.B6=0.49 +encoder.band_wavelength_overrides.optical.B7=0.665 +encoder.band_wavelength_overrides.optical.B8=0.443 +encoder.band_wavelength_overrides.optical.B8A=0.783 +encoder.band_wavelength_overrides.optical.B11=0.56 +encoder.band_wavelength_overrides.optical.B12=0.842"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=1.61 +encoder.band_wavelength_overrides.optical.B2=0.783 +encoder.band_wavelength_overrides.optical.B3=0.74 +encoder.band_wavelength_overrides.optical.B4=0.705 +encoder.band_wavelength_overrides.optical.B5=0.865 +encoder.band_wavelength_overrides.optical.B6=0.945 +encoder.band_wavelength_overrides.optical.B7=2.19 +encoder.band_wavelength_overrides.optical.B8=0.49 +encoder.band_wavelength_overrides.optical.B8A=0.665 +encoder.band_wavelength_overrides.optical.B9=0.842 +encoder.band_wavelength_overrides.optical.B11=0.443 +encoder.band_wavelength_overrides.optical.B12=0.56"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=0.842 +encoder.band_wavelength_overrides.optical.B3=2.19 +encoder.band_wavelength_overrides.optical.B4=0.49 +encoder.band_wavelength_overrides.optical.B8=1.61 +encoder.band_wavelength_overrides.optical.B11=0.56 +encoder.band_wavelength_overrides.optical.B12=0.665"
                    fi
                    ;;
                3)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.56 +encoder.band_wavelength_overrides.optical.B2=0.74 +encoder.band_wavelength_overrides.optical.B3=2.19 +encoder.band_wavelength_overrides.optical.B4=0.443 +encoder.band_wavelength_overrides.optical.B5=0.665 +encoder.band_wavelength_overrides.optical.B6=1.61 +encoder.band_wavelength_overrides.optical.B7=0.49 +encoder.band_wavelength_overrides.optical.B8=0.783 +encoder.band_wavelength_overrides.optical.B8A=0.842 +encoder.band_wavelength_overrides.optical.B11=0.705 +encoder.band_wavelength_overrides.optical.B12=0.865"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.band_wavelength_overrides.optical.B1=0.842 +encoder.band_wavelength_overrides.optical.B2=1.61 +encoder.band_wavelength_overrides.optical.B3=0.783 +encoder.band_wavelength_overrides.optical.B4=0.443 +encoder.band_wavelength_overrides.optical.B5=2.19 +encoder.band_wavelength_overrides.optical.B6=0.705 +encoder.band_wavelength_overrides.optical.B7=0.56 +encoder.band_wavelength_overrides.optical.B8=0.49 +encoder.band_wavelength_overrides.optical.B8A=0.74 +encoder.band_wavelength_overrides.optical.B9=0.665 +encoder.band_wavelength_overrides.optical.B11=0.865 +encoder.band_wavelength_overrides.optical.B12=0.945"
                    else
                        args="+encoder.band_wavelength_overrides.optical.B2=0.842 +encoder.band_wavelength_overrides.optical.B3=0.665 +encoder.band_wavelength_overrides.optical.B4=1.61 +encoder.band_wavelength_overrides.optical.B8=2.19 +encoder.band_wavelength_overrides.optical.B11=0.49 +encoder.band_wavelength_overrides.optical.B12=0.56"
                    fi
                    ;;
            esac
            ;;
        smarties_vitb)  # discrete projection index, override dict
            case "$draw" in
                1)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.proj_idx_override.optical.B1=14 +encoder.proj_idx_override.optical.B2=11 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B4=1 +encoder.proj_idx_override.optical.B5=9 +encoder.proj_idx_override.optical.B6=0 +encoder.proj_idx_override.optical.B7=8 +encoder.proj_idx_override.optical.B8=7 +encoder.proj_idx_override.optical.B8A=13 +encoder.proj_idx_override.optical.B11=10 +encoder.proj_idx_override.optical.B12=4"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.proj_idx_override.optical.B1=1 +encoder.proj_idx_override.optical.B2=7 +encoder.proj_idx_override.optical.B3=9 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B5=10 +encoder.proj_idx_override.optical.B6=13 +encoder.proj_idx_override.optical.B7=12 +encoder.proj_idx_override.optical.B8=14 +encoder.proj_idx_override.optical.B8A=4 +encoder.proj_idx_override.optical.B9=11 +encoder.proj_idx_override.optical.B11=6 +encoder.proj_idx_override.optical.B12=8"
                    else
                        args="+encoder.proj_idx_override.optical.B2=14 +encoder.proj_idx_override.optical.B3=9 +encoder.proj_idx_override.optical.B4=1 +encoder.proj_idx_override.optical.B8=6 +encoder.proj_idx_override.optical.B11=4 +encoder.proj_idx_override.optical.B12=13"
                    fi
                    ;;
                2)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.proj_idx_override.optical.B1=7 +encoder.proj_idx_override.optical.B2=8 +encoder.proj_idx_override.optical.B3=13 +encoder.proj_idx_override.optical.B4=11 +encoder.proj_idx_override.optical.B5=14 +encoder.proj_idx_override.optical.B6=1 +encoder.proj_idx_override.optical.B7=6 +encoder.proj_idx_override.optical.B8=0 +encoder.proj_idx_override.optical.B8A=10 +encoder.proj_idx_override.optical.B11=4 +encoder.proj_idx_override.optical.B12=9"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.proj_idx_override.optical.B1=13 +encoder.proj_idx_override.optical.B2=10 +encoder.proj_idx_override.optical.B3=8 +encoder.proj_idx_override.optical.B4=7 +encoder.proj_idx_override.optical.B5=11 +encoder.proj_idx_override.optical.B6=12 +encoder.proj_idx_override.optical.B7=14 +encoder.proj_idx_override.optical.B8=1 +encoder.proj_idx_override.optical.B8A=6 +encoder.proj_idx_override.optical.B9=9 +encoder.proj_idx_override.optical.B11=0 +encoder.proj_idx_override.optical.B12=4"
                    else
                        args="+encoder.proj_idx_override.optical.B2=9 +encoder.proj_idx_override.optical.B3=14 +encoder.proj_idx_override.optical.B4=1 +encoder.proj_idx_override.optical.B8=13 +encoder.proj_idx_override.optical.B11=4 +encoder.proj_idx_override.optical.B12=6"
                    fi
                    ;;
                3)
                    if [ "$dsgroup" = "mados" ]; then
                        args="+encoder.proj_idx_override.optical.B1=4 +encoder.proj_idx_override.optical.B2=8 +encoder.proj_idx_override.optical.B3=14 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B5=6 +encoder.proj_idx_override.optical.B6=13 +encoder.proj_idx_override.optical.B7=1 +encoder.proj_idx_override.optical.B8=10 +encoder.proj_idx_override.optical.B8A=9 +encoder.proj_idx_override.optical.B11=7 +encoder.proj_idx_override.optical.B12=11"
                    elif [ "$dsgroup" = "s2_12b" ]; then
                        args="+encoder.proj_idx_override.optical.B1=9 +encoder.proj_idx_override.optical.B2=13 +encoder.proj_idx_override.optical.B3=10 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B5=14 +encoder.proj_idx_override.optical.B6=7 +encoder.proj_idx_override.optical.B7=4 +encoder.proj_idx_override.optical.B8=1 +encoder.proj_idx_override.optical.B8A=8 +encoder.proj_idx_override.optical.B9=6 +encoder.proj_idx_override.optical.B11=11 +encoder.proj_idx_override.optical.B12=12"
                    else
                        args="+encoder.proj_idx_override.optical.B2=9 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B4=13 +encoder.proj_idx_override.optical.B8=14 +encoder.proj_idx_override.optical.B11=1 +encoder.proj_idx_override.optical.B12=4"
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

    local -r job_name="p4wl_${enc}_${ds}_${draw}"
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
readonly DRAWS="1 2 3"

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
echo "All P4 wavelength full-shuffle ablation jobs queued (up to 72 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
