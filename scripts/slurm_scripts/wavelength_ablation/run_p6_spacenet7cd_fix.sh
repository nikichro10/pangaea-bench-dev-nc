#!/bin/bash
#
# P6 (Fully Random λ) SpaceNet7-CD rerun with the dice-collapse fix
#
# The original P6 SpaceNet7-CD runs (run_p6_random_wl_ablation.sh /
# run_p6_random_wl_ablation_upernet.sh, submitted 2026-07-22/23) used
# criterion=binary_cross_entropy against a dataset config whose `distribution`
# was a stale/wrong placeholder -- the same root cause that collapsed the
# regular S2-default SpaceNet7-CD benchmark under `dice`. spacenet7cd.yaml has
# since been corrected (measured distribution [0.99366, 0.00634]) and the
# benchmark-wide fix uses criterion=weighted_bce_sqrt + best_metric_key=IoU +
# data_replicate=2 (validated in Ablations/run_spacenet7cd_weighted_bce_sqrt_fix.sh,
# same pattern already used for the True-Wavelength SpaceNet7-CD rerun in this
# directory). This reruns P6's SpaceNet7-CD leg with that same fix -- the two
# random wavelength draws (seed 501/502, sn7cd group) and everything else about
# the ablation stay identical to the original P6 runs.
#
# Encoders: DOFA, DOFAv2, Panopticon, Copernicus-FM, CARL (imagewise) -- same
# 5-encoder scope as the rest of P6 (SMARTIES excluded -- no continuous
# wavelength input).
# Decoders: MLP-Fusion (seg_siam_mlp_fusion) + UPerNet (seg_siamupernet_conc).
# finetune=false (frozen encoder), matching the original ablation.
#
# Work dirs match the existing P6 folders exactly
# (p6_random_wl_ablation/{enc}_draw{n}, p6_random_wl_ablation_upernet/{enc}_draw{n})
# so this rerun lands alongside the old runs -- the notebook loader in
# analyze_wrong_wavelength_summary.ipynb already keeps the last (chronologically
# newest) run per dataset within a folder, so no notebook changes are needed
# once these complete.
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated keys (e.g. "dofa_mlp_1 carl_imagewise_up_2")
# -----------------------------------------------------------------------------

readonly SN7_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/spacenet7"
readonly LOG_BASE="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p6_spacenet7cd_fix"
readonly WORK_BASE_MLP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p6_random_wl_ablation"
readonly WORK_BASE_UP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p6_random_wl_ablation_upernet"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# enc_key -> "enc_cfg|preprocessing"
declare -A encoders=(
    ["dofa"]="dofa|seg_default"
    ["dofav2"]="dofav2|seg_default"
    ["panopticon"]="panopticon|seg_default"
    ["copernicus_fm_base"]="copernicus_fm_base|seg_default"
    ["carl_imagewise"]="carl_imagewise|seg_default_no_norm"
)

# draw -> literal sampled sn7cd values (B4/B3/B2), from p6_random_wl_draws.json --
# decoder-independent, identical to the original P6 SpaceNet7-CD runs.
wave_args() {
    local -r enc=$1 draw=$2
    case "$enc" in
        dofa|dofav2)  # micrometres
            case "$draw" in
                1) echo "encoder.wave_list.optical.B4=1.334215 encoder.wave_list.optical.B3=2.135323 encoder.wave_list.optical.B2=0.462891" ;;
                2) echo "encoder.wave_list.optical.B4=0.621842 encoder.wave_list.optical.B3=0.674262 encoder.wave_list.optical.B2=1.424243" ;;
            esac
            ;;
        panopticon|copernicus_fm_base)  # nanometres
            case "$draw" in
                1) echo "encoder.wave_list.optical.B4=1334.215 encoder.wave_list.optical.B3=2135.323 encoder.wave_list.optical.B2=462.891" ;;
                2) echo "encoder.wave_list.optical.B4=621.842 encoder.wave_list.optical.B3=674.262 encoder.wave_list.optical.B2=1424.243" ;;
            esac
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            case "$draw" in
                1) echo "+encoder.band_wavelength_overrides.optical.B4=1.334215 +encoder.band_wavelength_overrides.optical.B3=2.135323 +encoder.band_wavelength_overrides.optical.B2=0.462891" ;;
                2) echo "+encoder.band_wavelength_overrides.optical.B4=0.621842 +encoder.band_wavelength_overrides.optical.B3=0.674262 +encoder.band_wavelength_overrides.optical.B2=1.424243" ;;
            esac
            ;;
    esac
}

launch_job() {
    local -r enc=$1 draw=$2 decoder_family=$3   # "mlp" or "upernet"

    IFS='|' read -r enc_cfg preprocessing <<< "${encoders[$enc]}"

    local decoder work_base
    if [ "$decoder_family" = "upernet" ]; then
        decoder="seg_siamupernet_conc"; work_base="$WORK_BASE_UP"
    else
        decoder="seg_siam_mlp_fusion"; work_base="$WORK_BASE_MLP"
    fi

    local -r work_dir="${work_base}/${enc}_draw${draw}"
    local -r job_name="p6sn7fix_${enc}_${decoder_family}_${draw}"
    local -r sbatch_script="p6sn7fix_${enc}_${decoder_family}_${draw}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %-8s | draw %s]\n' "$enc" "$decoder_family" "$draw"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(wave_args "$enc" "$draw")"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=spacenet7cd"
    cmd+=" encoder=${enc_cfg}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=weighted_bce_sqrt"
    cmd+=" task=change_detection"
    cmd+=" finetune=false"
    cmd+=" dataset.root_path='${SN7_PATH}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=wavelength-ablation"
    cmd+=" task.trainer.best_metric_key=IoU"
    cmd+=" data_replicate=2"
    cmd+=" ${swap[*]}"

    printf '\nLaunching [%-20s | %-8s | draw %s]:\n%s\n' "$enc" "$decoder_family" "$draw" "$cmd"

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
#SBATCH --time=8:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_BASE}/${enc}_${decoder_family}_${draw}_output.%j
#SBATCH --error=${LOG_BASE}/${enc}_${decoder_family}_${draw}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

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

RUN_JOBS="${RUN_JOBS:-all}"
for enc in "${!encoders[@]}"; do
    for decoder_family in mlp upernet; do
        for draw in 1 2; do
            combo_key="${enc}_${decoder_family}_${draw}"
            if [ "$RUN_JOBS" != "all" ]; then
                [[ " $RUN_JOBS " != *" $combo_key "* ]] && continue
            fi
            launch_job "$enc" "$draw" "$decoder_family"
        done
    done
done

echo ""
echo "All P6 SpaceNet7-CD fix jobs queued (up to 20 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
