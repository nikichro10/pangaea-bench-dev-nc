OUTDIR="cluster_smi_hpda2_compute_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

FAILED_FILE="${OUTDIR}/failed_nodes.txt"
: > "$FAILED_FILE"  # truncate

# Expand partition nodelist -> one hostname per line
NODES="$(scontrol show hostnames "$(sinfo -h -p hpda2_compute_gpu -o %N | paste -sd, -)")"

# Ensure Ctrl+C stops everything cleanly
trap 'echo "Interrupted. Killing running cluster-smi..."; pkill -P $$ cluster-smi 2>/dev/null; exit 130' INT TERM

failed_csv=""

for node in $NODES; do
  echo "Querying $node ..."
  outfile="${OUTDIR}/${node}.txt"

  # Avoid hangs: kill after 15s (tune if you want)
  timeout 1s cluster-smi -n "^${node}\$" > "$outfile" 2>&1
  rc=$?

  # Consider nonzero/timeout as failed
  if grep -q "NVIDIA A100" "$outfile"; then
    is_failed=0
  else
    is_failed=1
  fi

  if [ "$is_failed" -eq 1 ]; then
    if [ -z "$failed_csv" ]; then
      failed_csv="$node"
    else
      failed_csv="${failed_csv},${node}"
    fi
  fi
done

printf "%s" "$failed_csv" > "$FAILED_FILE"
echo "Done. Per-node outputs in: $OUTDIR"
echo "Failed nodes CSV in: $FAILED_FILE"