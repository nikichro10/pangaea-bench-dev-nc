#!/bin/bash
# Run this script on the login node (internet access required).
# Downloads all 6 GeoBench classification datasets directly via huggingface_hub.

set -e

MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"
export GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"

export MAMBA_ROOT_PREFIX="/p/project1/hai_1280/micromamba"
eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate "$ENV_PATH"

echo "=== Downloading GeoBench classification datasets ==="

python - << 'EOF'
import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

geo_bench_dir = Path(os.environ["GEO_BENCH_DIR"])
repo_id = "recursix/geo-bench-1.0"

# classification_v1.0/normalizer.json + one zip per dataset
# Note: mbrickkiln.py in the codebase has a bug (downloads m-pv4ger.zip),
# using the correct filename m-brick-kiln.zip here.
files = [
    "classification_v1.0/normalizer.json",
    "classification_v1.0/m-bigearthnet.zip",
    "classification_v1.0/m-so2sat.zip",
    "classification_v1.0/m-brick-kiln.zip",
    "classification_v1.0/m-forestnet.zip",
    "classification_v1.0/m-eurosat.zip",
    "classification_v1.0/m-pv4ger.zip",
]

for filename in files:
    dest = geo_bench_dir / filename
    if dest.exists():
        print(f"Already exists, skipping: {filename}")
        continue

    print(f"\nDownloading {filename} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=geo_bench_dir,
        repo_type="dataset",
    )

    if filename.endswith(".zip"):
        print(f"Extracting {filename} ...")
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(geo_bench_dir / "classification_v1.0")
        print(f"Done: {filename}")

print("\nAll datasets downloaded and extracted.")
EOF

echo ""
echo "Datasets saved to: ${GEO_BENCH_DIR}/classification_v1.0/"
