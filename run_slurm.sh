#!/bin/bash
#SBATCH --job-name=lewm-deltanet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --nodes=9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# --- CONFIGURATION (override via env vars) ---
MODEL="${MODEL:-lewm_deltanet}"
DATA="${DATA:-pusht}"
STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"
VENV_PATH="${VENV_PATH:-$PWD/.venv}"

# --- SETUP ---
mkdir -p logs "$STABLEWM_HOME"

# Activate environment
if [ -n "${VIRTUAL_ENV:-}" ]; then
    deactivate 2>/dev/null || true
fi
source "$VENV_PATH/bin/activate"

# Check dataset exists, download if missing
LANCE_PATH="$STABLEWM_HOME/pusht_expert_train.lance"
if [ ! -d "$LANCE_PATH" ]; then
    echo "Dataset not found at $LANCE_PATH — downloading..."
    python -c "
import os, zstandard, shutil
from huggingface_hub import hf_hub_download
STABLEWM_HOME = os.environ['STABLEWM_HOME']
h5_path = f'{STABLEWM_HOME}/pusht_expert_train.h5'
h5_zst = hf_hub_download(repo_id='quentinll/lewm-pusht', repo_type='dataset',
                          filename='pusht_expert_train.h5.zst')
with open(h5_zst, 'rb') as inp, open(h5_path, 'wb') as out:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(inp) as reader:
        shutil.copyfileobj(reader, out)
import stable_worldmodel as swm
swm.data.convert(h5_path, f'{STABLEWM_HOME}/pusht_expert_train.lance')
os.remove(h5_path)
print('Dataset ready')
"
fi

# --- TRAIN ---
export STABLEWM_HOME
export HYDRA_FULL_ERROR=1

python train.py \
    model="$MODEL" \
    data="$DATA" \
    "$@"
