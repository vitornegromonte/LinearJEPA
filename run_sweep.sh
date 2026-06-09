#!/bin/bash
#SBATCH --job-name=lewm_sweep
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o sweep_%j.out
#SBATCH -e sweep_%j.err
#SBATCH --time=04:00:00

set -eo pipefail

echo "==========================================="
echo " LeWM Hyperparameter Sweep"
echo " Nó: $(hostname)"
echo " Usuário: $USER"
echo " Data/Hora: $(date)"
echo "==========================================="

# ---------------------------
# Checar GPU
# ---------------------------
echo "🔍 Checando GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi não encontrado — CPU only."
else
    nvidia-smi || echo "Falha ao listar GPUs."
fi

# ---------------------------
# Ativar ambiente (venv)
# ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "🔧 Ambiente venv ativado: $(which python)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate lewm || { echo "❌ Falha ao ativar conda env"; exit 1; }
    echo "🔧 Ambiente Conda ativado: $(which python)"
else
    echo "⚠️  Nenhum ambiente Python encontrado, usando python padrão"
fi

# ---------------------------
# Confirmar PyTorch + CUDA
# ---------------------------
echo "🔎 Verificando PyTorch e CUDA..."
python3 - <<'EOF'
import torch, sys
print(f"Torch: {torch.__version__}")
print(f"Python: {sys.executable}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
from module import DeltaNetAttention
d = DeltaNetAttention(192, heads=8, dim_head=64)
print(f"DeltaNet otimizado: {d._use_optimized}")
EOF

echo ""
echo "==========================================="
echo " 1. Speed benchmark (L1)"
echo "==========================================="
export CUDA_VISIBLE_DEVICES=0

python bench_hparam_sweep.py \
    --model lewm,lewm_deltanet,lewm_mamba \
    --fidelity L1 \
    --n-samples 200 \
    --batch-size 32 \
    --speed-trials 100 \
    --out results/sweep_speed \
    || { echo "⚠️  L1 sweep falhou"; exit 1; }

echo ""
echo "==========================================="
echo " 2. Pareto analysis (speed × params)"
echo "==========================================="
python bench_hparam_sweep.py \
    --analyze results/sweep_speed_L1.csv \
    --out results/pareto_speed.csv \
    || echo "⚠️  Análise Pareto falhou"

echo ""
echo "==========================================="
echo " 3. Full L2 sweep (train + quality)"
echo "==========================================="
python bench_hparam_sweep.py \
    --model lewm,lewm_deltanet,lewm_mamba \
    --fidelity L2 \
    --n-samples 100 \
    --task linear_ar \
    --episodes 250 \
    --train-steps 500 \
    --seeds 3 \
    --batch-size 32 \
    --speed-trials 50 \
    --out results/sweep_full \
    || echo "⚠️  L2 sweep falhou"

echo ""
echo "==========================================="
echo " Job finalizado com sucesso!"
echo " Data/Hora: $(date)"
echo "==========================================="
