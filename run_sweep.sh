#!/bin/bash
#SBATCH --job-name=linearjepa_synth_sweep
#SBATCH -p short-simple
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o job.log
#SBATCH -w cluster-node8
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=08:00:00

set -eo pipefail

echo "==========================================="
echo " LinearJEPA — Synthetic Dataset Pipeline"
echo " Nó: $(hostname)"
echo " Usuário: $USER"
echo " Data/Hora: $(date)"
echo "==========================================="

# ---------------------------
# Ativar ambiente Python (Apuana)
# ---------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lewm_py310
echo "🔧 Ambiente: $(which python)"

# ---------------------------
# Ir para o projeto
# ---------------------------
cd "$HOME/LinearJEPA" || { echo "❌ Diretório $HOME/LinearJEPA não encontrado"; exit 1; }
echo "📁 Diretório: $(pwd)"

# ---------------------------
# Checar GPU
# ---------------------------
echo "🔍 GPU:"
nvidia-smi || echo "Falha ao listar GPUs."

# ---------------------------
# Confirmar PyTorch + CUDA
# ---------------------------
echo "🔎 PyTorch:"
python3 - <<'EOF'
import torch, sys
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
EOF

export CUDA_VISIBLE_DEVICES=0

TASKS="linear_ar,nback,delayed_copy,slowfast,chaotic"
MODELS="lewm,lewm_deltanet,lewm_mamba"
EPISODES=200
STEPS=2000
SEEDS=3

echo ""
echo "==========================================="
echo " 1. Gerando datasets sintéticos"
echo "==========================================="

for task in linear_ar nback delayed_copy slowfast chaotic; do
    echo "--- $task ---"
    python synth_data.py --task "$task" --modality state \
        --num-episodes $EPISODES --ep-len 128
done

echo ""
echo "==========================================="
echo " 2. Step-generalization benchmark"
echo "==========================================="

python bench_generalization.py \
    --task "$TASKS" \
    --modality state \
    --models "$MODELS" \
    --episodes $EPISODES --steps $STEPS --seeds $SEEDS \
    --out results/benchmark.csv \
    || echo "⚠️  Benchmark falhou"

echo ""
echo "==========================================="
echo " 3. Speed benchmark (fair params)"
echo "==========================================="

python bench_generalization.py \
    --task "$TASKS" \
    --modality state \
    --models "$MODELS" \
    --episodes $EPISODES --steps $STEPS --seeds $SEEDS \
    --bench-speed --match-params \
    --speed-lens 16,32,64,128,256 \
    --batch-size 128 --speed-trials 100 \
    --out results/benchmark_speed.csv \
    || echo "⚠️  Speed benchmark falhou"

echo ""
echo "==========================================="
echo " Job finalizado com sucesso!"
echo " Data/Hora: $(date)"
echo "==========================================="
