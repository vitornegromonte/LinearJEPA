#!/bin/bash
#SBATCH --job-name=linearjepa_synth
#SBATCH --partition=short-simple
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --nodes=9
#SBATCH -o synth_%j.out
#SBATCH -e synth_%j.err
#SBATCH --time=04:00:00

set -eo pipefail

echo "==========================================="
echo " LinearJEPA — Synthetic Dataset Pipeline"
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
# Ativar ambiente
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
echo "Tarefas: $TASKS"
echo "Modelos: $MODELS"

python bench_generalization.py \
    --task "$TASKS" \
    --modality state \
    --models "$MODELS" \
    --episodes $EPISODES --steps $STEPS --seeds $SEEDS \
    --out results/benchmark.csv \
    || echo "⚠️  Benchmark falhou"

echo ""
echo "==========================================="
echo " 3. Speed benchmark (all tasks, fair params)"
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
