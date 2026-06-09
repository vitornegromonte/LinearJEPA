#!/bin/bash
# Testes para bench_hparam_sweep.py — multi-fidelity hyperparameter sweep.
#
# Uso:
#   ./test_sweep.sh                    # rodar todos os testes
#   ./test_sweep.sh --gpu              # forçar GPU (default: auto)

set -eo pipefail

echo "==========================================="
echo " Test: bench_hparam_sweep.py"
echo " Nó: $(hostname)"
echo " Data: $(date)"
echo "==========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ativar ambiente (venv ou conda)
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate lewm 2>/dev/null || true
fi

echo "🔎 Python: $(which python)"
echo ""

# Os testes são em Python — executa direto
python3 - <<'PYEOF'
import sys, time
sys.path.insert(0, ".")

# Patch DeltaNet CPU fallback
import torch
from module import DeltaNetAttention as _DNA
import types
def _cpu_init(self, dim, heads=8, dim_head=64, dropout=0.0):
    torch.nn.Module.__init__(self)
    self._use_optimized = False
    inner_dim = dim_head * heads
    self.heads = heads
    self.norm = torch.nn.LayerNorm(dim)
    self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
    self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, dim), torch.nn.Dropout(dropout))
_DNA.__init__ = _cpu_init

from bench_hparam_sweep import (
    SEARCH_SPACES, sample_configs, eval_L0, eval_speed, eval_quality,
    build_predictor, pareto_frontier, analyze_pareto,
)

PASS = 0; FAIL = 0
def ok(m): global PASS; PASS += 1; print(f"  ✓ {m}")
def fail(m): global FAIL; FAIL += 1; print(f"  ✗ {m}")

# ---- Test 1: Search spaces ----
print("  Test 1: Search space definitions")
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    ok(f"space '{name}' with {len(SEARCH_SPACES[name])} params") if name in SEARCH_SPACES else fail(f"missing {name}")

# ---- Test 2: Config sampling ----
print("\n  Test 2: Config sampling")
c1 = sample_configs(["lewm"], 10)
ok(f"sampled {len(c1)} configs") if len(c1) == 10 else fail(f"expected 10")
ok("unique IDs") if len(set(c["config_id"] for c in c1)) == 10 else fail("duplicate IDs")
c2 = sample_configs(["lewm", "lewm_mamba"], 20)
ok(f"multi-model: {len(c2)} configs") if len(c2) == 20 else fail("expected 20")

# ---- Test 3: L0 ----
print("\n  Test 3: L0 param count")
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    l0 = eval_L0(cfg)
    ok(f"{name}: {l0['params']:,} params") if l0["params"] > 0 else fail("zero")

# ---- Test 4: L1 speed ----
print("\n  Test 4: L1 speed benchmark")
for name in ("lewm", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    spd = eval_speed(cfg, lengths=(16, 32), batch_size=4, n_trials=5)
    ok(f"{name}: forward_ms_16={spd['forward_ms_16']:.1f}ms") if spd["forward_ms_16"] > 0 else fail("zero")
cfg = sample_configs(["lewm_deltanet"], 1)[0]
spd = eval_speed(cfg, lengths=(16,), batch_size=2, n_trials=3)
ok(f"lewm_deltanet: forward_ms_16={spd['forward_ms_16']:.1f}ms") if spd["forward_ms_16"] > 0 else fail("zero")

# ---- Test 5: L2 quality ----
print("\n  Test 5: L2 train + quality")
cfg = sample_configs(["lewm"], 1)[0]
t0 = time.time()
qual = eval_quality(cfg, episodes=10, eval_lens=(16, 32), train_steps=50, batch_size=16, seed=0)
elapsed = time.time() - t0
ok(f"trained in {elapsed:.1f}s") if elapsed < 120 else fail(f"too slow: {elapsed:.1f}s")
for L in (16, 32):
    ok(f"  mean_mse_{L}={qual[f'mean_mse_{L}']:.6f}") if qual[f"mean_mse_{L}"] > 0 else fail("zero")

# ---- Test 6: Pareto ----
print("\n  Test 6: Pareto frontier")
dummy = [
    {"config_id": "a", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "1000000"},
    {"config_id": "b", "forward_tok_s_64": "2000", "mean_mse_64": "0.2", "params": "2000000"},
    {"config_id": "c", "forward_tok_s_64": "500",  "mean_mse_64": "0.05", "params": "500000"},
]
f = pareto_frontier(dummy, maximize=["forward_tok_s_64"], minimize=["mean_mse_64", "params"])
ok(f"frontier: {len(f)} non-dominated") if len(f) == 3 else fail(f"expected 3, got {len(f)}")
dummy2 = [
    {"config_id": "a", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "1000000"},
    {"config_id": "b", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "2000000"},
]
f2 = pareto_frontier(dummy2, maximize=["forward_tok_s_64"], minimize=["mean_mse_64", "params"])
ok(f"a dominates b ({len(f2)} front)") if len(f2) == 1 else fail("wrong")

# ---- Test 7: build_predictor ----
print("\n  Test 7: Build predictor")
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    m = build_predictor(name, cfg, num_frames=16)
    p = sum(p.numel() for p in m.parameters())
    ok(f"{name}: {p:,} params") if p > 0 else fail("zero")

# ---- Summary ----
print(f"\n{'='*50}")
print(f"  RESULTADO: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
PYEOF
