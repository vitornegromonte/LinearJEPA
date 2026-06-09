#!/bin/bash
# Testes para bench_generalization.py — speed benchmarking.
#
# Uso:
#   ./test_speed_bench.sh              # CPU, dados mínimos
#   ./test_speed_bench.sh --gpu        # forçar GPU

set -eo pipefail

echo "==========================================="
echo " Test: bench_generalization.py speed bench"
echo " Nó: $(hostname)"
echo " Data: $(date)"
echo "==========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ativar ambiente
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate lewm 2>/dev/null || true
fi

echo "🔎 Python: $(which python)"
echo ""

# Testes em Python
python3 - <<'PYEOF'
import sys, time
sys.path.insert(0, ".")

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

from bench_generalization import bench_speed, _estimate_param_matched_depths

PASS = 0; FAIL = 0
def ok(m): global PASS; PASS += 1; print(f"  ✓ {m}")
def fail(m): global FAIL; FAIL += 1; print(f"  ✗ {m}")

EMBED = 64
BS = 4
TRIALS = 5
LENS = [16, 32]
MODELS = ["lewm", "lewm_deltanet", "lewm_mamba"]

# Test 1: Default speed bench
print("  Test 1: Speed benchmark (default depths)")
spd = bench_speed(MODELS, LENS, batch_size=BS, embed_dim=EMBED, n_trials=TRIALS, match_params=False)
by_model = {}
for r in spd:
    by_model.setdefault(r["model"], []).append(r)
ok(f"{len(spd)} rows") if len(spd) > 0 else fail("no rows")
for m in MODELS:
    ok(f"model '{m}' present") if m in by_model else fail(f"missing {m}")
for m in MODELS:
    ds = {r["depth"] for r in by_model[m]}
    ok(f"{m} depth={ds}") if ds == {6} else fail(f"unexpected depth {ds}")

# Test 2: Param-matched depths
print("\n  Test 2: Param-matched depth estimation")
dm = _estimate_param_matched_depths(MODELS, input_dim=EMBED, hidden_dim=EMBED, output_dim=EMBED)
ok("all models in depth_map") if set(dm.keys()) == set(MODELS) else fail("missing models")
ok("Mamba depth > Transformer") if dm["lewm_mamba"] > dm["lewm"] else fail("Mamba not scaled")

# Test 3: Speed with param matching
print("\n  Test 3: Speed benchmark (--match-params)")
spd2 = bench_speed(MODELS, LENS, batch_size=BS, embed_dim=EMBED, n_trials=TRIALS, match_params=True)
by_model = {}
for r in spd2:
    by_model.setdefault(r["model"], []).append(r)
params = {m: by_model[m][0]["params"] for m in MODELS}
ratio = max(params.values()) / min(params.values())
ok(f"param ratio <= 1.1x ({ratio:.3f}x)") if ratio <= 1.1 else fail(f"ratio {ratio:.3f}x")
depths = {m: by_model[m][0]["depth"] for m in MODELS}
ok(f"Mamba depth={depths['lewm_mamba']} (scaled)") if depths["lewm_mamba"] > 6 else fail("not scaled")

# Test 4: Metrics sanity
print("\n  Test 4: Metrics sanity")
for r in spd:
    ok(f"{r['model']}: forward_ms > 0 ({r['forward_ms']})") if r["forward_ms"] > 0 else fail("zero")
    ok(f"{r['model']}: tok_s > 0 ({r['forward_tok_s']})") if r["forward_tok_s"] > 0 else fail("zero")
    ok(f"{r['model']}: rollout_ms > 0 ({r['rollout_step_ms']})") if r["rollout_step_ms"] > 0 else fail("zero")

print(f"\n{'='*50}")
print(f"  RESULTADO: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
PYEOF
