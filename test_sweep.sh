#!/usr/bin/env python3
"""Tests for bench_hparam_sweep.py — multi-fidelity hyperparameter sweep."""

import csv
import os
import sys
import time

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
    build_predictor, _config_id, pareto_frontier, analyze_pareto,
)

PASS = 0
FAIL = 0
OUT_DIR = "results"


def ok(msg):
    global PASS; PASS += 1
    print(f"  ✓ {msg}")

def fail(msg):
    global FAIL; FAIL += 1
    print(f"  ✗ {msg}")


# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Test 1: Search space definitions")
print("=" * 60)
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    ok(f"space '{name}' defined") if name in SEARCH_SPACES else fail(f"missing space '{name}'")
    ok(f"  {len(SEARCH_SPACES[name])} params") if len(SEARCH_SPACES[name]) >= 3 else fail("too few params")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 2: Config sampling")
print("=" * 60)
cfg1 = sample_configs(["lewm"], 10)
ok(f"sampled {len(cfg1)} lewm configs") if len(cfg1) == 10 else fail(f"expected 10, got {len(cfg1)}")
for cfg in cfg1:
    ok(f"  {cfg['config_id']} depth={cfg['depth']} hidden_dim={cfg['hidden_dim']}") if all(k in cfg for k in ("depth", "hidden_dim", "mlp_dim")) else fail("missing keys")
ids = [c["config_id"] for c in cfg1]
ok("unique IDs") if len(set(ids)) == len(ids) else fail("duplicate IDs")

# Multi-model
cfg2 = sample_configs(["lewm", "lewm_mamba"], 20)
ok(f"multi-model: {len(cfg2)} configs") if len(cfg2) == 20 else fail(f"expected 20, got {len(cfg2)}")
models = set(c["model"] for c in cfg2)
ok(f"  models: {models}") if "lewm" in models and "lewm_mamba" in models else fail("missing models")

# Target params
cfg3 = sample_configs(["lewm_deltanet"], 8, target_params=[1e6, 1e7])
ok(f"target-param: {len(cfg3)} configs") if len(cfg3) == 8 else fail(f"expected 8")
# Params only available after eval_L0 (configs are sampled without params)
ok("no params until eval_L0 (expected)") if not all("params" in c for c in cfg3) else fail("unexpected params")
params = [eval_L0(c)["params"] for c in cfg3]
ok(f"params range: {min(params):,} – {max(params):,}") if all(p > 0 for p in params) else fail("zero params")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 3: L0 param count")
print("=" * 60)
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    l0 = eval_L0(cfg)
    ok(f"{name}: {l0['params']:,} params") if l0["params"] > 0 else fail("zero")
    ok(f"  trainable matches total") if l0["params"] == l0["trainable_params"] else fail("mismatch")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 4: L1 speed benchmark")
print("=" * 60)
for name in ("lewm", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    spd = eval_speed(cfg, lengths=(16, 32), batch_size=4, n_trials=5)
    ok(f"{name}: forward_ms_16={spd['forward_ms_16']:.1f}ms") if spd["forward_ms_16"] > 0 else fail("zero")
    ok(f"  tok_s_32={spd['forward_tok_s_32']:.0f}") if spd["forward_tok_s_32"] > 0 else fail("zero tok/s")
    ok(f"  rollout_step_ms_16={spd['rollout_step_ms_16']:.2f}ms") if spd["rollout_step_ms_16"] > 0 else fail("zero rollout")
    ok(f"  params match: {spd['params']:,}") if spd["params"] > 0 else fail("zero params")

# DeltaNet needs CPU patch active
cfg = sample_configs(["lewm_deltanet"], 1)[0]
spd = eval_speed(cfg, lengths=(16,), batch_size=2, n_trials=3)
ok(f"lewm_deltanet: forward_ms_16={spd['forward_ms_16']:.1f}ms") if spd["forward_ms_16"] > 0 else fail("zero")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 5: L2 train + quality")
print("=" * 60)
cfg = sample_configs(["lewm"], 1)[0]
t0 = time.time()
qual = eval_quality(cfg, task="linear_ar", episodes=10, eval_lens=(16, 32),
                    train_steps=50, batch_size=16, seed=0)
elapsed = time.time() - t0
ok(f"trained in {elapsed:.1f}s") if elapsed < 120 else fail(f"too slow: {elapsed:.1f}s")
ok(f"  train_time_s={qual['train_time_s']}") if qual["train_time_s"] > 0 else fail("zero")
for L in (16, 32):
    ok(f"  mean_mse_{L}={qual[f'mean_mse_{L}']:.6f}") if qual[f"mean_mse_{L}"] > 0 else fail("zero mse")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 6: Pareto frontier")
print("=" * 60)
dummy = [
    {"config_id": "a", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "1000000"},
    {"config_id": "b", "forward_tok_s_64": "2000", "mean_mse_64": "0.2", "params": "2000000"},
    {"config_id": "c", "forward_tok_s_64": "500",  "mean_mse_64": "0.05", "params": "500000"},
]
frontier = pareto_frontier(dummy, maximize=["forward_tok_s_64"], minimize=["mean_mse_64", "params"])
ids = sorted(r["config_id"] for r in frontier)
# All three are non-dominated: a is balanced, b is fastest, c is most accurate/efficient
ok(f"frontier: all 3 non-dominated ({ids})") if len(frontier) == 3 else fail(f"expected 3, got {len(frontier)}")

# Test with equal metrics (only a should dominate)
dummy2 = [
    {"config_id": "a", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "1000000"},
    {"config_id": "b", "forward_tok_s_64": "1000", "mean_mse_64": "0.1", "params": "2000000"},
]
frontier2 = pareto_frontier(dummy2, maximize=["forward_tok_s_64"], minimize=["mean_mse_64", "params"])
ok(f"equal metrics → a dominates b") if len(frontier2) == 1 and frontier2[0]["config_id"] == "a" else fail("wrong")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 7: Build predictor from config")
print("=" * 60)
for name in ("lewm", "lewm_deltanet", "lewm_mamba"):
    cfg = sample_configs([name], 1)[0]
    model = build_predictor(name, cfg, num_frames=16)
    params = sum(p.numel() for p in model.parameters())
    ok(f"{name}: built, {params:,} params") if params > 0 else fail("zero params on built model")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 8: end-to-end L0-only CLI")
print("=" * 60)
import subprocess
p = subprocess.run(
    [sys.executable, "bench_hparam_sweep.py",
     "--model", "lewm,lewm_mamba",
     "--n-samples", "8",
     "--fidelity", "L0",
     "--out", f"{OUT_DIR}/e2e_test"],
    capture_output=True, text=True, timeout=120,
)
ok(f"CLI exited rc={p.returncode}") if p.returncode == 0 else fail(f"error: {p.stderr[:200]}")
csv_path = f"{OUT_DIR}/e2e_test_L0.csv"
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
ok(f"  L0 CSV: {len(rows)} rows") if len(rows) == 8 else fail(f"expected 8, got {len(rows)}")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 9: config_id determinism")
print("=" * 60)
a = sample_configs(["lewm"], 1, seed=42)
b = sample_configs(["lewm"], 1, seed=42)
ok(f"deterministic IDs") if a[0]["config_id"] == b[0]["config_id"] else fail("not deterministic")
c = sample_configs(["lewm"], 1, seed=99)
ok(f"different seed → different config") if a[0]["config_id"] != c[0]["config_id"] else fail("same config")

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)
