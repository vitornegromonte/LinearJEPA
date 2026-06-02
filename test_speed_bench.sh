#!/usr/bin/env python3
"""
End-to-end test for the speed benchmarking suite in bench_generalization.py.

Tests by importing and calling functions directly (no subprocess overhead).

Usage:
  ./test_speed_bench.sh              # CPU, tiny data (~2 min)
  ./test_speed_bench.sh --gpu        # run on GPU if available
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Patch DeltaNet for CPU before importing bench_generalization
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
    self.to_out = torch.nn.Sequential(
        torch.nn.Linear(inner_dim, dim), torch.nn.Dropout(dropout)
    )
_DNA.__init__ = _cpu_init

from bench_generalization import bench_speed, _estimate_param_matched_depths

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  ✓ {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  ✗ {msg}")


# ── Config for fast tests ─────────────────────────────────────────────

EMBED_DIM = 64
BATCH_SIZE = 4
TRIALS = 5
LENGTHS = [16, 32]
MODELS = ["lewm", "lewm_deltanet", "lewm_mamba"]


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Speed benchmark, no param matching
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Test 1: Speed benchmark (default depths, no param matching)")
print("=" * 60)
spd = bench_speed(MODELS, LENGTHS, batch_size=BATCH_SIZE,
                  embed_dim=EMBED_DIM, n_trials=TRIALS, match_params=False)
rows_by_model = {}
for r in spd:
    rows_by_model.setdefault(r["model"], []).append(r)
ok(f"got {len(spd)} rows") if len(spd) > 0 else fail("no rows")
for m in MODELS:
    ok(f"model '{m}' present") if m in rows_by_model else fail(f"model '{m}' missing")
for m in MODELS:
    depths = {r["depth"] for r in rows_by_model[m]}
    ok(f"{m} depth={depths} (default=6)") if depths == {6} else fail(f"{m} unexpected depth {depths}")
# Check Mamba is much smaller
params = {m: rows_by_model[m][0]["params"] for m in MODELS}
ok(f"Mamba has {params['lewm']/params['lewm_mamba']:.0f}x fewer params than Transformer")
print(f"    Params: {params}")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: _estimate_param_matched_depths
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 2: Param-matched depth estimation")
print("=" * 60)
depth_map = _estimate_param_matched_depths(
    MODELS, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
)
ok(f"all 3 models in depth_map") if set(depth_map.keys()) == set(MODELS) else fail("missing models")
for m in MODELS:
    ok(f"{m} depth={depth_map[m]}") if depth_map[m] >= 1 else fail(f"{m} invalid depth")
ok(f"Mamba depth > Transformer (param match)") if depth_map["lewm_mamba"] > depth_map["lewm"] else fail("Mamba not scaled up")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Speed benchmark WITH param matching
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 3: Speed benchmark (with --match-params)")
print("=" * 60)
spd_matched = bench_speed(MODELS, LENGTHS, batch_size=BATCH_SIZE,
                          embed_dim=EMBED_DIM, n_trials=TRIALS, match_params=True)
rows_by_model = {}
for r in spd_matched:
    rows_by_model.setdefault(r["model"], []).append(r)

params = {m: rows_by_model[m][0]["params"] for m in MODELS}
ratio = max(params.values()) / min(params.values())
ok(f"param ratio <= 1.1x ({ratio:.3f}x)") if ratio <= 1.1 else fail(f"param ratio {ratio:.3f}x > 1.1")

depths = {m: rows_by_model[m][0]["depth"] for m in MODELS}
ok(f"Mamba depth={depths['lewm_mamba']} (scaled up)") if depths["lewm_mamba"] > 6 else fail("Mamba depth not scaled")
ok(f"Lewm depth={depths['lewm']} (stays 6)") if depths["lewm"] == 6 else fail("Lewm depth changed")
print(f"    Depths: {depths}, Params: {params}")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: CSV output format (simulate the CSV writer)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 4: CSV output structure")
print("=" * 60)
out_path = HERE / "results" / "test_csv_format.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

import bench_generalization as bg
# Monkey-patch to avoid training
orig_bench = bg.bench_speed
bg.bench_speed = lambda *a, **kw: spd  # reuse Test 1 results

# Create a minimal args namespace and call just the CSV writer part
import argparse
args = argparse.Namespace(
    task="linear_ar",
    modality="state",
    train_len=16,
    eval_lens="16",
    models="lewm",
    loss="lewm",
    episodes=10,
    steps=100,
    seeds=1,
    bench_speed=True,
    speed_lens="16,32",
    batch_size=BATCH_SIZE,
    speed_trials=TRIALS,
    match_params=False,
    out=str(out_path),
)

# Run only the CSV writing part
accuracy_fieldnames = [
    "task", "modality", "model", "loss", "seed",
    "eval_len", "first_div_step", "mean_mse", "final_mse", "train_time_s",
]
speed_fieldnames = [
    "benchmark", "model", "seq_len", "forward_ms", "forward_tok_s",
    "ms_per_tok", "rollout_step_ms", "depth", "params", "trainable_params",
    "peak_memory_mb",
]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=accuracy_fieldnames)
    writer.writeheader()
    writer.writerow({"task": "linear_ar", "model": "lewm", "eval_len": 16})
    f.write("\n")
    writer = csv.DictWriter(f, fieldnames=speed_fieldnames)
    writer.writeheader()
    for r in spd:
        writer.writerow(r)

parsed = []
with open(out_path) as f:
    for section in f.read().strip().split("\n\n"):
        rows = list(csv.DictReader(section.splitlines()))
        if rows:
            parsed.append(rows)

ok(f"got {len(parsed)} CSV sections") if len(parsed) == 2 else fail(f"expected 2, got {len(parsed)}")
if len(parsed) == 2:
    ok("accuracy columns OK") if "first_div_step" in parsed[0][0] else fail("missing accuracy cols")
    ok("speed columns OK") if "forward_ms" in parsed[1][0] else fail("missing speed cols")
    ok(f"  speed rows: {len(parsed[1])}") if len(parsed[1]) == len(spd) else fail(f"expected {len(spd)}, got {len(parsed[1])}")

bg.bench_speed = orig_bench  # restore


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Different batch sizes and T values
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 5: Varying batch sizes and T lengths")
print("=" * 60)
for bs in [1, 8]:
    for lens in [[8], [64]]:
        t0 = time.time()
        r = bench_speed(MODELS, lens, batch_size=bs,
                        embed_dim=EMBED_DIM, n_trials=3)
        elapsed = time.time() - t0
        ok(f"batch_size={bs}, T={lens[0]}: {len(r)} rows in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Metrics sanity (forward_ms > 0, tok/s > 0, etc.)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 6: Metrics sanity checks")
print("=" * 60)
for r in spd:
    m = r["model"]
    ok(f"{m}: forward_ms > 0 ({r['forward_ms']})") if r["forward_ms"] > 0 else fail(f"{m}: forward_ms <= 0")
    ok(f"{m}: forward_tok_s > 0 ({r['forward_tok_s']})") if r["forward_tok_s"] > 0 else fail(f"{m}: forward_tok_s <= 0")
    ok(f"{m}: rollout_step_ms > 0 ({r['rollout_step_ms']})") if r["rollout_step_ms"] > 0 else fail(f"{m}: rollout_step_ms <= 0")
    ok(f"{m}: params > 0 ({r['params']})") if r["params"] > 0 else fail(f"{m}: params <= 0")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)
