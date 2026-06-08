#!/usr/bin/env python3
"""
Multi-fidelity hyperparameter sweep for LeWM predictor architectures.

Evaluates model configs across three fidelities:
  L0 — Parameter count (analytical, fast)
  L1 — Speed benchmark (forward latency @ T=16,32,64; rollout step; memory)
  L2 — Train + rollout quality (MSE, first divergence step)

Configs are sampled via random search with optional biasing toward
specific parameter budgets (isocline sampling).  Results are saved
as CSV after each fidelity level, and the Pareto frontier is computed
automatically.

Usage:
  # Full L0+L1+L2 cascade
  python bench_hparam_sweep.py --model lewm_mamba --n-samples 200 \\
      --episodes 250 --train-steps 500 --out results/sweep

  # Speed only (L0+L1, no training)
  python bench_hparam_sweep.py --model lewm --fidelity L1 --n-samples 500 --out results/sweep

  # Analyze existing results
  python bench_hparam_sweep.py --analyze results/sweep_L2.csv --out results/pareto
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_generalization import (
    DEVICE, generate_dataset, train_model, evaluate_rollout,
    make_action_encoder, LinearEncoder,
)
from module import ARPredictor, MambaPredictor, ConditionalBlock, DeltaNetConditionalBlock

# ─────────────────────────────────────────────────────────────────────
#  Search spaces
# ─────────────────────────────────────────────────────────────────────

SEARCH_SPACES = {
    "lewm": OrderedDict([
        ("depth",       {"type": "int",     "range": [2, 12]}),
        ("hidden_dim",  {"type": "logint",  "range": [64, 384]}),
        ("heads",       {"type": "choice",  "values": [4, 8, 16]}),
        ("dim_head",    {"type": "choice",  "values": [32, 64]}),
        ("mlp_mult",    {"type": "choice",  "values": [2, 4, 8]}),
    ]),
    "lewm_deltanet": OrderedDict([
        ("depth",       {"type": "int",     "range": [2, 12]}),
        ("hidden_dim",  {"type": "logint",  "range": [64, 384]}),
        ("heads",       {"type": "choice",  "values": [4, 8, 16]}),
        ("dim_head",    {"type": "choice",  "values": [32, 64]}),
        ("mlp_mult",    {"type": "choice",  "values": [2, 4, 8]}),
    ]),
    "lewm_mamba": OrderedDict([
        ("depth",       {"type": "int",     "range": [2, 48]}),
        ("hidden_dim",  {"type": "logint",  "range": [64, 384]}),
        ("d_state",     {"type": "int",     "range": [8, 32]}),
        ("d_conv",      {"type": "choice",  "values": [4, 8]}),
        ("expand",      {"type": "choice",  "values": [1, 2, 4]}),
    ]),
}

# Approximate per-block param cost for isocline targeting
_BLOCK_PARAM_EST = {
    "lewm": {
        64: 551680, 128: 945152, 192: 1798400, 256: 2839808, 384: 5805056,
    },
    "lewm_deltanet": {
        64: 551680, 128: 945152, 192: 1798400, 256: 2839808, 384: 5805056,
    },
    "lewm_mamba": {
        64: 64512, 128: 129024, 192: 537600, 256: 258048, 384: 516096,
    },
}
_BASE_PARAM_EST = {
    "lewm": 960, "lewm_deltanet": 960, "lewm_mamba": 37440,
}


# ─────────────────────────────────────────────────────────────────────
#  Config generation
# ─────────────────────────────────────────────────────────────────────

def _config_id(config):
    """Deterministic hash of hyperparameters."""
    raw = json.dumps({k: v for k, v in config.items() if k not in ("config_id", "model")},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def sample_configs(model_names, n_total, target_params=None, seed=42):
    """Generate n_total random hyperparameter configurations across model_names.

    If target_params is a list of param budgets, configurations are biased
    toward those budgets (isocline sampling).  Returns list of config dicts.
    """
    rng = random.Random(seed)
    configs = []
    n_per_model = max(1, n_total // len(model_names))

    for model_name in model_names:
        space = SEARCH_SPACES[model_name]
        collected = []

        if target_params:
            n_per_target = max(1, n_per_model // len(target_params))
            for target in target_params:
                for _ in range(n_per_target):
                    cfg = _sample_near_target(model_name, space, target, rng)
                    if cfg:
                        collected.append(cfg)

        # Fill remainder with random samples
        while len(collected) < n_per_model:
            cfg = _sample_random(model_name, space, rng)
            collected.append(cfg)

        configs.extend(collected)

    # Assign IDs
    for cfg in configs:
        cfg["config_id"] = _config_id(cfg)

    return configs[:n_total]


def _sample_random(model_name, space, rng):
    cfg = {"model": model_name}
    for key, spec in space.items():
        if spec["type"] == "int":
            cfg[key] = rng.randint(*spec["range"])
        elif spec["type"] == "logint":
            lo, hi = spec["range"]
            log_v = rng.uniform(math.log(lo), math.log(hi))
            v = int(round(math.exp(log_v)))
            cfg[key] = max(lo, min(hi, v))
        elif spec["type"] == "choice":
            cfg[key] = rng.choice(spec["values"])
    _derived_fields(cfg, model_name)
    return cfg


def _sample_near_target(model_name, space, target, rng):
    """Sample a config with param count near target (isocline sampling)."""
    hidden_dims = list(_BLOCK_PARAM_EST.get(model_name, {}).keys())
    if not hidden_dims:
        return _sample_random(model_name, space, rng)

    hid = rng.choice(hidden_dims)
    per_block = _BLOCK_PARAM_EST[model_name].get(hid, 100000)
    base = _BASE_PARAM_EST.get(model_name, 0)

    d = max(1, int(round((target - base) / per_block)))
    space_specs = {k: v for k, v in space.items()}

    if "depth" in space_specs:
        d = max(space_specs["depth"]["range"][0],
                min(space_specs["depth"]["range"][1], d))

    cfg = {"model": model_name, "hidden_dim": hid}
    for key, spec in space_specs.items():
        if key == "hidden_dim":
            continue
        elif key == "depth":
            cfg[key] = d
        elif spec["type"] == "int":
            cfg[key] = rng.randint(*spec["range"])
        elif spec["type"] == "choice":
            cfg[key] = rng.choice(spec["values"])

    _derived_fields(cfg, model_name)
    return cfg


def _derived_fields(cfg, model_name):
    if model_name in ("lewm", "lewm_deltanet"):
        cfg["mlp_dim"] = cfg["hidden_dim"] * cfg.pop("mlp_mult")


# ─────────────────────────────────────────────────────────────────────
#  Model construction
# ─────────────────────────────────────────────────────────────────────

def build_predictor(model_name, config, num_frames=64, embed_dim=192):
    """Build a predictor from a hyperparameter config."""
    hdim = config["hidden_dim"]

    if model_name == "lewm_mamba":
        model = MambaPredictor(
            num_frames=num_frames, depth=config["depth"],
            input_dim=embed_dim, hidden_dim=hdim, output_dim=embed_dim,
            d_state=config["d_state"], d_conv=config["d_conv"],
            expand=config["expand"], dropout=0.1, emb_dropout=0.0,
        )
        # MambaPredictor's cond_proj assumes hidden_dim == input_dim.
        # When sweeping hidden_dim, patch to map from embed_dim → hidden_dim.
        if embed_dim != hdim:
            model.cond_proj = nn.Linear(embed_dim, hdim)
        return model
    else:
        bc = ConditionalBlock if model_name == "lewm" else DeltaNetConditionalBlock
        return ARPredictor(
            num_frames=num_frames, depth=config["depth"],
            heads=config["heads"], dim_head=config["dim_head"],
            mlp_dim=config["mlp_dim"],
            input_dim=embed_dim, hidden_dim=hdim, output_dim=embed_dim,
            dropout=0.1, emb_dropout=0.0, block_class=bc,
        )


# ─────────────────────────────────────────────────────────────────────
#  L0 — Parameter count
# ─────────────────────────────────────────────────────────────────────

def eval_L0(config, embed_dim=192):
    """Return param-count metrics for a config (no GPU needed)."""
    model = build_predictor(config["model"], config, num_frames=16, embed_dim=embed_dim)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": total, "trainable_params": trainable}


# ─────────────────────────────────────────────────────────────────────
#  L1 — Speed benchmark
# ─────────────────────────────────────────────────────────────────────

def eval_speed(config, lengths=(16, 32, 64), batch_size=32, n_warmup=5, n_trials=50,
               embed_dim=192, device=DEVICE):
    """Benchmark forward + rollout speed. Returns dict of metrics."""
    hsize = max(lengths)
    model = build_predictor(config["model"], config, num_frames=hsize, embed_dim=embed_dim)
    model.to(device)
    model.eval()

    metrics = {}
    ctx_size = 3

    params = sum(p.numel() for p in model.parameters())
    metrics["params"] = params
    metrics["trainable_params"] = sum(p.numel() for p in model.parameters()
                                      if p.requires_grad)

    for T in lengths:
        x = torch.randn(batch_size, T, embed_dim, device=device)
        c = torch.randn(batch_size, T, embed_dim, device=device)

        # Warmup
        torch.cuda.synchronize(device) if device != "cpu" else None
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x, c)
        torch.cuda.synchronize(device) if device != "cpu" else None
        if device != "cpu":
            torch.cuda.empty_cache()

        # Timed trials
        torch.cuda.synchronize(device) if device != "cpu" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(x, c)
        torch.cuda.synchronize(device) if device != "cpu" else None
        t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000
        ms = total_ms / n_trials
        tok_s = batch_size * T / (total_ms / 1000 / n_trials)

        metrics[f"forward_ms_{T}"] = round(ms, 2)
        metrics[f"forward_tok_s_{T}"] = round(tok_s, 0)

        # Rollout step speed
        x_ctx = torch.randn(batch_size, ctx_size, embed_dim, device=device)
        c_ctx = torch.randn(batch_size, ctx_size, embed_dim, device=device)

        has_step = hasattr(model, "step") and config["model"] == "lewm_mamba"

        if has_step:
            with torch.no_grad():
                state = None
                for _ in range(ctx_size):
                    _, state = model.step(x_ctx[:, -1:], c_ctx[:, -1:], state)
                for _ in range(n_warmup):
                    _, state = model.step(x_ctx[:, -1:], c_ctx[:, -1:], state)

            torch.cuda.synchronize(device) if device != "cpu" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_trials):
                    _, state = model.step(x_ctx[:, -1:], c_ctx[:, -1:], state)
            torch.cuda.synchronize(device) if device != "cpu" else None
            t1 = time.perf_counter()
        else:
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = model(x_ctx, c_ctx)

            torch.cuda.synchronize(device) if device != "cpu" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_trials):
                    _ = model(x_ctx, c_ctx)
            torch.cuda.synchronize(device) if device != "cpu" else None
            t1 = time.perf_counter()

        rollout_ms = (t1 - t0) * 1000 / n_trials
        metrics[f"rollout_step_ms_{T}"] = round(rollout_ms, 3)

        # Peak memory
        if device != "cpu":
            metrics[f"peak_memory_mb_{T}"] = round(
                torch.cuda.max_memory_allocated(device) / 1e6, 0)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
        else:
            metrics[f"peak_memory_mb_{T}"] = 0

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────
#  L2 — Train + Quality
# ─────────────────────────────────────────────────────────────────────

_DATASET_CACHE = {}


def _get_dataset(task, modality, num_episodes, seed, eval_lens):
    """Generate or load cached dataset."""
    cache_key = (task, modality, num_episodes, seed, tuple(eval_lens))
    if cache_key not in _DATASET_CACHE:
        data = generate_dataset(
            task=task, modality=modality,
            num_episodes=num_episodes, ep_len=max(eval_lens),
            seed=seed, obs_dim=16, action_dim=4, image_size=64,
        )
        embed_dim = 192
        obs_proj = LinearEncoder(16, embed_dim)
        with torch.no_grad():
            emb_proj = obs_proj(data["emb"].reshape(-1, 16))
            emb_proj = emb_proj.reshape(data["emb"].shape[0], max(eval_lens), embed_dim)
        _DATASET_CACHE[cache_key] = dict(emb=emb_proj, act=data["act"])
    return _DATASET_CACHE[cache_key]


def eval_quality(config, task="linear_ar", modality="state", episodes=250,
                 eval_lens=(16, 32, 64), train_steps=500, batch_size=64,
                 loss_mode="lewm", seed=0, embed_dim=192, device=DEVICE):
    """Train predictor on synthetic data and evaluate rollout quality."""
    data = _get_dataset(task, modality, episodes, seed, eval_lens)

    # Build fresh model + action encoder
    predictor = build_predictor(config["model"], config, num_frames=max(eval_lens),
                                embed_dim=embed_dim)
    act_encoder = make_action_encoder(input_dim=4, emb_dim=embed_dim)

    # Train
    t0 = time.time()
    trained_pred, trained_act = train_model(
        predictor, act_encoder, data,
        loss_mode=loss_mode, steps=train_steps,
        batch_size=batch_size, device=device,
    )
    train_time = time.time() - t0

    metrics = {"train_time_s": round(train_time, 1)}

    # Evaluate at each eval_len
    for eval_len in eval_lens:
        eval_data = dict(
            emb=data["emb"][:, :eval_len],
            act=data["act"][:, :eval_len],
        )
        result = evaluate_rollout(
            trained_pred, trained_act, eval_data,
            eval_len=eval_len, device=device,
        )
        metrics[f"mean_mse_{eval_len}"] = round(result["mean_mse"], 6)
        metrics[f"first_div_step_{eval_len}"] = result["first_div_step"]

    return metrics


# ─────────────────────────────────────────────────────────────────────
#  CSV I/O
# ─────────────────────────────────────────────────────────────────────

L0_FIELDS = ["config_id", "model", "depth", "hidden_dim"]
L1_FIELDS_PREFIX = ["forward_ms", "forward_tok_s", "rollout_step_ms", "peak_memory_mb"]
L2_FIELDS_PREFIX = ["mean_mse", "first_div_step"]


def _get_L0_fieldnames(configs):
    """All scalar hyperparameter keys + params/trainable_params."""
    keys = set()
    for cfg in configs:
        keys.update(k for k in cfg if k not in ("config_id", "model", "mlp_mult"))
    return ["config_id", "model"] + sorted(keys) + ["params", "trainable_params"]


def _get_L1_fieldnames(lengths):
    fields = ["config_id"]
    for prefix in L1_FIELDS_PREFIX:
        for T in lengths:
            fields.append(f"{prefix}_{T}")
    fields.extend(["params", "trainable_params"])
    return fields


def _get_L2_fieldnames(eval_lens, l1_rows=None):
    fields = ["config_id", "seed", "train_time_s"]
    for prefix in L2_FIELDS_PREFIX:
        for L in eval_lens:
            fields.append(f"{prefix}_{L}")
    # Include L1 columns if any data available (for merged Pareto analysis)
    if l1_rows:
        for k in l1_rows[0]:
            if k not in fields and k != "config_id":
                fields.append(k)
    return fields


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────
#  Pareto frontier
# ─────────────────────────────────────────────────────────────────────

def pareto_frontier(rows, maximize=(), minimize=()):
    """Return Pareto-optimal rows (non-dominated sorting, first front).

    maximize:  list of column names where higher is better
    minimize:  list of column names where lower is better
    """
    dominated = [False] * len(rows)

    for i in range(len(rows)):
        if dominated[i]:
            continue
        for j in range(len(rows)):
            if i == j or dominated[j]:
                continue

            # Check if i dominates j
            better = False
            worse_or_equal = True
            for col in maximize:
                vi = float(rows[i][col])
                vj = float(rows[j][col])
                if vi > vj:
                    better = True
                elif vi < vj:
                    worse_or_equal = False
            for col in minimize:
                vi = float(rows[i][col])
                vj = float(rows[j][col])
                if vi < vj:
                    better = True
                elif vi > vj:
                    worse_or_equal = False

            if better and worse_or_equal:
                dominated[j] = True

    return [rows[i] for i in range(len(rows)) if not dominated[i]]


def analyze_pareto(csv_path, out_path,
                   maximize=("forward_tok_s_64",),
                   minimize=("mean_mse_64", "params")):
    """Read L2 CSV, compute Pareto frontier, write summary."""
    rows = read_csv(csv_path)
    if not rows:
        print("No data to analyze.")
        return

    # Validate objective columns exist
    all_cols = set(rows[0].keys())
    for col in maximize + minimize:
        if col not in all_cols:
            print(f"  Column '{col}' not in CSV. Available: {sorted(all_cols)}")
            return

    print(f"\nPareto analysis: {len(rows)} configs × seeds")
    print(f"  Objectives: max({', '.join(maximize)})  min({', '.join(minimize)})")

    # Group by config_id, average across seeds
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["config_id"]].append(r)

    avg_rows = []
    for cid, variants in grouped.items():
        avg = dict(variants[0])
        for col in maximize + minimize:
            vals = [float(v[col]) for v in variants]
            avg[col] = sum(vals) / len(vals)
        for col in minimize:
            if col in ("params",):
                avg[col] = int(variants[0][col])
        avg_rows.append(avg)

    frontier = pareto_frontier(avg_rows, maximize=maximize, minimize=minimize)
    frontier.sort(key=lambda r: -float(r[maximize[0]]))

    # Print
    header_cols = ["config_id", "model", "hidden_dim", "depth", "params"]
    for c in maximize:
        header_cols.append(c)
    for c in minimize:
        if c not in header_cols:
            header_cols.append(c)

    print(f"\nPareto frontier ({len(frontier)} configs):")
    print("  " + "  ".join(f"{c:>14}" for c in header_cols))
    print("  " + "-" * (14 * len(header_cols)))
    for r in frontier:
        vals = []
        for c in header_cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:>14.4f}")
            else:
                vals.append(f"{str(v):>14}")
        print("  " + "  ".join(vals))

    # Write CSV
    write_csv(out_path, frontier, header_cols)
    print(f"\nSummary: {len(frontier)} Pareto-optimal configs out of {len(avg_rows)} unique configs.")

    return frontier


# ─────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────

def run_sweep(args):
    """L0 → L1 → L2 cascade."""
    base_path = args.out.replace(".csv", "")
    model_names = args.model.split(",")
    lengths = [16, 32, 64]
    eval_lens = [int(x) for x in args.eval_lens.split(",")]

    # ── Step 0: Generate configs + L0 ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Generating configs: {model_names}  n={args.n_samples}")
    print(f"{'='*60}")

    target_params = None
    if args.target_params:
        target_params = [float(x) for x in args.target_params.split(",")]

    configs = sample_configs(model_names, args.n_samples, target_params=target_params)

    print(f"  Sampling → {len(configs)} configs")

    # L0: param count
    print(f"  Running L0 (param count)...")
    for cfg in configs:
        if "params" not in cfg:
            l0 = eval_L0(cfg)
            cfg.update(l0)

    l0_path = f"{base_path}_L0.csv"
    write_csv(l0_path, configs, _get_L0_fieldnames(configs))

    if args.fidelity == "L0":
        return

    # ── Step 1: L1 speed benchmark ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  L1: Speed benchmark  ({len(configs)} configs)")
    print(f"{'='*60}")

    l1_rows = []
    for i, cfg in enumerate(configs):
        if args.max_params and cfg["params"] > args.max_params:
            continue
        t0 = time.time()
        try:
            spd = eval_speed(cfg, lengths=lengths, batch_size=args.batch_size,
                             n_trials=args.speed_trials)
            elapsed = time.time() - t0
            row = {"config_id": cfg["config_id"]}
            row.update(spd)
            l1_rows.append(row)
            print(f"  [{i+1}/{len(configs)}] {cfg['config_id']}  "
                  f"params={cfg['params']:,}  "
                  f"tok_s_64={spd.get('forward_tok_s_64', 0):.0f}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] {cfg['config_id']} ERROR: {e}")

    l1_path = f"{base_path}_L1.csv"
    write_csv(l1_path, l1_rows, _get_L1_fieldnames(lengths))

    if args.fidelity == "L1":
        return

    # ── Step 2: L2 train + quality ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  L2: Train + quality  ({len(l1_rows)} configs × {args.seeds} seeds)")
    print(f"{'='*60}")
    print(f"  Dataset: {args.task}, {args.episodes} episodes, "
          f"{args.train_steps} steps")

    l2_rows = []
    for i, cfg_row in enumerate(l1_rows):
        cid = cfg_row["config_id"]
        cfg = next(c for c in configs if c["config_id"] == cid)

        for seed in range(args.seeds):
            t0 = time.time()
            try:
                qual = eval_quality(
                    cfg, task=args.task, modality=args.modality,
                    episodes=args.episodes, eval_lens=eval_lens,
                    train_steps=args.train_steps, loss_mode=args.loss,
                    seed=seed, device=DEVICE,
                )
                elapsed = time.time() - t0
                row = {"config_id": cid, "seed": seed}
                # Merge L1 speed data so Pareto analysis sees both speed and quality
                for k, v in cfg_row.items():
                    if k not in row and k != "config_id":
                        row[k] = v
                row.update(qual)
                l2_rows.append(row)
                print(f"  [{i+1}/{len(l1_rows)} seed={seed}] {cid}  "
                      f"mse_64={qual.get('mean_mse_64', 0):.6f}  "
                      f"({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [{i+1}/{len(l1_rows)} seed={seed}] {cid} ERROR: {e}")

    l2_path = f"{base_path}_L2.csv"
    write_csv(l2_path, l2_rows, _get_L2_fieldnames(eval_lens, l1_rows))

    # ── Step 3: Pareto analysis ────────────────────────────────────────
    pareto_path = f"{base_path}_pareto.csv"
    analyze_pareto(l2_path, pareto_path)

    print(f"\nResults:")
    print(f"  L0: {l0_path}")
    print(f"  L1: {l1_path}")
    print(f"  L2: {l2_path}")
    print(f"  Pareto: {pareto_path}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-fidelity hyperparameter sweep for LeWM predictors",
    )
    parser.add_argument("--model", default="lewm",
                        help="Comma-separated: lewm,lewm_deltanet,lewm_mamba")
    parser.add_argument("--task", default="linear_ar",
                        help="Synthetic data task")
    parser.add_argument("--modality", default="state")
    parser.add_argument("--loss", default="lewm")
    parser.add_argument("--eval-lens", default="16,32,64",
                        help="Eval lengths (no 128 — too slow)")
    parser.add_argument("--episodes", type=int, default=250,
                        help="Dataset episodes (standard test set)")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Data seeds per config at L2")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of hyperparameter configs")
    parser.add_argument("--target-params", default=None,
                        help="Isocline targets, e.g. '1e6,3e6,1e7,3e7'")
    parser.add_argument("--fidelity", default="L2",
                        choices=["L0", "L1", "L2"],
                        help="Highest fidelity to run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--speed-trials", type=int, default=50)
    parser.add_argument("--max-params", type=float, default=None,
                        help="Skip configs with params > this at L1")
    parser.add_argument("--out", default="results/sweep",
                        help="Output path prefix (or .csv for analyze)")
    parser.add_argument("--analyze", default=None, metavar="CSV",
                        help="Analyze existing L2 CSV for Pareto frontier")

    args = parser.parse_args()

    if args.analyze:
        base = args.out.replace(".csv", "")
        out = f"{base}_pareto.csv" if not args.out.endswith(".csv") else args.out
        analyze_pareto(args.analyze, out)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
