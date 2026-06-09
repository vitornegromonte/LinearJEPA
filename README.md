
# LinearJEPA
### Model-Agnostic Hyperparameter Sweep for Autoregressive Predictors

This repository implements a **model-agnostic hyperparameter sweep** comparing three autoregressive predictor architectures on synthetic step-generalization tasks:

| Model | Attention | Complexity |
|-------|-----------|------------|
| **LeWM** (Transformer) | Softmax attention (FlashAttention) | O(T²) |
| **DeltaNet** | Linear attention via delta rule | O(T) |
| **Mamba** | State-space model (SSM) | O(T) |

It provides a complete multi-fidelity evaluation pipeline — from parameter counting (L0) through speed benchmarking (L1) to full training and rollout-quality measurement (L2) — with automated Pareto frontier analysis to identify optimal configurations across **speed × quality × parameter count**.

## Synthetic Benchmarks

Generate synthetic data to stress-test predictor architectures (Transformer, DeltaNet, Mamba) on controlled dynamics:

```bash
# Generate a synthetic linear AR dataset (state modality, 200 episodes)
python synth_data.py --task linear_ar --modality state --num-episodes 200 --ep-len 128

# Train on it
python train.py data=synth model=lewm_deltanet data.name=synth_linear_ar.h5 wandb.enabled=False

# Customize hyperparameters
python synth_data.py --task nback --modality image --image-size 64 --n-back 5 --num-episodes 500
```

Available tasks: `linear_ar`, `nback`, `delayed_copy`, `slowfast`, `chaotic`.

### Step-Generalization Benchmark

Compare how the three predictors handle training/eval length mismatches, long-range memory, and rollout quality:

```bash
# Full sweep: 5 tasks × 2 loss modes × 3 seeds
python bench_generalization.py --out results/benchmark.csv

# Include speed benchmarking (forward latency, rollout step speed, memory)
python bench_generalization.py --bench-speed --out results/benchmark.csv

# Speed-only vs accuracy: control sequence lengths and batch size
python bench_generalization.py --bench-speed \
    --speed-lens 16,32,64,128,256 \
    --batch-size 128 --speed-trials 100 \
    --task linear_ar --models lewm,lewm_deltanet,lewm_mamba \
    --episodes 50 --steps 200 --seeds 1

# Fair speed comparison: scale depths so all models have ~same param count
python bench_generalization.py --bench-speed --match-params \
    --speed-lens 16,32,64,128,256 \
    --task linear_ar --episodes 50 --steps 200 --seeds 1

# Custom run
python bench_generalization.py \
    --task nback,delayed_copy \
    --modality state \
    --train-len 16 --eval-lens 16,32,64,128 \
    --models lewm,lewm_deltanet,lewm_mamba \
    --loss lewm,jepa \
    --episodes 200 --steps 2000 --seeds 3
```

Output is a CSV with accuracy columns: `task, model, loss, seed, eval_len, first_div_step, mean_mse`.
With `--bench-speed`, an additional speed table is appended with columns:
`model, seq_len, forward_ms, forward_tok_s, ms_per_tok, rollout_step_ms, depth, params, peak_memory_mb`.
Mamba's `step()`-based rollout is flagged `★` in the terminal output.
With `--match-params`, model depths are scaled so all have similar param counts
(fairer compute comparison; e.g. Mamba gets ~51 layers vs Transformer's 6).

### Multi-Fidelity Hyperparameter Sweep

`bench_hparam_sweep.py` performs a model-agnostic Pareto sweep across speed, rollout quality, and parameter count. It implements a three-fidelity cascade:

| Fidelity | What's measured | Cost | Decision |
|----------|----------------|------|----------|
| **L0** | Parameter count | ~0s | Initial filter |
| **L1** | Forward latency @ T=16,32,64; rollout step; memory | ~1–2s/config | Keep top performers |
| **L2** | Train + rollout MSE @ 16,32,64; first divergence step | ~10–30s/config × seeds | Final metric |

Configs are sampled via **random search** with optional isocline biasing toward specific parameter budgets. The Pareto frontier is computed automatically at the end of the run (non-dominated sorting across speed × quality × params).

```bash
# Full L0+L1+L2 cascade (200 configs, ~30–100 min on GPU)
python bench_hparam_sweep.py --model lewm_mamba --n-samples 200 \
    --task linear_ar --episodes 250 --train-steps 500 --seeds 3 \
    --out results/sweep_mamba

# All three architectures
python bench_hparam_sweep.py --model lewm,lewm_deltanet,lewm_mamba \
    --n-samples 300 --episodes 250 --train-steps 500 \
    --out results/sweep_all

# Speed only (L0 + L1, ~5 min)
python bench_hparam_sweep.py --model lewm --fidelity L1 \
    --n-samples 500 --out results/sweep_speed

# Isocline sampling toward specific param budgets
python bench_hparam_sweep.py --model lewm_mamba \
    --target-params 1e6,3e6,1e7,3e7 \
    --n-samples 200 --out results/sweep_mamba

# Analyze existing L2 results for Pareto frontier
python bench_hparam_sweep.py --analyze results/sweep_mamba_L2.csv \
    --out results/pareto.csv
```

Search spaces (sampled via log-uniform for `hidden_dim`, uniform for `depth`, discrete choices for heads/expand/etc.):

| Parameter | lewm / lewm_deltanet | lewm_mamba |
|-----------|---------------------|------------|
| depth | 2 – 12 | 2 – 48 |
| hidden_dim | 64 – 384 (log) | 64 – 384 (log) |
| heads | {4, 8, 16} | — |
| dim_head | {32, 64} | — |
| mlp_mult | {2, 4, 8} | — |
| d_state | — | 8 – 32 |
| d_conv | — | {4, 8} |
| expand | — | {1, 2, 4} |

Output files: `{out}_L0.csv`, `{out}_L1.csv`, `{out}_L2.csv`, `{out}_pareto.csv`.
The Pareto summary prints the optimal configs and their metrics in a table.
