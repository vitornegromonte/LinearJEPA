#!/usr/bin/env python3
"""Step-generalization benchmark for LeWM predictors.

Compares baseline Transformer, DeltaNet, and Mamba on synthetic tasks
across train/eval length mismatches, long-range memory, and rollout fidelity.

Usage:
  # Minimal smoke test
  python bench_generalization.py --task linear_ar --modality state \
      --train-len 16 --eval-lens 16,32 --episodes 50 --steps 200 --seeds 1

  # Full sweep
  python bench_generalization.py \
      --task linear_ar,nback,delayed_copy,slowfast,chaotic \
      --modality state \
      --train-len 16 --eval-lens 16,32,64,128 \
      --models lewm,lewm_deltanet,lewm_mamba \
      --loss lewm,jepa \
      --episodes 200 --steps 2000 --seeds 3 \
      --out results/benchmark.csv
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from einops import rearrange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Project modules (assumes running from repo root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module import ARPredictor, MambaPredictor, Embedder, MLP, SIGReg
from module import ConditionalBlock, DeltaNetConditionalBlock, Block

# Force pure-PyTorch fallback for DeltaNet on CPU
# fla's CUDA kernels crash without a GPU, so we patch the __init__.
if DEVICE == "cpu":
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


# ─── Data generation ───────────────────────────────────────────────────

def generate_dataset(task, modality, num_episodes, ep_len, seed=0,
                     obs_dim=16, action_dim=4, image_size=64):
    """Generate synthetic dataset in-memory as a single flat torch tensor.

    Returns dict with keys:
      - emb: (num_episodes, ep_len, obs_dim) float32 — the "embedding" / state
      - act: (num_episodes, ep_len, action_dim) float32 — actions
      - pixels: optional (num_episodes, ep_len, H, W, 3) uint8
    """
    from synth_data.tasks import TASKS, linear_ar, nback, delayed_copy, slowfast, chaotic
    from synth_data.render import render_state_to_image

    task_fn = {
        "linear_ar": linear_ar,
        "nback": nback,
        "delayed_copy": delayed_copy,
        "slowfast": slowfast,
        "chaotic": chaotic,
    }[task]

    params = dict(TASKS[task][1])
    params.update(obs_dim=obs_dim, action_dim=action_dim)

    episodes = []
    for ep_idx in range(num_episodes):
        ep_rng = np.random.RandomState(seed * 1000 + ep_idx)
        ep = task_fn(ep_rng, ep_len, action_dim=action_dim, obs_dim=obs_dim,
                     **{k: v for k, v in params.items()
                        if k not in ("obs_dim", "action_dim")})
        if modality in ("image", "both"):
            pixels = np.empty((ep_len, image_size, image_size, 3), dtype=np.uint8)
            prev = None
            for t in range(ep_len):
                pixels[t] = render_state_to_image(
                    ep["state"][t], ep["action"][t], task,
                    image_size, image_size, prev,
                )
                prev = pixels[t]
            ep["pixels"] = pixels
        episodes.append(ep)

    # Stack into flat arrays
    state = np.stack([e["state"] for e in episodes]).astype(np.float32)       # (N, T, obs_dim)
    action = np.stack([e["action"] for e in episodes]).astype(np.float32)     # (N, T, act_dim)
    result = dict(emb=torch.from_numpy(state), act=torch.from_numpy(action))

    if modality in ("image", "both"):
        pixels = np.stack([e["pixels"] for e in episodes])                    # (N, T, H, W, 3)
        result["pixels"] = torch.from_numpy(pixels)

    return result


# ─── Model construction ────────────────────────────────────────────────

def make_predictor(model_name, input_dim, hidden_dim=192, output_dim=192,
                   depth=6, heads=16, dim_head=64, mlp_dim=2048,
                   history_size=3):
    """Create a predictor module for the given architecture."""
    if model_name == "lewm":
        block_class = ConditionalBlock
    elif model_name == "lewm_deltanet":
        block_class = DeltaNetConditionalBlock
    elif model_name == "lewm_mamba":
        return MambaPredictor(
            num_frames=history_size, depth=depth,
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            d_state=16, d_conv=4, expand=2, dropout=0.1, emb_dropout=0.0,
        )  # MambaPredictor uses CondBlock internally
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ARPredictor(
        num_frames=history_size, depth=depth, heads=heads, mlp_dim=mlp_dim,
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
        dim_head=dim_head, dropout=0.1, emb_dropout=0.0,
        block_class=block_class,
    )


def make_action_encoder(input_dim, emb_dim=192):
    return Embedder(input_dim=input_dim, smoothed_dim=emb_dim, emb_dim=emb_dim)


class LinearEncoder(nn.Module):
    """Simple encoder: projects obs_dim -> embed_dim with optional ViT fallback for images."""
    def __init__(self, obs_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(obs_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


# ─── Training ──────────────────────────────────────────────────────────

class LeWMModel(nn.Module):
    """Minimal JEPA-like model wrapping a predictor and action encoder.

    For the benchmark we skip the full JEPA wrapper and use a simpler
    interface: encode observations, then predict next embeddings.
    """

    def __init__(self, predictor, action_encoder, encoder=None, sigreg=None):
        super().__init__()
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.encoder = encoder or nn.Identity()  # maps obs -> embed
        self.sigreg = sigreg or SIGReg(knots=17, num_proj=1024)
        self.pred_proj = nn.Identity()

    def forward(self, emb, act):
        """emb: (B, T, D), act: (B, T, act_dim).
        Returns (pred_loss, sigreg_loss, total_loss, pred_emb, tgt_emb).
        """
        B, T, D = emb.shape
        history_size = 3
        n_preds = 1

        ctx_emb = emb[:, :history_size]
        ctx_act = self.action_encoder(act[:, :history_size])
        tgt_emb = emb[:, n_preds:]

        pred_emb = self.predictor(ctx_emb, ctx_act)
        # pred_emb: (B, history_size, D) — one prediction per position

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()

        sigreg_loss = self.sigreg(emb.transpose(0, 1))

        loss = pred_loss + 0.09 * sigreg_loss

        return pred_loss, sigreg_loss, loss, pred_emb, tgt_emb


def loss_jepa_only(pred_emb, tgt_emb):
    return (pred_emb - tgt_emb).pow(2).mean()


def train_model(predictor, act_encoder, data, loss_mode, steps=1000,
                lr=5e-5, batch_size=64, device=DEVICE):
    """Train a predictor with the given loss mode.

    data: dict with 'emb' (N, T_full, D) and 'act' (N, T_full, A)
    loss_mode: 'lewm' or 'jepa'
    """
    predictor = predictor.to(device)
    act_encoder = act_encoder.to(device)

    params = list(predictor.parameters()) + list(act_encoder.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-3)
    history_size = 3
    num_steps = history_size + 1  # 4 = 3 context + 1 target
    window_stride = 1

    sigreg = SIGReg(knots=17, num_proj=1024).to(device) if loss_mode == "lewm" else None
    sigreg_weight = 0.09

    emb_all = data["emb"].to(device)  # (N, T_full, D)
    act_all = data["act"].to(device)  # (N, T_full, A)

    N, T_full, D = emb_all.shape

    # Pre-compute all sliding windows (list of (ep_idx, start))
    windows = []
    for ep_idx in range(N):
        for start in range(0, T_full - num_steps + 1, window_stride):
            windows.append((ep_idx, start))

    predictor.train()
    act_encoder.train()

    for step in range(steps):
        # Sample batch of windows
        idx = np.random.randint(0, len(windows), size=batch_size).tolist()
        sel = [windows[i] for i in idx]
        batch_emb = torch.stack([emb_all[ep, s:s + num_steps] for ep, s in sel])
        batch_act = torch.stack([act_all[ep, s:s + num_steps] for ep, s in sel])
        # batch_emb: (B, num_steps, D), batch_act: (B, num_steps, A)

        B = batch_emb.shape[0]
        ctx_emb = batch_emb[:, :history_size]
        ctx_act = act_encoder(batch_act[:, :history_size])
        tgt_emb = batch_emb[:, 1:]  # (B, history_size, D)

        pred_emb = predictor(ctx_emb, ctx_act)  # (B, history_size, D)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()

        s_loss = 0.0
        if loss_mode == "lewm":
            with torch.no_grad():
                s_loss = sigreg(batch_emb.transpose(0, 1))
            loss = pred_loss + sigreg_weight * s_loss
        else:
            loss = pred_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    return predictor, act_encoder


# ─── Evaluation ────────────────────────────────────────────────────────

def evaluate_rollout(predictor, act_encoder, data, eval_len, history_size=3,
                     device=DEVICE):
    """Evaluate autoregressive rollout quality.

    Uses the same stateless truncation strategy as jepa._rollout_truncation.

    Returns dict with:
      - per_step_mse: list of len eval_len - history_size
      - first_div_step: int (first step where MSE > 2x first-step MSE, or -1)
      - mean_mse: float
    """
    predictor.eval()
    act_encoder.eval()

    emb = data["emb"].to(device)  # (N, T_full, D)
    act = data["act"].to(device)  # (N, T_full, A)
    N, _T_full, D = emb.shape

    all_mse = []

    with torch.no_grad():
        for ep_idx in range(N):
            # Take first eval_len from each episode
            ep_emb = emb[ep_idx:ep_idx + 1, :eval_len]  # (1, eval_len, D)
            ep_act = act[ep_idx:ep_idx + 1, :eval_len]

            # Context = first history_size steps (ground truth)
            ctx_emb = ep_emb[:, :history_size]  # (1, HS, D)
            ctx_act = ep_act[:, :history_size]

            # We'll extend ctx_emb and ctx_act step by step
            # Store the growing rollout
            rollout = ctx_emb.clone()

            for step in range(eval_len - history_size):
                # Take the last history_size steps as context
                ctx = rollout[:, -history_size:]
                c_act = act_encoder(ep_act[:, step:step + history_size])
                pred = predictor(ctx, c_act)  # (1, HS, D)
                pred_next = pred[:, -1:]  # (1, 1, D)
                rollout = torch.cat([rollout, pred_next], dim=1)

                # MSE against ground truth at this step
                gt = ep_emb[:, history_size + step:history_size + step + 1]
                mse = (pred_next - gt).pow(2).mean().item()
                all_mse.append(mse)

    # Aggregate across episodes
    per_step = np.array(all_mse).reshape(N, eval_len - history_size).mean(0)

    first_step_mse = per_step[0]
    threshold = 2.0 * first_step_mse
    div_idx = np.argmax(per_step > threshold) if np.any(per_step > threshold) else -1

    return dict(
        per_step_mse=per_step.tolist(),
        first_div_step=int(div_idx) if div_idx >= 0 else eval_len - history_size,
        mean_mse=float(per_step.mean()),
    )


# ─── Main sweep ────────────────────────────────────────────────────────

def run_benchmark(args):
    """Run the full benchmark sweep and write results to CSV."""
    tasks = args.task.split(",")
    models = args.models.split(",")
    losses = args.loss.split(",")
    eval_lens = [int(x) for x in args.eval_lens.split(",")]
    seeds = args.seeds

    fieldnames = [
        "task", "modality", "model", "loss", "seed",
        "eval_len", "first_div_step", "mean_mse", "final_mse",
        "train_time_s",
    ]
    results = []

    for task in tasks:
        for seed in range(seeds):
            print(f"\n{'='*60}")
            print(f"  Generating data: {task}, seed={seed}")
            print(f"{'='*60}")

            t0 = time.time()
            data = generate_dataset(
                task=task, modality=args.modality,
                num_episodes=args.episodes, ep_len=max(eval_lens),
                seed=seed, obs_dim=16, action_dim=4, image_size=64,
            )
            gen_time = time.time() - t0
            print(f"  Data generated ({gen_time:.1f}s): "
                  f"{data['emb'].shape}, act={data['act'].shape}")

            embed_dim = 192

            for model_name in models:
                for loss_mode in losses:
                    print(f"\n  --- {model_name} / {loss_mode} / seed={seed} ---")

                    t0 = time.time()

                    predictor = make_predictor(
                        model_name, input_dim=embed_dim, hidden_dim=embed_dim,
                        output_dim=embed_dim, history_size=3,
                    )
                    act_encoder = make_action_encoder(input_dim=4, emb_dim=embed_dim)

                    # Project obs_dim -> embed_dim (detached — preprocessing only)
                    obs_proj = LinearEncoder(16, embed_dim)
                    with torch.no_grad():
                        emb_proj = obs_proj(data["emb"].reshape(-1, 16))
                        emb_proj = emb_proj.reshape(data["emb"].shape[0], max(eval_lens), embed_dim)
                    data_proj = dict(emb=emb_proj, act=data["act"])

                    trained_predictor, trained_act = train_model(
                        predictor, act_encoder, data_proj,
                        loss_mode=loss_mode, steps=args.steps,
                        device=DEVICE,
                    )
                    train_time = time.time() - t0
                    print(f"  Trained ({train_time:.1f}s)")

                    # Evaluate at each eval_len
                    for eval_len in eval_lens:
                        # Subset data to eval_len
                        eval_data = dict(
                            emb=data_proj["emb"][:, :eval_len],
                            act=data_proj["act"][:, :eval_len],
                        )

                        metrics = evaluate_rollout(
                            trained_predictor, trained_act, eval_data,
                            eval_len=eval_len,
                        )

                        print(f"    eval_len={eval_len:4d}  "
                              f"first_div={metrics['first_div_step']:4d}  "
                              f"mean_mse={metrics['mean_mse']:.6f}")

                        results.append(dict(
                            task=task,
                            modality=args.modality,
                            model=model_name,
                            loss=loss_mode,
                            seed=seed,
                            eval_len=eval_len,
                            first_div_step=metrics["first_div_step"],
                            mean_mse=metrics["mean_mse"],
                            final_mse=metrics["per_step_mse"][-1] if metrics["per_step_mse"] else -1,
                            train_time_s=round(train_time, 1),
                        ))

    # Write CSV
    out_path = args.out
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step-generalization benchmark for LeWM predictors"
    )
    parser.add_argument("--task", default="linear_ar",
                        help="Comma-separated tasks: linear_ar,nback,delayed_copy,slowfast,chaotic")
    parser.add_argument("--modality", default="state", choices=["state", "image", "both"])
    parser.add_argument("--train-len", type=int, default=16,
                        help="Training sequence length (actual num_steps=4 from sliding window)")
    parser.add_argument("--eval-lens", default="16,32,64,128",
                        help="Comma-separated eval lengths")
    parser.add_argument("--models", default="lewm,lewm_deltanet,lewm_mamba",
                        help="Comma-separated model names")
    parser.add_argument("--loss", default="lewm,jepa",
                        help="Comma-separated loss modes: lewm,jepa")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps per model")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/benchmark.csv",
                        help="Output CSV path")

    args = parser.parse_args()
    run_benchmark(args)
