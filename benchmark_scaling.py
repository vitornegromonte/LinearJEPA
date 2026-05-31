"""Benchmark time complexity scaling for all 3 predictor variants.

Measures forward pass and rollout times across increasing sequence
lengths to validate O(T²) vs O(T) vs O(1) scaling.

Usage:
    python benchmark_scaling.py
"""

import time
import torch
import numpy as np

from module import (
    ARPredictor, ConditionalBlock, DeltaNetConditionalBlock,
    MambaPredictor, Embedder,
)
from jepa import JEPA


def make_encoder(img_size=32):
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(
        size="tiny", patch_size=14, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )


def make_predictor_variants(embed_dim=192, depth=3, history_size=4, max_frames=128):
    common = dict(
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=depth,
    )

    return {
        "baseline": ARPredictor(
            num_frames=max_frames, heads=4, mlp_dim=384,
            dim_head=32, dropout=0.0, emb_dropout=0.0,
            block_class=ConditionalBlock, **common,
        ),
        "deltanet": ARPredictor(
            num_frames=max_frames, heads=4, mlp_dim=384,
            dim_head=32, dropout=0.0, emb_dropout=0.0,
            block_class=DeltaNetConditionalBlock, **common,
        ),
        "mamba": MambaPredictor(
            num_frames=history_size, d_state=8, d_conv=3,
            expand=2, dropout=0.0, emb_dropout=0.0, **common,
        ),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_forward(predictors, seq_lengths, embed_dim=192, batch_size=4, num_runs=20):
    """Measure forward pass time across sequence lengths.

    Each run: predictor(x, c) where x,c have shape (B, T, D).
    """
    print("\n--- Forward pass time (predictor only, no encoder) ---")
    print(f"  batch_size={batch_size}, embed_dim={embed_dim}, num_runs={num_runs}")
    print(f"  {'T':>6} | {'baseline':>10} {'deltanet':>10} {'mamba':>10} | {'best':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")

    results = {k: [] for k in predictors}

    for T in seq_lengths:
        x = torch.randn(batch_size, T, embed_dim)
        c = torch.randn(batch_size, T, embed_dim)
        row = []
        best_time = float("inf")
        best_name = ""

        for name, pred in predictors.items():
            pred.eval()
            with torch.no_grad():
                # warmup
                for _ in range(5):
                    pred(x, c)

                # measure
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()
                for _ in range(num_runs):
                    pred(x, c)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = (time.perf_counter() - start) / num_runs

            results[name].append(elapsed)
            row.append(elapsed)
            if elapsed < best_time:
                best_time = elapsed
                best_name = name

        t_per_step = {name: f"{el:.2e}" for name, el in zip(predictors, row)}
        print(f"  {T:>6} | {t_per_step['baseline']:>10} {t_per_step['deltanet']:>10} "
              f"{t_per_step['mamba']:>10} | {best_name:>10}")

    return results


def benchmark_rollout(predictors, horizons, embed_dim=192, action_dim=2,
                      history_size=4, num_samples=4, num_runs=10):
    """Measure rollout time for increasing horizons.

    Simulates the eval rollout: encode context, then predict n_steps
    autoregressively.  For stateful (Mamba) vs truncation (baseline, DeltaNet).
    """
    print(f"\n--- Rollout time (full JEPA with ViT encoder) ---")
    print(f"  history_size={history_size}, num_samples={num_samples}, "
          f"embed_dim={embed_dim}, num_runs={num_runs}")
    print(f"  {'T':>6} | {'baseline':>10} {'deltanet':>10} {'mamba':>10} | {'best':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")

    results = {k: [] for k in predictors}
    img_size = 32

    for T in horizons:
        if T <= history_size:
            continue
        row = []
        best_time = float("inf")
        best_name = ""

        for name, pred in predictors.items():
            model = JEPA(
                encoder=make_encoder(img_size=img_size),
                predictor=pred,
                action_encoder=Embedder(
                    input_dim=action_dim, smoothed_dim=action_dim, emb_dim=embed_dim,
                ),
                projector=torch.nn.Identity(),
                pred_proj=torch.nn.Identity(),
            )
            model.eval()

            pixels = torch.randn(1, num_samples, history_size, 3, img_size, img_size)
            action_sequence = torch.randn(1, num_samples, T, action_dim)
            info = {"pixels": pixels}

            with torch.no_grad():
                # warmup
                for _ in range(3):
                    model.rollout(info, action_sequence, history_size=history_size)

                # measure
                start = time.perf_counter()
                for _ in range(num_runs):
                    model.rollout(info, action_sequence, history_size=history_size)
                elapsed = (time.perf_counter() - start) / num_runs

            results[name].append(elapsed)
            row.append(elapsed)
            if elapsed < best_time:
                best_time = elapsed
                best_name = name

        t_str = {name: f"{el:.3f}" for name, el in zip(predictors, row)}
        print(f"  {T:>6} | {t_str['baseline']:>10} {t_str['deltanet']:>10} "
              f"{t_str['mamba']:>10} | {best_name:>10}")

    return results


def print_relative_speedup(results, seq_lengths):
    """Print speedup of mamba/deltanet over baseline."""
    print("\n--- Relative speedup (vs baseline) ---")
    baseline = results["baseline"]
    deltanet = results["deltanet"]
    mamba = results["mamba"]

    print(f"  {'T':>6} | {'deltanet/baseline':>16} {'mamba/baseline':>16} | "
          f"{'mamba/deltanet':>16}")
    print(f"  {'-'*6}-+-{'-'*16}-{'-'*16}-+-{'-'*16}")

    for i, T in enumerate(seq_lengths):
        db = baseline[i] / deltanet[i] if deltanet[i] > 0 else float("inf")
        mb = baseline[i] / mamba[i] if mamba[i] > 0 else float("inf")
        md = deltanet[i] / mamba[i] if mamba[i] > 0 else float("inf")
        print(f"  {T:>6} | {db:>15.2f}x {' ':>1} {mb:>15.2f}x {' ':>1} | "
              f"{md:>15.2f}x")


def print_parameter_counts(predictors):
    """Print parameter counts for each variant."""
    print("\n--- Parameter counts ---")
    for name, pred in predictors.items():
        total = count_params(pred)
        # also count in just the sequence model (blocks/transformer)
        if hasattr(pred, "transformer"):
            seq_params = count_params(pred.transformer)
        elif hasattr(pred, "blocks"):
            seq_params = sum(count_params(b) for b in pred.blocks)
        else:
            seq_params = total
        print(f"  {name:>10}: {total:>8,} total  ({seq_params:>8,} in sequence model)")


def main():
    print("=" * 70)
    print("  LeWM Predictor Scaling Benchmark (CPU)")
    print("=" * 70)

    embed_dim = 192
    # dense around the crossover region where O(T²) starts to dominate
    seq_lengths = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    horizons = [6, 10, 14, 18, 26, 34, 50, 66, 98, 130]
    history_size = 4

    predictors = make_predictor_variants(embed_dim=embed_dim, depth=3, history_size=history_size)
    print_parameter_counts(predictors)

    # --- forward benchmark ---
    forward_results = benchmark_forward(predictors, seq_lengths,
                                        embed_dim=embed_dim, batch_size=4, num_runs=10)
    print_relative_speedup(forward_results, seq_lengths)

    # --- rollout benchmark ---
    rollout_results = benchmark_rollout(predictors, horizons,
                                        embed_dim=embed_dim, history_size=history_size,
                                        num_runs=3)
    print_relative_speedup(rollout_results, horizons)

    print("\nDone.")


if __name__ == "__main__":
    main()
