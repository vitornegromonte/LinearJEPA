#!/usr/bin/env python3
"""Synthetic dataset generator for LeWM step-generalization benchmarks.

Quick start:
  python synth_data.py --task linear_ar --modality state --num-episodes 200 --ep-len 128
  python train.py data=synth model=lewm_deltanet wandb.enabled=False

All tasks:
  python synth_data.py --task linear_ar    # (default) linear AR
  python synth_data.py --task nback        # N-back discrete memory
  python synth_data.py --task delayed_copy # Delayed pattern recall
  python synth_data.py --task slowfast     # Multi-scale dynamics
  python synth_data.py --task chaotic      # Lorenz-like chaotic

Image modality:
  python synth_data.py --task linear_ar --modality image --image-size 64
"""

import sys
from synth_data.cli import main

if __name__ == "__main__":
    main()
