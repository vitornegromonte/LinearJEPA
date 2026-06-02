"""CLI entry point for synthetic data generation."""

import argparse
import os
import numpy as np

from .tasks import TASKS, linear_ar, nback, delayed_copy, slowfast, chaotic
from .render import render_state_to_image
from .h5_writer import write_h5


def _get_generator(task_name):
    mapping = {
        "linear_ar": linear_ar,
        "nback": nback,
        "delayed_copy": delayed_copy,
        "slowfast": slowfast,
        "chaotic": chaotic,
    }
    return mapping[task_name]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LeWM dataset")
    parser.add_argument("--task", choices=list(TASKS), default="linear_ar")
    parser.add_argument("--modality", choices=["state", "image", "both"], default="state")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--ep-len", type=int, default=100,
                        help="Length of each episode in raw steps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--obs-dim", type=int, default=None,
                        help="Override task's default obs_dim")
    parser.add_argument("--action-dim", type=int, default=None,
                        help="Override task's default action_dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="Output path (default: $STABLEWM_HOME/synth_<task>.h5)")

    # Task-specific params (parsed as --param value)
    parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    args, unknown = parser.parse_known_args()

    rng = np.random.RandomState(args.seed)
    task_fn = _get_generator(args.task)
    _, default_params = TASKS[args.task]

    params = dict(default_params)
    if args.obs_dim is not None:
        params["obs_dim"] = args.obs_dim
    if args.action_dim is not None:
        params["action_dim"] = args.action_dim
    # Override from --key=value extra args
    for arg in unknown:
        if arg.startswith("--"):
            kv = arg.lstrip("-").split("=", 1)
            if len(kv) == 2:
                k, v = kv
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                params[k] = v

    out = args.out
    if out is None:
        home = os.environ.get("STABLEWM_HOME", os.path.join(os.getcwd(), "data"))
        os.makedirs(home, exist_ok=True)
        out = os.path.join(home, f"synth_{args.task}.h5")

    obs_dim = params["obs_dim"]
    action_dim = params["action_dim"]

    print(f"Generating {args.num_episodes} episodes of {args.task} "
          f"(T={args.ep_len}, obs={obs_dim}, act={action_dim})...")

    # Generate all episodes
    episodes = []
    for ep_idx in range(args.num_episodes):
        ep_rng = np.random.RandomState(args.seed * 1000 + ep_idx)
        ep = task_fn(ep_rng, args.ep_len, action_dim=action_dim, obs_dim=obs_dim,
                     **{k: v for k, v in params.items()
                        if k not in ("obs_dim", "action_dim")})
        if args.modality in ("image", "both"):
            pixels = np.empty((args.ep_len, args.image_size, args.image_size, 3),
                              dtype=np.uint8)
            prev = None
            for t in range(args.ep_len):
                pixels[t] = render_state_to_image(
                    ep["state"][t], ep["action"][t], args.task,
                    args.image_size, args.image_size, prev,
                )
                prev = pixels[t]
            ep["pixels"] = pixels
        episodes.append(ep)

    write_h5(episodes, args.task, out, params,
             image_size=args.image_size if args.modality in ("image", "both") else None)

    disk_mb = os.path.getsize(out) / 1e6
    print(f"  Written to {out} ({disk_mb:.0f} MB)")


if __name__ == "__main__":
    main()
