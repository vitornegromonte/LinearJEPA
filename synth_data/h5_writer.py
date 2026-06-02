"""Write generated episodes to HDF5 in the schema expected by swm.data.load_dataset."""

import json
import os
import h5py
import numpy as np

from .tasks import TASKS


def write_h5(episodes, task_name, filepath, params=None, image_size=None):
    """Write a list of episode dicts to a .h5 file.

    Each episode dict has keys:
      - state: (T, obs_dim) float64
      - action: (T, action_dim) float64
      - info: dict (optional)

    If image_size is given, renders pixels to (T, image_size, image_size, 3) uint8.
    """
    if params is None:
        params = {}
    num_episodes = len(episodes)
    ep_lens = np.array([len(e["state"]) for e in episodes], dtype=np.int32)
    ep_offset = np.zeros(num_episodes, dtype=np.int64)
    ep_offset[1:] = np.cumsum(ep_lens[:-1])
    total_steps = int(ep_lens.sum())

    # Determine shapes from first episode
    state_shape = episodes[0]["state"].shape[1:]
    action_shape = episodes[0]["action"].shape[1:]

    with h5py.File(filepath, "w") as f:
        # Metadata
        f.create_dataset("ep_len", data=ep_lens, dtype=np.int32)
        f.create_dataset("ep_offset", data=ep_offset, dtype=np.int64)

        # State column
        f.create_dataset("state", shape=(total_steps, *state_shape),
                         dtype=np.float32, chunks=(1, *state_shape))
        # Action column
        f.create_dataset("action", shape=(total_steps, *action_shape),
                         dtype=np.float32, chunks=(1, *action_shape))

        # Proprio column (copy of state for compatibility)
        f.create_dataset("proprio", shape=(total_steps, *state_shape),
                         dtype=np.float32, chunks=(1, *state_shape))

        # Pixels column (optional)
        if image_size is not None:
            img_shape = (image_size, image_size, 3)
            pixel_dset = f.create_dataset(
                "pixels", shape=(total_steps, *img_shape),
                dtype=np.uint8, chunks=(1, *img_shape),
                compression="lzf"
            )
        else:
            pixel_dset = None

        # Write episodes flat
        idx = 0
        for ep in episodes:
            T = len(ep["action"])
            f["state"][idx:idx + T] = ep["state"].astype(np.float32)
            f["action"][idx:idx + T] = ep["action"].astype(np.float32)
            f["proprio"][idx:idx + T] = ep["state"].astype(np.float32)

            if pixel_dset is not None and "pixels" in ep:
                pixel_dset[idx:idx + T] = ep["pixels"]

            idx += T

    # Write meta.json
    meta_path = filepath.replace(".h5", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(dict(
            task=task_name,
            params=params,
            num_episodes=num_episodes,
            total_steps=int(total_steps),
            obs_dim=int(np.prod(state_shape)),
            action_dim=int(np.prod(action_shape)),
            image_size=image_size,
        ), f, indent=2)

    return filepath
