#!/usr/bin/env python3
"""Download and decompress a LeWM dataset.

Usage:
  python download_data.py tworoom       # lightest (~3.4 GB compressed)
  python download_data.py pusht         # ~13 GB compressed
  python download_data.py reacher       # ~24 GB compressed
  python download_data.py cube          # ~46 GB compressed

Outputs the .h5 file under STABLEWM_HOME (default: $PWD/data).
Then just: python train.py model=lewm_deltanet data=<name>
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

DATASETS = {
    "tworoom": ("quentinll/lewm-tworooms", "tworoom.tar.zst"),
    "pusht": ("quentinll/lewm-pusht", "pusht_expert_train.h5.zst"),
    "reacher": ("quentinll/lewm-reacher", "reacher.tar.zst"),
    "cube": ("quentinll/lewm-cube", "cube_single_expert.tar.zst"),
}

# Synthetic datasets — generated locally, not downloaded from HF
SYNTH_TASKS = ["linear_ar", "nback", "delayed_copy", "slowfast", "chaotic"]


def main():
    parser = argparse.ArgumentParser(description="Download or generate LeWM dataset")
    all_choices = list(DATASETS) + SYNTH_TASKS
    parser.add_argument("dataset", choices=all_choices, help="Dataset name; synth tasks: " + ", ".join(SYNTH_TASKS))
    parser.add_argument(
        "--home", default=None,
        help="STABLEWM_HOME (default: $STABLEWM_HOME or $PWD/data)",
    )
    parser.add_argument("--modality", default="state", choices=["state", "image", "both"])
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--ep-len", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    home = args.home or os.environ.get("STABLEWM_HOME",
                                       os.path.join(os.getcwd(), "data"))
    os.makedirs(home, exist_ok=True)

    # Synthetic dataset — generate locally
    if args.dataset in SYNTH_TASKS:
        from synth_data.cli import main as synth_main
        synth_argv = [
            "synth_data.py",
            f"--task={args.dataset}",
            f"--modality={args.modality}",
            f"--num-episodes={args.num_episodes}",
            f"--ep-len={args.ep_len}",
            f"--image-size={args.image_size}",
            f"--seed={args.seed}",
            f"--out={os.path.join(home, f'synth_{args.dataset}.h5')}",
        ]
        import sys as _sys
        _old_argv, _sys.argv = _sys.argv, synth_argv
        try:
            synth_main()
        finally:
            _sys.argv = _old_argv
        return

    # Standard dataset — download from HF
    repo_id, archive_name = DATASETS[args.dataset]
    stem = archive_name.replace(".tar.zst", "").replace(".h5.zst", "")
    h5_path = os.path.join(home, f"{stem}.h5")

    if os.path.exists(h5_path):
        size = os.path.getsize(h5_path) / 1e9
        print(f"✓ {stem}.h5 already exists ({size:.1f} GB)")
        return

    from huggingface_hub import hf_hub_download

    print(f"Downloading {archive_name} from {repo_id}...")
    src = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                          filename=archive_name)

    if archive_name.endswith(".tar.zst"):
        print("Extracting tar.zst...")
        with tempfile.TemporaryDirectory() as tmp:
            ok = subprocess.run(
                ["tar", "--zstd", "-xf", src, "-C", tmp],
                capture_output=True, text=True,
            ).returncode == 0
            if not ok:
                print("  tar --zstd failed, using Python zstd fallback...")
                tarball = src.replace(".zst", "")
                import zstandard
                with open(src, "rb") as inp, open(tarball, "wb") as out:
                    dctx = zstandard.ZstdDecompressor()
                    with dctx.stream_reader(inp) as reader:
                        shutil.copyfileobj(reader, out)
                subprocess.run(["tar", "-xf", tarball, "-C", tmp], check=True)
                os.remove(tarball)

            for root, _dirs, files in os.walk(tmp):
                for f in files:
                    if f.endswith(".h5"):
                        shutil.move(os.path.join(root, f), h5_path)
                        break
    else:
        print("Decompressing zst -> hdf5...")
        import zstandard
        with open(src, "rb") as inp, open(h5_path, "wb") as out:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(inp) as reader:
                shutil.copyfileobj(reader, out)

    size = os.path.getsize(h5_path) / 1e9
    print(f"✓ {stem}.h5 ready ({size:.1f} GB) at {h5_path}")


if __name__ == "__main__":
    main()
