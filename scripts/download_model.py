"""Download Gemma 3 270M into a local project folder.

Hugging Face is the default source.  The Gemma repo is gated there, so a logged
in HF account with the Google license accepted is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-3-270m")
    parser.add_argument("--local-dir", default="models/gemma-3-270m")
    parser.add_argument("--provider", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is not installed. Run:\n"
                "  D:\\.conda\\envs\\py310\\python.exe -m pip install huggingface_hub"
            ) from exc
        path = snapshot_download(
            repo_id=args.model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "modelscope is not installed. Run:\n"
                "  D:\\.conda\\envs\\py310\\python.exe -m pip install modelscope"
            ) from exc
        path = snapshot_download(args.model_id, local_dir=str(local_dir))
    print(path)


if __name__ == "__main__":
    main()
