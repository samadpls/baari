"""Kaggle job: fine-tune baari from Indic Smart Turn tiny, export ONNX and score it.

Inputs: the clips made by samadpls/baari-build-clips, plus the training rows of
adimyth/indic-smart-turn-data (public, downloaded here) so the model keeps what it knows.
Output: run/ with baari-fp32.onnx, baari-int8.onnx, baari.pt and metrics.json. Needs no secrets.
"""

import os
import subprocess
import sys
from pathlib import Path

BRANCH = "main"
ARGS = ["--epochs", "3", "--lr", "2e-5", "--batch", "64", "--workers", "4"]

pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run([*pip, f"baari[pipeline] @ git+https://github.com/samadpls/baari@{BRANCH}"], check=True)

from huggingface_hub import snapshot_download  # installed by the line above

indic = Path(
    snapshot_download(
        "adimyth/indic-smart-turn-data",
        repo_type="dataset",
        allow_patterns=["built/*.parquet"],
        local_dir="/tmp/indic",
    )
)
# Kaggle mounts another job's output at /kaggle/input/<slug> or deeper, depending on when it was attached.
clips = next(p.parent for p in Path("/kaggle/input").rglob("valid.parquet") if p.parent.name == "clips")
extra = [a for p in sorted((indic / "built").glob("*.parquet")) for a in ("--extra", str(p))]
baari = [sys.executable, "-c", "from baari.cli import app; app()"]
env = dict(os.environ, DATA_DIR="/tmp/baari")
train = ["train", "--clips", str(clips / "train.parquet"), "--test", str(clips / "valid.parquet")]
subprocess.run([*baari, *train, "--out", "/kaggle/working/run", *extra, *ARGS], check=True, env=env)
