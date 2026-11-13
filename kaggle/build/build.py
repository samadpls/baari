"""Kaggle job: cut the baari Urdu clips (test and training) from IndicVoices.

Inputs: the label dataset samadpls/baari-labels, and a Hugging Face token that can read
ai4bharat/IndicVoices as the Kaggle secret HF_API (Add-ons > Secrets). Kaggle only hands secrets to
runs started from its website, so start it there with Save Version > Save & Run All.
Output: data/clips/valid.parquet and data/clips/train.parquet, about 8.5 GB.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

BRANCH = "main"

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_API")
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run([*pip, f"baari[pipeline] @ git+https://github.com/samadpls/baari@{BRANCH}"], check=True)

data = Path("/kaggle/working/data")
# Kaggle mounts a dataset at /kaggle/input/<slug> or deeper, depending on when it was attached.
labels = next(p.parent.parent for p in Path("/kaggle/input").rglob("valid.parquet") if p.parent.name == "index")
for part in ("index", "labels"):
    shutil.copytree(labels / part, data / part, dirs_exist_ok=True)
baari = [sys.executable, "-c", "from baari.cli import app; app()"]
env = dict(os.environ, DATA_DIR=str(data))
for split, labelers in (("valid", "a,b"), ("train", "a")):
    subprocess.run(
        [*baari, "build", "--split", split, "--labelers", labelers, "--delete-shards"], check=True, env=env
    )
shutil.rmtree(data / "hf", ignore_errors=True)
