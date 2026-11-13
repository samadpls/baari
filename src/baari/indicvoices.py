"""Index the Urdu conversation segments of IndicVoices and fetch their audio.

The index reads only the small text columns over HTTP, so scanning a 0.5 GB shard
costs a few MB. Audio is downloaded per shard only when clips are built.
"""

import io
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

REPO = "ai4bharat/IndicVoices"
SR = 16_000
# Only conversation segments are named <session uuid>_<part>_chunk_<n>.flac.
PATH_RE = re.compile(r"^(?P<uuid>[0-9a-f-]{36})_(?P<part>\d+)_chunk_(?P<chunk>\d+)\.flac$")
COLUMNS = ["audio_filepath.path", "task_name", "speaker_id", "duration", "text"]


def shards(split: str, token: str) -> list[str]:
    files = HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"urdu/{split}-") and f.endswith(".parquet"))


def index_shard(shard: str, token: str) -> list[dict]:
    """One row per conversation segment: where it is, which session and chunk, and its transcript."""
    with HfFileSystem(token=token).open(f"datasets/{REPO}/{shard}", block_size=1 << 20) as f:
        t = pq.ParquetFile(f).read(columns=COLUMNS).to_pydict()
    rows = []
    for i, (p, task) in enumerate(zip(t["audio_filepath"], t["task_name"], strict=True)):
        m = PATH_RE.match((p["path"] if isinstance(p, dict) else p) or "")
        if task == "Conversation" and m:
            rows.append(
                {
                    "shard": shard,
                    "row": i,
                    "session": f"{m['uuid']}_{m['part']}",
                    "chunk": int(m["chunk"]),
                    "speaker_id": t["speaker_id"][i],
                    "duration": float(t["duration"][i]),
                    "text": t["text"][i] or "",
                }
            )
    return rows


def call_id(session: str) -> str:
    return session.rsplit("_", 1)[0]


def without_test(rows: list[dict], test: list[dict]) -> list[dict]:
    """Drop every segment from a test speaker or from either side of a test call.

    IndicVoices puts many valid-split speakers, and the other side of many valid calls, in train.
    """
    speakers = {r["speaker_id"] for r in test}
    calls = {call_id(r["session"]) for r in test}
    return [r for r in rows if r["speaker_id"] not in speakers and call_id(r["session"]) not in calls]


def decode(flac: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(flac), dtype="float32")
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    return x if x.ndim == 1 else x.mean(axis=1)


def shard_audio(shard: str, rows: set[int], token: str, cache: Path) -> dict[int, np.ndarray]:
    """Download one shard and decode the requested rows."""
    path = hf_hub_download(REPO, shard, repo_type="dataset", token=token, cache_dir=cache)
    pf = pq.ParquetFile(path)
    out, base = {}, 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if want := [r for r in rows if base <= r < base + n]:
            col = pf.read_row_group(g, columns=["audio_filepath"]).column(0)
            for r in want:
                out[r] = decode(col[r - base].as_py()["bytes"])
        base += n
    return out
