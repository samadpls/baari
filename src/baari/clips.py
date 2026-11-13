"""Turn labelled segments into Smart Turn style clips: the last 8 s of 16 kHz audio.

Adapted from indic_turn/build.py in Indic Smart Turn (https://github.com/adimyth/indic-smart-turn).
Copyright (c) 2026, adimyth. BSD-2-Clause; see LICENSE.indic-smart-turn.

A segment's clip is the segment plus the preceding consecutive segments of the same speaker,
joined with a short gap after an unfinished segment and a long one after a finished segment
(the other person would have spoken in between), with trailing silence trimmed to 0.2 s.
"""

import io
import random

import numpy as np
import pyarrow as pa
import soundfile as sf

from baari.indicvoices import SR
from baari.labels import Key, SegmentLabel

WINDOW = 8 * SR
TRAIL = int(0.2 * SR)


def frame_energy(x: np.ndarray, win: int = 400, hop: int = 160) -> np.ndarray:
    n = max(0, (len(x) - win) // hop + 1)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)


def trim_trailing_silence(
    x: np.ndarray, rel: float = 0.03, floor: float = 1e-3, hop: int = 160
) -> np.ndarray:
    e = frame_energy(x, hop=hop)
    if len(e) == 0:
        return x
    voiced = np.flatnonzero(e > max(floor, rel * e.max()))
    if len(voiced) == 0:
        return x
    return x[: min(len(x), (voiced[-1] + 1) * hop + 400)]


def fit_window(x: np.ndarray) -> np.ndarray:
    x = np.concatenate([trim_trailing_silence(x), np.zeros(TRAIL, np.float32)])
    return x[-WINDOW:]


def build(
    chunks: list[dict], audio: dict[int, np.ndarray], labels: dict[Key, SegmentLabel], rng: random.Random
) -> list[np.ndarray | None]:
    """One clip per chunk of one session, in order; None where the chunk's audio is missing."""
    clips = []
    for i, c in enumerate(chunks):
        if c["chunk"] not in audio:
            clips.append(None)
            continue
        parts, total, j = [audio[c["chunk"]]], len(audio[c["chunk"]]), i - 1
        while (
            total < WINDOW
            and j >= 0
            and chunks[j]["chunk"] == chunks[j + 1]["chunk"] - 1
            and chunks[j]["chunk"] in audio
        ):
            prev = chunks[j]
            finished = labels[(prev["session"], prev["chunk"])].complete
            gap = np.zeros(
                int((rng.uniform(1.5, 4.0) if finished else rng.uniform(0.15, 0.6)) * SR), np.float32
            )
            parts = [audio[prev["chunk"]], gap, *parts]
            total += len(gap) + len(audio[prev["chunk"]])
            j -= 1
        clips.append(fit_window(np.concatenate(parts)))
    return clips


def wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def table(rows: list[dict]) -> pa.Table:
    """rows carry 'audio' as float32 arrays; stored as 16-bit WAV bytes, as Hugging Face Audio expects."""
    return pa.Table.from_pylist(
        [
            {**r, "audio": {"bytes": wav_bytes(r["audio"]), "path": f"{r['session']}_{r['chunk']}.wav"}}
            for r in rows
        ]
    )
