"""Run baari on audio. Needs only numpy and onnxruntime."""

import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from baari.features import compute_whisper_log_mel_features

SAMPLE_RATE = 16000
MODEL_URL = "https://huggingface.co/samadpls/baari/resolve/main/baari-int8.onnx"


def model_path() -> str:
    """Path to baari's int8 ONNX file, downloaded once into ~/.cache/baari (or $BAARI_CACHE)."""
    path = Path(os.environ.get("BAARI_CACHE", Path.home() / ".cache" / "baari")) / "baari-int8.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, part)
        except OSError as e:
            raise RuntimeError(
                f"Could not download the baari model from {MODEL_URL} ({e}). "
                "Pass TurnDetector(path=...) to use a local Smart Turn style ONNX file."
            ) from e
        part.replace(path)
    return str(path)


class TurnDetector:
    """Decides, at a pause, whether an Urdu speaker has finished their turn.

    threads: CPU threads for one call. 1 suits a server handling many calls at once; 4 makes a single
    call about three times faster.
    """

    def __init__(self, path: str | None = None, threads: int = 1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path or model_path(), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def probability(self, audio: np.ndarray) -> float:
        """Probability that the speaker has finished. `audio` is 16 kHz mono, float or int16, ending at
        the pause. Only the last 8 seconds are used."""
        x = np.asarray(audio)
        if x.ndim != 1:
            raise ValueError(f"expected mono audio as a 1-D array, got shape {x.shape}")
        x = x / 32768.0 if x.dtype == np.int16 else x
        n = 8 * SAMPLE_RATE
        x = x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))
        features = compute_whisper_log_mel_features(x.astype(np.float32))[None]
        return float(np.asarray(self.session.run(None, {"input_features": features})[0]).ravel()[0])

    def is_finished(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """True when the speaker has finished and the agent should reply."""
        return self.probability(audio) > threshold
