"""Run baari on audio. Needs only numpy and onnxruntime."""

import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from baari.features import compute_whisper_log_mel_features

SAMPLE_RATE = 16000
# Each release pins one model commit, so a cached file always matches the code that reads it and an
# updated model reaches people through a new release, never by changing under an old one.
MODEL_REVISION = "81a0b995fdb599aa75117a8727a97b6ac4b72c9d"  # samadpls/baari, first release
MODEL_URL = "https://huggingface.co/samadpls/baari/resolve/{revision}/{file}"
# int8 is about 2.5x faster; fp32 scores the same on every CPU, while int8 varies by about 2 points
# between Arm and x86.
MODEL_FILES = {"int8": "baari-int8.onnx", "fp32": "baari-fp32.onnx"}


def model_path(precision: str = "int8") -> str:
    """Path to a baari ONNX file ("int8" or "fp32"), downloaded once into ~/.cache/baari (or $BAARI_CACHE)."""
    if precision not in MODEL_FILES:
        raise ValueError(f"precision must be one of {sorted(MODEL_FILES)}, got {precision!r}")
    file = MODEL_FILES[precision]
    url = MODEL_URL.format(revision=MODEL_REVISION, file=file)
    path = Path(os.environ.get("BAARI_CACHE", Path.home() / ".cache" / "baari")) / MODEL_REVISION / file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file of its own per process, renamed into place only when complete, so several workers
        # starting at once can't read or write each other's half-finished download.
        fd, part = tempfile.mkstemp(dir=path.parent, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, part)
            os.replace(part, path)
        except OSError as e:
            Path(part).unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download the baari model from {url} ({e}). "
                "Pass TurnDetector(path=...) to use a local Smart Turn style ONNX file."
            ) from e
    return str(path)


class TurnDetector:
    """Decides, at a pause, whether an Urdu speaker has finished their turn.

    threads: CPU threads for one call. 1 suits a server handling many calls at once; 4 makes a single
    call about three times faster.
    precision: "int8" (default, fastest) or "fp32". Ignored when `path` points to a local ONNX file.
    """

    def __init__(self, path: str | None = None, threads: int = 1, precision: str = "int8"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path or model_path(precision), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def probability(self, audio: np.ndarray, *, sample_rate: int) -> float:
        """Probability that the speaker has finished. `audio` is mono, float or int16, and ends at the
        pause. Only the last 8 seconds are used."""
        if sample_rate != SAMPLE_RATE:
            # 8 kHz phone audio read as 16 kHz would be silently wrong, so refuse instead of guessing.
            raise ValueError(
                f"baari needs {SAMPLE_RATE} Hz audio, got {sample_rate} Hz. Resample first, for example "
                f"with soxr: soxr.resample(audio, {sample_rate}, {SAMPLE_RATE})"
            )
        x = np.asarray(audio)
        if x.ndim != 1:
            raise ValueError(f"expected mono audio as a 1-D array, got shape {x.shape}")
        x = x / 32768.0 if x.dtype == np.int16 else x
        n = 8 * SAMPLE_RATE
        x = x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))
        features = compute_whisper_log_mel_features(x.astype(np.float32))[None]
        return float(np.asarray(self.session.run(None, {"input_features": features})[0]).ravel()[0])

    def is_finished(self, audio: np.ndarray, *, sample_rate: int, threshold: float = 0.5) -> bool:
        """True when the speaker has finished and the agent should reply."""
        return self.probability(audio, sample_rate=sample_rate) > threshold
