"""Score existing end-of-turn models on the Urdu clips.

Smart Turn style models take an 80 x 800 log-mel of the last 8 s and return P(finished).
PuristanLabs1/urdu-turn-v2 is a PyTorch Whisper-tiny encoder with attention pooling, trained on
3 s clips; its documented package is not on PyPI, so it is rebuilt here from its model.py.
"""

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from transformers import WhisperFeatureExtractor

from baari.indicvoices import SR

ONNX_MODELS = {
    "Smart Turn v3.2 (CPU, int8)": ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"),
    "Smart Turn v3.2 (GPU, fp32)": ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"),
    "Indic Smart Turn tiny (int8)": ("adimyth/indic-smart-turn", "indic-smart-turn-tiny-int8.onnx"),
    "Indic Smart Turn tiny (fp32)": ("adimyth/indic-smart-turn", "indic-smart-turn-tiny-fp32.onnx"),
    "Indic Smart Turn base (fp32)": ("adimyth/indic-smart-turn", "indic-smart-turn-base-fp32.onnx"),
}
PURISTAN = "PuristanLabs1/urdu-turn-v2 (Urdu)"


def last(x: np.ndarray, seconds: int) -> np.ndarray:
    n = seconds * SR
    return x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))


class OnnxModel:
    fe = WhisperFeatureExtractor(chunk_length=8)

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, wavs: list[np.ndarray]) -> np.ndarray:
        feats = self.fe(
            [last(w, 8) for w in wavs],
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=8 * SR,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        p = np.asarray(self.sess.run(None, {self.input: feats})[0], dtype=np.float64).reshape(len(wavs), -1)[
            :, -1
        ]
        return p if 0 <= p.min() and p.max() <= 1 else 1 / (1 + np.exp(-p))


class PuristanModel:
    def __init__(self, cache: Path):
        import torch
        from torch import nn
        from transformers import WhisperConfig, WhisperModel

        class WhisperTurnDetector(
            nn.Module
        ):  # same layout as the repo's model.py, so the state dict loads strictly
            def __init__(self):
                super().__init__()
                self.whisper = WhisperModel(
                    WhisperConfig.from_pretrained("openai/whisper-tiny", cache_dir=cache)
                )
                h = self.whisper.config.d_model
                self.attn = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
                self.silence_embedding = nn.Parameter(torch.zeros(h))
                self.classifier = nn.Sequential(
                    nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)
                )

            def forward(self, feats, mask):
                hidden = self.whisper.encoder(feats).last_hidden_state
                weights = self.attn(hidden).squeeze(-1).masked_fill(mask[:, ::2] == 0, -1e9)
                pooled = torch.bmm(torch.softmax(weights, dim=1).unsqueeze(1), hidden).squeeze(1)
                return self.classifier(pooled)

        self.torch = torch
        self.model = WhisperTurnDetector().eval()
        weights = hf_hub_download("PuristanLabs1/urdu-turn-v2", "whisper_turn_model.pth", cache_dir=cache)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", cache_dir=cache)

    def __call__(self, wavs: list[np.ndarray], seconds: int = 3) -> np.ndarray:
        f = self.fe(
            [w[-seconds * SR :] for w in wavs],
            sampling_rate=SR,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with self.torch.no_grad():
            logits = self.model(f.input_features, f.attention_mask)
        return self.torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float64)


def load_clips(path: Path, subset: str) -> tuple[list[np.ndarray], np.ndarray]:
    t = pq.read_table(path).to_pydict()
    keep = [i for i, s in enumerate(t["split"]) if subset == "all" or s == subset]
    wavs = [sf.read(io.BytesIO(t["audio"][i]["bytes"]), dtype="float32")[0] for i in keep]
    return wavs, np.array([int(t["endpoint_bool"][i]) for i in keep])


def score(model, wavs: list[np.ndarray], batch: int = 32) -> np.ndarray:
    return np.concatenate([model(wavs[i : i + batch]) for i in range(0, len(wavs), batch)])


def balanced(pred: np.ndarray, y: np.ndarray) -> float:
    return ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean()) / 2


def metrics(p: np.ndarray, y: np.ndarray, resamples: int = 1000, seed: int = 0) -> dict:
    """Balanced accuracy is the headline: most clips are finished, so plain accuracy rewards always saying 'finished'."""
    pred = (p > 0.5).astype(int)
    rng = np.random.default_rng(seed)
    boot = [balanced(pred[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(resamples))]
    return {
        "balanced": balanced(pred, y),
        "ci": np.quantile(boot, [0.025, 0.975]),
        "acc": (pred == y).mean(),
        "waits": (pred[y == 0] == 0).mean(),
        "replies": (pred[y == 1] == 1).mean(),
        "auc": roc_auc_score(y, p) if p.std() > 0 else 0.5,
    }


def run(clips: Path, subset: str, cache: Path) -> str:
    wavs, y = load_clips(clips, subset)
    models = {
        name: OnnxModel(hf_hub_download(repo, f, cache_dir=cache)) for name, (repo, f) in ONNX_MODELS.items()
    }
    models[PURISTAN] = PuristanModel(cache)
    lines = [
        f"# Existing models on Urdu clips ({clips.stem}, {subset})",
        "",
        f"{len(y)} clips: {int(y.sum())} finished, {int(len(y) - y.sum())} unfinished. Threshold 0.5.",
        "Balanced accuracy is the mean of the two middle columns.",
        "",
        "| Model | Balanced accuracy (95% CI) | Waits when unfinished | Replies when finished | Plain accuracy | AUC |",
        "|---|---|---|---|---|---|",
    ]
    scores = {"Always says finished (reference)": np.ones(len(y))}
    scores |= {name: score(model, wavs) for name, model in models.items()}
    for name, p in scores.items():
        m = metrics(p, y)
        lines.append(
            f"| {name} | {100 * m['balanced']:.1f}% ({100 * m['ci'][0]:.1f}–{100 * m['ci'][1]:.1f}) | "
            f"{100 * m['waits']:.1f}% | {100 * m['replies']:.1f}% | {100 * m['acc']:.1f}% | {m['auc']:.3f} |"
        )
    return "\n".join(lines) + "\n"
