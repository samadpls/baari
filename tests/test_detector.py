from pathlib import Path

import numpy as np
import pytest

from baari import TurnDetector
from baari.indicvoices import SR


def speech_like(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (
        0.3 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))
        + 0.05 * rng.standard_normal(t.size)
    ).astype(np.float32)


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from transformers import WhisperConfig

    from baari import train

    config = WhisperConfig(
        d_model=16, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=32, num_mel_bins=80
    )
    torch.manual_seed(0)
    model = train.SmartTurn(config).eval()
    for value in model.state_dict().values():
        value.copy_(torch.randn_like(value))
    path = tmp_path_factory.mktemp("model") / "m.onnx"
    train.to_onnx(model, path)
    return str(path)


def test_numpy_features_match_the_transformers_feature_extractor():
    from transformers import WhisperFeatureExtractor

    from baari.features import compute_whisper_log_mel_features

    wav = speech_like(8.0, seed=0)
    reference = WhisperFeatureExtractor(chunk_length=8)(
        wav, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=8 * SR, do_normalize=True
    ).input_features[0]
    assert np.abs(compute_whisper_log_mel_features(wav) - reference).max() < 1e-4


def test_turn_detector_gives_the_same_probability_as_the_evaluation_code(tiny_onnx):
    from baari.evaluate import OnnxModel

    detector, reference = TurnDetector(tiny_onnx), OnnxModel(tiny_onnx)
    for seconds in (3.0, 8.0, 11.5):
        wav = speech_like(seconds, seed=int(seconds))
        assert detector.probability(wav, sample_rate=SR) == pytest.approx(
            float(reference([wav])[0]), abs=1e-4
        )


def test_turn_detector_reads_int16_audio_and_rejects_stereo(tiny_onnx):
    detector = TurnDetector(tiny_onnx)
    wav = speech_like(5.0, seed=1)
    pcm = np.round(wav * 32767).astype(np.int16)
    same = detector.probability(pcm / 32768.0, sample_rate=SR)
    assert detector.probability(pcm, sample_rate=SR) == pytest.approx(same, abs=1e-6)
    with pytest.raises(ValueError):
        detector.probability(np.stack([wav, wav], axis=1), sample_rate=SR)


def test_turn_detector_refuses_audio_that_is_not_16_khz(tiny_onnx):
    with pytest.raises(ValueError, match="16000 Hz"):
        TurnDetector(tiny_onnx).is_finished(speech_like(2.0, seed=2)[::2], sample_rate=8000)


def test_model_download_lands_complete_under_its_revision(tmp_path, monkeypatch):
    from baari import detector

    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx bytes")
    monkeypatch.setattr(detector, "MODEL_URL", source.as_uri())
    monkeypatch.setenv("BAARI_CACHE", str(tmp_path / "cache"))
    path = detector.model_path()
    assert Path(path).read_bytes() == b"onnx bytes"
    assert path.endswith(f"{detector.MODEL_REVISION}/baari-int8.onnx")
    assert [p.name for p in (tmp_path / "cache" / detector.MODEL_REVISION).iterdir()] == ["baari-int8.onnx"]


def test_failed_model_download_leaves_no_partial_file(tmp_path, monkeypatch):
    from baari import detector

    monkeypatch.setattr(detector, "MODEL_URL", (tmp_path / "missing.onnx").as_uri())
    monkeypatch.setenv("BAARI_CACHE", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="Could not download"):
        detector.model_path()
    assert not [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
