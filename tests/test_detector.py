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
        assert detector.probability(wav) == pytest.approx(float(reference([wav])[0]), abs=1e-4)


def test_turn_detector_reads_int16_audio_and_rejects_stereo(tiny_onnx):
    detector = TurnDetector(tiny_onnx)
    wav = speech_like(5.0, seed=1)
    pcm = np.round(wav * 32767).astype(np.int16)
    assert detector.probability(pcm) == pytest.approx(detector.probability(pcm / 32768.0), abs=1e-6)
    with pytest.raises(ValueError):
        detector.probability(np.stack([wav, wav], axis=1))
