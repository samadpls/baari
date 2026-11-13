import random

import numpy as np
import pytest

from baari import clips, labels
from baari.indicvoices import SR, without_test


def tone(seconds: float) -> np.ndarray:
    return (0.5 * np.sin(np.arange(int(seconds * SR)) / 10)).astype(np.float32)


def test_fit_window_trims_trailing_silence_and_keeps_last_8_s():
    x = np.concatenate([tone(10), np.zeros(SR, np.float32)])
    out = clips.fit_window(x)
    assert len(out) == clips.WINDOW
    assert np.abs(out[-clips.TRAIL :]).max() == 0
    assert np.abs(out[-clips.TRAIL - 800 : -clips.TRAIL]).max() > 0.1


def test_build_joins_only_consecutive_chunks_with_label_dependent_gaps():
    chunks = [{"session": "s", "chunk": n} for n in (1, 2, 4)]
    audio = {n: tone(1) for n in (1, 2, 4)}
    lab = {
        ("s", 1): labels.SegmentLabel(session="s", chunk=1, complete=True, confidence=0.9),
        ("s", 2): labels.SegmentLabel(session="s", chunk=2, complete=False, confidence=0.9),
        ("s", 4): labels.SegmentLabel(session="s", chunk=4, complete=True, confidence=0.9),
    }
    out = clips.build(chunks, audio, lab, random.Random(0))
    lengths = [len(c) / SR for c in out]
    assert lengths[1] - lengths[0] >= 1 + 1.5  # chunk 1 finished, so a long gap before chunk 2
    assert lengths[2] < 1.3  # chunk 3 is missing, so chunk 4 gets no context


def test_load_results_rejects_missing_and_duplicate_labels(tmp_path):
    index = [{"session": "s", "chunk": 1}, {"session": "s", "chunk": 2}]
    line = '{"session": "s", "chunk": 1, "complete": true, "confidence": 0.9}\n'
    (tmp_path / "batch-000.jsonl").write_text(line)
    with pytest.raises(ValueError, match="1 segments unlabelled"):
        labels.load_results(tmp_path, index)
    (tmp_path / "batch-000.jsonl").write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        labels.load_results(tmp_path, index)


def test_without_test_drops_test_speakers_and_both_sides_of_test_calls():
    test = [{"session": "call1_0", "speaker_id": "A"}]
    rows = [
        {"session": "call1_1", "speaker_id": "B"},  # other side of a test call
        {"session": "call2_0", "speaker_id": "A"},  # test speaker in another call
        {"session": "call3_0", "speaker_id": "C"},
    ]
    assert without_test(rows, test) == [rows[2]]
