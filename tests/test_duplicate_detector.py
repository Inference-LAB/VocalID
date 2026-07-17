"""
test_duplicate_detector.py

The real EmbeddingExtractor loads a full ECAPA-TDNN model from
speechbrain, which is slow and needs a network connection on first
run. For unit tests we swap it for a fake extractor that just turns
the raw audio into its "embedding" - two clips with the same tone
frequency come out identical, different frequencies come out
different, which is enough to exercise the cosine-similarity logic
in duplicate_detector.py itself. Swap this fixture out once you're
ready to write a slower, real-model integration test.
"""

import numpy as np
import soundfile as sf
import pytest

import vocalid.duplicate_detector as dd_module
from vocalid.duplicate_detector import DuplicateDetector


class _FakeExtractor:
    def extract(self, audio):
        return np.asarray(audio).flatten()


def _fake_load_audio(path):
    samples, sample_rate = sf.read(path, dtype="float32")
    return samples, sample_rate


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(dd_module, "EmbeddingExtractor", _FakeExtractor)
    monkeypatch.setattr(dd_module, "load_audio", _fake_load_audio)
    return DuplicateDetector()


def test_identical_clip_is_flagged(detector, write_wav):
    original = write_wav("original.wav", freq=220.0)
    replay = write_wav("replay.wav", freq=220.0)  # same tone == same "voice"

    match = detector.find_duplicate(replay, [original])
    assert match == original
    assert detector.is_duplicate(replay, [original]) is True


def test_different_clip_is_not_flagged(detector, write_wav):
    original = write_wav("original.wav", freq=220.0)
    different = write_wav("different.wav", freq=880.0)  # different tone == different "voice"

    match = detector.find_duplicate(different, [original])
    assert match is None
    assert detector.is_duplicate(different, [original]) is False


def test_no_existing_clips_never_flags(detector, write_wav):
    clip = write_wav("only.wav")
    assert detector.find_duplicate(clip, []) is None


def test_custom_threshold_is_respected(write_wav, monkeypatch):
    monkeypatch.setattr(dd_module, "EmbeddingExtractor", _FakeExtractor)
    monkeypatch.setattr(dd_module, "load_audio", _fake_load_audio)

    strict_detector = DuplicateDetector(threshold=0.999999)
    original = write_wav("original.wav", freq=220.0, amplitude=0.3)
    slightly_different = write_wav("close.wav", freq=222.0, amplitude=0.3)

    # Close but not identical tones shouldn't clear an extremely strict threshold.
    assert strict_detector.is_duplicate(slightly_different, [original]) is False
