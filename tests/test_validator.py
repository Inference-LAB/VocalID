"""
test_validator.py
"""

from vocalid.validator import check_audio


def test_accepts_good_clip(write_wav):
    path = write_wav("good.wav", duration=5.0)
    is_valid, reason = check_audio(path)
    assert is_valid is True
    assert reason == "ok"


def test_rejects_too_short(write_wav):
    path = write_wav("short.wav", duration=1.5)
    is_valid, reason = check_audio(path)
    assert is_valid is False
    assert "short" in reason


def test_rejects_too_long(write_wav):
    path = write_wav("long.wav", duration=8.0)
    is_valid, reason = check_audio(path)
    assert is_valid is False
    assert "long" in reason


def test_rejects_silence(write_wav):
    path = write_wav("silent.wav", duration=5.0, silent=True)
    is_valid, reason = check_audio(path)
    assert is_valid is False
    assert "quiet" in reason or "silence" in reason


def test_rejects_clipping(write_wav):
    path = write_wav("clipped.wav", duration=5.0, clipped=True)
    is_valid, reason = check_audio(path)
    assert is_valid is False
    assert "clip" in reason
