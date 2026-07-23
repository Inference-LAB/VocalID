"""
test_recorder.py

record_audio / save_audio (from audio_utils) are mocked out - these
tests check recorder.py's own logic (sample-rate defaulting, file
naming, loop count), not the real microphone or real file writing.
"""

import pytest
import vocalid.recorder as recorder_module


@pytest.fixture
def fake_audio_utils(monkeypatch):
    calls = {"record_audio": [], "save_audio": []}

    def fake_record_audio(duration, sample_rate):
        calls["record_audio"].append((duration, sample_rate))
        return f"audio@{duration}s"

    def fake_save_audio(audio, path, sample_rate):
        calls["save_audio"].append((audio, path, sample_rate))

    monkeypatch.setattr(recorder_module, "record_audio", fake_record_audio)
    monkeypatch.setattr(recorder_module, "save_audio", fake_save_audio)
    return calls


def test_record_one_uses_given_seconds_and_rate(fake_audio_utils):
    audio = recorder_module.record_one(seconds=6.0, sample_rate=22050)
    assert audio == "audio@6.0s"
    assert fake_audio_utils["record_audio"] == [(6.0, 22050)]


def test_record_one_falls_back_to_config_sample_rate(fake_audio_utils, monkeypatch):
    monkeypatch.setattr(recorder_module.config, "SAMPLE_RATE", 16000)
    recorder_module.record_one(seconds=5.0, sample_rate=None)
    assert fake_audio_utils["record_audio"] == [(5.0, 16000)]


def test_record_batch_records_and_saves_expected_count(fake_audio_utils, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    paths = recorder_module.record_batch(str(tmp_path), count=3, seconds=5.0, sample_rate=16000)

    assert len(paths) == 3
    assert len(fake_audio_utils["record_audio"]) == 3
    assert len(fake_audio_utils["save_audio"]) == 3


def test_record_batch_names_files_sequentially(fake_audio_utils, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    paths = recorder_module.record_batch(str(tmp_path), count=2, prefix="clip")

    assert paths[0].endswith("clip_001.wav")
    assert paths[1].endswith("clip_002.wav")


def test_record_batch_creates_output_dir(fake_audio_utils, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    out_dir = tmp_path / "nested" / "clips"

    recorder_module.record_batch(str(out_dir), count=1)

    assert out_dir.is_dir()
