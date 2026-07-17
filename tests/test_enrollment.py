"""
test_enrollment.py

Unit tests for EnrollmentSession.

External dependencies are mocked:

- recorder.record_one
- validator.check_audio
- audio_utils.save_audio

DatasetManager is used normally against pytest's temporary directory.
"""

import pytest

import vocalid.enrollment as enrollment_module
from vocalid.enrollment import EnrollmentSession


# ---------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_recording(monkeypatch):

    monkeypatch.setattr(
        enrollment_module,
        "record_one",
        lambda seconds, sample_rate: "fake_audio"
    )

    monkeypatch.setattr(
        "vocalid.audio_utils.save_audio",
        lambda audio, path, sr: None
    )


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_accepts_valid_clip(monkeypatch, tmp_path):

    monkeypatch.setattr(
        enrollment_module,
        "check_audio",
        lambda path: (True, "ok")
    )

    session = EnrollmentSession(
        dataset_root=str(tmp_path / "dataset")
    )

    saved = session.enroll(
        label="positive",
        count=1,
        seconds=5.0
    )

    assert len(saved) == 1


def test_retries_after_validation_failure(monkeypatch, tmp_path):

    results = iter([
        (False, "too quiet"),
        (True, "ok")
    ])

    monkeypatch.setattr(
        enrollment_module,
        "check_audio",
        lambda path: next(results)
    )

    session = EnrollmentSession(
        dataset_root=str(tmp_path / "dataset"),
        max_attempts_per_sample=3
    )

    saved = session.enroll(
        label="positive",
        count=1,
        seconds=5.0
    )

    assert len(saved) == 1


def test_gives_up_after_max_attempts(monkeypatch, tmp_path):

    monkeypatch.setattr(
        enrollment_module,
        "check_audio",
        lambda path: (False, "too quiet")
    )

    session = EnrollmentSession(
        dataset_root=str(tmp_path / "dataset"),
        max_attempts_per_sample=2
    )

    saved = session.enroll(
        label="positive",
        count=1,
        seconds=5.0
    )

    assert saved == []


def test_returns_path_for_every_sample(monkeypatch, tmp_path):

    monkeypatch.setattr(
        enrollment_module,
        "check_audio",
        lambda path: (True, "ok")
    )

    session = EnrollmentSession(
        dataset_root=str(tmp_path / "dataset")
    )

    saved = session.enroll(
        label="negative",
        count=3,
        seconds=5.0
    )

    assert len(saved) == 3
    assert len(set(saved)) == 3


def test_invalid_label_is_rejected(tmp_path):

    session = EnrollmentSession(
        dataset_root=str(tmp_path / "dataset")
    )

    with pytest.raises(
        ValueError,
        match="label must be 'positive' or 'negative'"
    ):

        session.enroll(
            label="kai",
            count=1,
            seconds=5.0
        )