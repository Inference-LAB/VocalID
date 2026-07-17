"""
test_enrollment.py

record_one, audio_utils.save_audio, and DuplicateDetector are all
mocked so this exercises EnrollmentSession's retry/accept/give-up
logic without needing a mic, a real audio file, or the ECAPA model.
DatasetManager is used for real (against tmp_path) since it's just
file I/O and worth testing end to end here.
"""

import pytest
import vocalid.enrollment as enrollment_module
from vocalid.enrollment import EnrollmentSession


class _FakeDupeDetector:
    """queue of paths to report as duplicates, one per find_duplicate() call"""
    def __init__(self, duplicate_paths=None):
        self._duplicate_paths = list(duplicate_paths or [])

    def find_duplicate(self, new_path, existing_paths):
        return self._duplicate_paths.pop(0) if self._duplicate_paths else None


@pytest.fixture(autouse=True)
def patch_recording(monkeypatch):
    monkeypatch.setattr(enrollment_module, "record_one", lambda seconds, sample_rate: "fake_audio")
    # save_audio is imported lazily inside enroll(), so the real patch
    # point is the audio_utils module itself, not enrollment_module.
    monkeypatch.setattr("vocalid.audio_utils.save_audio", lambda audio, path, sr: None)


def test_accepts_a_valid_unique_clip_on_first_try(monkeypatch, tmp_path):
    monkeypatch.setattr(enrollment_module, "check_audio", lambda path: (True, "ok"))
    monkeypatch.setattr(enrollment_module, "DuplicateDetector", lambda: _FakeDupeDetector())

    session = EnrollmentSession(dataset_root=str(tmp_path / "dataset"))
    saved = session.enroll(label="positive", count=1, seconds=5.0)

    assert len(saved) == 1


def test_retries_after_a_rejected_clip(monkeypatch, tmp_path):
    results = iter([(False, "too quiet"), (True, "ok")])
    monkeypatch.setattr(enrollment_module, "check_audio", lambda path: next(results))
    monkeypatch.setattr(enrollment_module, "DuplicateDetector", lambda: _FakeDupeDetector())

    session = EnrollmentSession(dataset_root=str(tmp_path / "dataset"), max_attempts_per_sample=3)
    saved = session.enroll(label="positive", count=1, seconds=5.0)

    assert len(saved) == 1  # accepted on the second attempt


def test_retries_after_a_duplicate_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(enrollment_module, "check_audio", lambda path: (True, "ok"))
    # first attempt flagged as a duplicate, second attempt is clean
    monkeypatch.setattr(
        enrollment_module, "DuplicateDetector",
        lambda: _FakeDupeDetector(duplicate_paths=["dataset/my_voice/sample001.wav"])
    )

    session = EnrollmentSession(dataset_root=str(tmp_path / "dataset"), max_attempts_per_sample=3)
    saved = session.enroll(label="positive", count=1, seconds=5.0)

    assert len(saved) == 1


def test_gives_up_after_max_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(enrollment_module, "check_audio", lambda path: (False, "too quiet"))
    monkeypatch.setattr(enrollment_module, "DuplicateDetector", lambda: _FakeDupeDetector())

    session = EnrollmentSession(dataset_root=str(tmp_path / "dataset"), max_attempts_per_sample=2)
    saved = session.enroll(label="positive", count=1, seconds=5.0)

    assert saved == []  # never accepted, session moved on


def test_returns_a_path_for_every_accepted_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(enrollment_module, "check_audio", lambda path: (True, "ok"))
    monkeypatch.setattr(enrollment_module, "DuplicateDetector", lambda: _FakeDupeDetector())

    session = EnrollmentSession(dataset_root=str(tmp_path / "dataset"))
    saved = session.enroll(label="negative", count=3, seconds=5.0)

    assert len(saved) == 3
    assert len(set(saved)) == 3  # all distinct paths, no overwrites
