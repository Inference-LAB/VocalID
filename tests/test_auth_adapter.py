"""
test_auth_adapter.py

VoiceVerifier is mocked out so this tests the adapter's translation
of (verified, score) -> AuthenticationResult, not the real
verification model.
"""

import pytest
import vocalid.auth_adapter as auth_module
from vocalid.auth_adapter import VoiceAuthenticator, AuthenticationResult


class _FakeVerifier:
    def __init__(self, model_path):
        self.model_path = model_path

    def verify_file(self, path):
        return True, 0.87

    def verify_live(self, audio):
        return False, 0.21


@pytest.fixture
def authenticator(monkeypatch):
    monkeypatch.setattr(auth_module, "VoiceVerifier", _FakeVerifier)
    return VoiceAuthenticator("dummy_model.pkl")


def test_authenticate_file_returns_granted_result(authenticator):
    result = authenticator.authenticate_file("some_clip.wav")
    assert isinstance(result, AuthenticationResult)
    assert result.success is True
    assert result.confidence == 0.87


def test_authenticate_live_returns_denied_result(authenticator, monkeypatch):
    # authenticate_live does `from .recorder import record_one` internally,
    # so the real patch point is vocalid.recorder.record_one.
    monkeypatch.setattr("vocalid.recorder.record_one",
                         lambda seconds, sample_rate: "fake_audio")

    result = authenticator.authenticate_live(seconds=4.0)

    assert result.success is False
    assert result.confidence == 0.21


def test_authenticate_live_defaults_to_four_seconds(authenticator, monkeypatch):
    captured = {}

    def fake_record_one(seconds, sample_rate):
        captured["seconds"] = seconds
        return "fake_audio"

    monkeypatch.setattr("vocalid.recorder.record_one", fake_record_one)
    authenticator.authenticate_live(seconds=None)

    assert captured["seconds"] == 4.0


def test_result_str_formatting():
    granted = AuthenticationResult(success=True, confidence=0.9123)
    denied = AuthenticationResult(success=False, confidence=0.1)

    assert "ACCESS GRANTED" in str(granted)
    assert "0.91" in str(granted)
    assert "ACCESS DENIED" in str(denied)
