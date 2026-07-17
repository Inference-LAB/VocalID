"""
auth_adapter.py
Adapts VocalID's verification result into an authentication decision.
Deliberately has no ML in it - it just calls VoiceVerifier and
translates (verified, score) into an AuthenticationResult that
application code (door lock, login screen, attendance app, etc.)
can branch on.
"""

from dataclasses import dataclass
from .verifier import VoiceVerifier
from . import config


@dataclass
class AuthenticationResult:
    success: bool
    confidence: float

    def __str__(self):
        status = "ACCESS GRANTED" if self.success else "ACCESS DENIED"
        return f"{status} (confidence: {self.confidence:.2f})"


class VoiceAuthenticator:
    def __init__(self, model_path: str):
        self.verifier = VoiceVerifier(model_path)

    def authenticate_file(self, audio_path: str) -> AuthenticationResult:
        verified, score = self.verifier.verify_file(audio_path)
        return AuthenticationResult(success=verified, confidence=score)

    def authenticate_live(self, seconds: float = None) -> AuthenticationResult:
        from .recorder import record_one

        seconds = seconds or 4.0

        audio = record_one(
        seconds,
        config.SAMPLE_RATE
    )

        verified, score = self.verifier.verify_array(audio)

        return AuthenticationResult(
        success=verified,
        confidence=score
    )