"""
conftest.py
Shared fixtures for the new modules' tests. Everything here writes
real, tiny .wav files with numpy + soundfile so tests run anywhere
(CI included) without a microphone or a real voice sample.
"""

import numpy as np
import soundfile as sf
import pytest

SAMPLE_RATE = 16000


def _tone(duration: float, sample_rate: int, freq: float = 220.0, amplitude: float = 0.3):
    """A simple sine wave - stands in for a 'real' voice-shaped signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def write_wav(tmp_path):
    """
    Returns a function write_wav(name, **kwargs) -> path
    that writes a synthetic clip and returns its path.

    kwargs:
        duration   seconds (default 5.0)
        sample_rate (default 16000)
        freq       tone frequency, changes the clip's "identity" (default 220.0)
        amplitude  0..1 (default 0.3)
        silent     if True, writes near-zero samples instead of a tone
        clipped    if True, writes a hard-clipped square-ish wave
    """
    def _write(name: str, duration: float = 5.0, sample_rate: int = SAMPLE_RATE,
               freq: float = 220.0, amplitude: float = 0.3,
               silent: bool = False, clipped: bool = False):
        if silent:
            samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
        elif clipped:
            samples = _tone(duration, sample_rate, freq, amplitude=1.0)
            samples = np.clip(samples * 5.0, -1.0, 1.0)
        else:
            samples = _tone(duration, sample_rate, freq, amplitude)

        path = tmp_path / name
        sf.write(str(path), samples, sample_rate)
        return str(path)

    return _write
