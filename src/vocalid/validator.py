"""
validator.py
Basic sanity checks on a recorded clip before it's allowed into the
dataset: right length, not silent, not clipped/overdriven.

Uses audio_utils.load_audio(path) -> (tensor, sample_rate), which the
library already needs for file-based verification.
"""
# src/vocalid/typing_compat.py

from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional,
    Union,
    Any,
    Callable,
)
import numpy as np
from .audio_utils import load_audio

MIN_SECONDS = 4.0
MAX_SECONDS = 6.5
MIN_RMS = 0.01      # below this, treat the clip as silence/noise floor
CLIP_THRESHOLD = 0.98  # fraction of full scale considered "clipping"
MAX_CLIPPED_RATIO = 0.01


def _to_numpy(audio):
    if hasattr(audio, "numpy"):
        return audio.numpy().flatten()
    return np.asarray(audio).flatten()


def check_audio(path: str) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason). reason is 'ok' when valid, otherwise
    a short human-readable explanation of why it was rejected.
    """
    audio, sample_rate = load_audio(path)
    samples = _to_numpy(audio)

    duration = len(samples) / float(sample_rate)
    if duration < MIN_SECONDS:
        return False, f"too short ({duration:.1f}s)"
    if duration > MAX_SECONDS:
        return False, f"too long ({duration:.1f}s)"

    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < MIN_RMS:
        return False, "too quiet / silence detected"

    clipped_ratio = float(np.mean(np.abs(samples) >= CLIP_THRESHOLD))
    if clipped_ratio > MAX_CLIPPED_RATIO:
        return False, "clipping / distortion detected"

    return True, "ok"
