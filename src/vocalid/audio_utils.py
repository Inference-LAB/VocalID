import torch
import torchaudio
import sounddevice as sd
import numpy as np
from .config import SAMPLE_RATE

import soundfile as sf
import torch

def load_audio(path,target_sr=SAMPLE_RATE):
    samples, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    waveform = torch.from_numpy(samples.T)  # -> (channels, frames), matches torchaudio's shape
    return waveform, sr
    # Convert to float32
    waveform = waveform.float()

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Mono
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ensure shape is (1, T)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Pad short audio
    min_len = target_sr  # 1 second minimum
    if waveform.shape[1] < min_len:
        pad_len = min_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform

def record_audio(seconds=4, fs=SAMPLE_RATE):
    print(f"Recording {seconds} seconds...")

    # Check if sounddevice sees any input devices
    try:
        devices = sd.query_devices()
        has_input = any(d["max_input_channels"] > 0 for d in devices)
        if not has_input:
            raise RuntimeError("No input microphone found. Live recording cannot run here.")
    except Exception:
        raise RuntimeError("No audio interface available. Live recording unsupported.")

    # Try to record
    try:
        audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        raise RuntimeError("Live recording failed. Likely running in Colab.") from e

    print("Recording complete")

    audio_tensor = torch.tensor(audio.T, dtype=torch.float32)

    # Ensure minimum length
    if audio_tensor.shape[1] < fs:
        pad_len = fs - audio_tensor.shape[1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len))

    return audio_tensor

def save_audio(audio_tensor, path, sample_rate=SAMPLE_RATE):
    """
    Save a mono waveform tensor to a WAV file.

    Parameters
    ----------
    audio_tensor : torch.Tensor
        Shape (1, T) or (T,)
    path : str
        Output WAV filename.
    sample_rate : int
        Sample rate.
    """
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    torchaudio.save(path, audio_tensor.cpu(), sample_rate)