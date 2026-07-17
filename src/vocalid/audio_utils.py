import torch
import torchaudio
import sounddevice as sd
import soundfile as sf

from .config import SAMPLE_RATE


def load_audio(path, target_sr=SAMPLE_RATE):
    """
    Load a WAV file and return a mono waveform tensor.

    Parameters
    ----------
    path : str
        Path to the WAV file.
    target_sr : int
        Desired sample rate.

    Returns
    -------
    tuple(torch.Tensor, int)
        Waveform with shape (1, T) and sample rate.
    """

    samples, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True
    )

    waveform = torch.from_numpy(samples.T).float()

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            target_sr
        )
        sr = target_sr

    # Ensure shape (1, T)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Pad to at least one second
    if waveform.shape[1] < target_sr:
        pad_len = target_sr - waveform.shape[1]
        waveform = torch.nn.functional.pad(
            waveform,
            (0, pad_len)
        )

    return waveform,sr


def record_audio(
    duration=4,
    sample_rate=SAMPLE_RATE
):
    """
    Record audio from the microphone.

    Parameters
    ----------
    duration : float
        Recording duration in seconds.
    sample_rate : int
        Recording sample rate.

    Returns
    -------
    torch.Tensor
        Recorded waveform with shape (1, T).
    """

    print(f"Recording {duration:.1f} seconds...")

    try:
        devices = sd.query_devices()

        if not any(device["max_input_channels"] > 0 for device in devices):
            raise RuntimeError("No microphone found.")

    except Exception as e:
        raise RuntimeError(
            "Unable to access an input microphone."
        ) from e

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )

        sd.wait()

    except Exception as e:
        raise RuntimeError(
            "Audio recording failed."
        ) from e

    print("Recording complete")

    waveform = torch.from_numpy(audio.T).float()

    # Ensure minimum duration of one second
    if waveform.shape[1] < sample_rate:
        pad_len = sample_rate - waveform.shape[1]
        waveform = torch.nn.functional.pad(
            waveform,
            (0, pad_len)
        )

    return waveform


def save_audio(
    waveform,
    path,
    sample_rate=SAMPLE_RATE
):
    """
    Save a waveform tensor as a WAV file.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio waveform of shape (1, T) or (T,).
    path : str
        Output WAV file path.
    sample_rate : int
        Sample rate.
    """

    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    sf.write(
        path,
        waveform.cpu().numpy(),
        sample_rate
    )