import numpy as np
import pytest
import torch

from unittest.mock import patch

from vocalid.audio_utils import (
    load_audio,
    record_audio,
    save_audio,
)


# ---------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------

@patch("vocalid.audio_utils.sf.read")
def test_load_audio_mono(mock_read):

    samples = np.random.rand(16000, 1).astype(np.float32)

    mock_read.return_value = (
        samples,
        16000,
    )

    waveform, sr = load_audio("dummy.wav")

    assert sr == 16000
    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (1, 16000)


@patch("vocalid.audio_utils.sf.read")
def test_load_audio_stereo(mock_read):

    samples = np.random.rand(16000, 2).astype(np.float32)

    mock_read.return_value = (
        samples,
        16000,
    )

    waveform, _ = load_audio("dummy.wav")

    assert waveform.shape == (1, 16000)


@patch("vocalid.audio_utils.torchaudio.functional.resample")
@patch("vocalid.audio_utils.sf.read")
def test_load_audio_resample(
    mock_read,
    mock_resample,
):

    samples = np.random.rand(8000, 1).astype(np.float32)

    mock_read.return_value = (
        samples,
        8000,
    )

    mock_resample.return_value = torch.rand(1, 16000)

    waveform, sr = load_audio("dummy.wav")

    assert sr == 16000
    mock_resample.assert_called_once()


@patch("vocalid.audio_utils.sf.read")
def test_load_audio_padding(mock_read):

    samples = np.random.rand(4000, 1).astype(np.float32)

    mock_read.return_value = (
        samples,
        16000,
    )

    waveform, _ = load_audio("dummy.wav")

    assert waveform.shape == (1, 16000)


@patch("vocalid.audio_utils.sf.read")
def test_load_audio_unsqueeze(mock_read):
    """
    Force execution of the waveform.ndim == 1 branch.
    """

    waveform = torch.rand(16000)

    with patch(
        "vocalid.audio_utils.torch.from_numpy",
        return_value=waveform,
    ):
        mock_read.return_value = (
            np.random.rand(16000, 1).astype(np.float32),
            16000,
        )

        result, _ = load_audio("dummy.wav")

        assert result.shape == (1, 16000)


# ---------------------------------------------------------------------
# record_audio
# ---------------------------------------------------------------------

@patch("vocalid.audio_utils.sd.wait")
@patch("vocalid.audio_utils.sd.rec")
@patch("vocalid.audio_utils.sd.query_devices")
def test_record_audio_success(
    mock_devices,
    mock_rec,
    mock_wait,
):

    mock_devices.return_value = [
        {"max_input_channels": 1}
    ]

    mock_rec.return_value = np.random.rand(
        16000,
        1,
    ).astype(np.float32)

    waveform = record_audio(duration=1)

    assert isinstance(waveform, torch.Tensor)
    assert waveform.shape == (1, 16000)

    mock_wait.assert_called_once()


@patch("vocalid.audio_utils.sd.query_devices")
def test_record_audio_no_microphone(
    mock_devices,
):

    mock_devices.return_value = [
        {"max_input_channels": 0}
    ]

    with pytest.raises(RuntimeError):
        record_audio()


@patch("vocalid.audio_utils.sd.rec")
@patch("vocalid.audio_utils.sd.query_devices")
def test_record_audio_recording_failure(
    mock_devices,
    mock_rec,
):

    mock_devices.return_value = [
        {"max_input_channels": 1}
    ]

    mock_rec.side_effect = Exception(
        "Recording failed"
    )

    with pytest.raises(RuntimeError):
        record_audio()


# ---------------------------------------------------------------------
# save_audio
# ---------------------------------------------------------------------

@patch("vocalid.audio_utils.sf.write")
def test_save_audio(mock_write):

    waveform = torch.rand(
        1,
        16000,
    )

    save_audio(
        waveform,
        "output.wav",
    )

    mock_write.assert_called_once()


@patch("vocalid.audio_utils.sf.write")
def test_save_audio_1d_waveform(mock_write):
    """
    Execute the branch where waveform.ndim != 2.
    """

    waveform = torch.rand(16000)

    save_audio(
        waveform,
        "output.wav",
    )

    mock_write.assert_called_once()