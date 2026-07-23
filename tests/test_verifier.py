

import numpy as np
import torch

from unittest.mock import MagicMock, patch

from vocalid.verifier import VoiceVerifier
from vocalid.config import THRESHOLD


def make_verifier():
    """
    Create a VoiceVerifier without calling __init__.
    """
    verifier = VoiceVerifier.__new__(VoiceVerifier)
    verifier.model = MagicMock()
    verifier.extractor = MagicMock()
    return verifier


# ---------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------

@patch("vocalid.verifier.EmbeddingExtractor")
@patch("vocalid.verifier.load_model")
def test_verifier_init(
    mock_load_model,
    mock_extractor,
):
    fake_model = MagicMock()
    fake_extractor = MagicMock()

    mock_load_model.return_value = fake_model
    mock_extractor.return_value = fake_extractor

    verifier = VoiceVerifier("dummy_model.pkl")

    mock_load_model.assert_called_once_with(
        "dummy_model.pkl"
    )

    mock_extractor.assert_called_once()

    assert verifier.model is fake_model
    assert verifier.extractor is fake_extractor


# ---------------------------------------------------------------------
# verify_file
# ---------------------------------------------------------------------

@patch("vocalid.verifier.load_audio")
def test_verify_file_success(mock_load_audio):

    verifier = make_verifier()

    waveform = torch.rand(1, 16000)

    mock_load_audio.return_value = (
        waveform,
        16000,
    )

    verifier.extractor.emb_waveform.return_value = np.random.rand(192)

    verifier.model.predict_proba.return_value = [
        [0.1, 0.9]
    ]

    verified, score = verifier.verify_file(
        "dummy.wav"
    )

    assert verified is True
    assert score == 0.9

    verifier.extractor.emb_waveform.assert_called_once_with(
        waveform
    )


@patch("vocalid.verifier.load_audio")
def test_verify_file_threshold(mock_load_audio):

    verifier = make_verifier()

    waveform = torch.rand(1, 16000)

    mock_load_audio.return_value = (
        waveform,
        16000,
    )

    verifier.extractor.emb_waveform.return_value = np.random.rand(192)

    verifier.model.predict_proba.return_value = [
        [0.8, 0.2]
    ]

    verified, score = verifier.verify_file(
        "dummy.wav"
    )

    assert verified is False
    assert score == 0.2


# ---------------------------------------------------------------------
# verify_array
# ---------------------------------------------------------------------

def test_verify_array_numpy():

    verifier = make_verifier()

    verifier.extractor.emb_waveform.return_value = np.random.rand(192)

    verifier.model.predict_proba.return_value = [
        [0.2, 0.8]
    ]

    waveform = np.random.rand(
        16000
    ).astype(np.float32)

    verified, score = verifier.verify_array(
        waveform
    )

    assert verified is True
    assert score == 0.8

    verifier.extractor.emb_waveform.assert_called_once()


def test_verify_array_tensor():

    verifier = make_verifier()

    verifier.extractor.emb_waveform.return_value = np.random.rand(192)

    verifier.model.predict_proba.return_value = [
        [0.6, 0.4]
    ]

    waveform = torch.rand(
        1,
        16000,
    )

    verified, score = verifier.verify_array(
        waveform
    )

    assert verified is False
    assert score == 0.4

    verifier.extractor.emb_waveform.assert_called_once_with(
        waveform
    )


# ---------------------------------------------------------------------
# verify_live
# ---------------------------------------------------------------------

@patch("vocalid.verifier.record_audio")
def test_verify_live(mock_record):

    verifier = make_verifier()

    waveform = torch.rand(
        1,
        16000,
    )

    mock_record.return_value = waveform

    verifier.verify_array = MagicMock(
        return_value=(
            True,
            0.95,
        )
    )

    verified, score = verifier.verify_live(
        seconds=5
    )

    mock_record.assert_called_once_with(
        5
    )

    verifier.verify_array.assert_called_once_with(
        waveform,
        THRESHOLD,
    )

    assert verified is True
    assert score == 0.95