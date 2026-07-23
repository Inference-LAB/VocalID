import numpy as np
import pytest
import torch

from unittest.mock import MagicMock, patch

from vocalid.embeddings import EmbeddingExtractor


def make_extractor():
    """
    Create an EmbeddingExtractor without loading SpeechBrain.
    """
    extractor = EmbeddingExtractor.__new__(EmbeddingExtractor)
    extractor.model = MagicMock()
    return extractor


# ---------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------

def test_embedding_extractor_init():

    mock_encoder = MagicMock()
    mock_model = MagicMock()

    mock_encoder.from_hparams.return_value = mock_model

    fake_speechbrain = MagicMock()
    fake_speechbrain.inference.EncoderClassifier = mock_encoder

    with patch.dict(
        "sys.modules",
        {
            "speechbrain": fake_speechbrain,
            "speechbrain.inference": fake_speechbrain.inference,
        },
    ):

        extractor = EmbeddingExtractor()

        assert extractor.model == mock_model

        mock_encoder.from_hparams.assert_called_once_with(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir="pretrained_models/ecapa",
        )


def test_embedding_extractor_import_error():

    with patch.dict(
        "sys.modules",
        {
            "speechbrain": None,
            "speechbrain.inference": None,
        },
    ):

        with pytest.raises(ImportError) as exc:

            EmbeddingExtractor()

        assert "SpeechBrain could not load" in str(exc.value)


# ---------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------

def test_normalize_vector():

    extractor = make_extractor()

    vec = np.array([3.0, 4.0])

    result = extractor._normalize(vec)

    assert np.allclose(
        result,
        np.array([0.6, 0.8])
    )


def test_normalize_zero_vector():

    extractor = make_extractor()

    vec = np.zeros(5)

    result = extractor._normalize(vec)

    assert np.array_equal(
        result,
        vec
    )


# ---------------------------------------------------------------------
# _prepare_waveform
# ---------------------------------------------------------------------

def test_prepare_waveform_numpy():

    extractor = make_extractor()

    waveform = np.random.rand(
        16000
    ).astype(np.float32)

    result = extractor._prepare_waveform(
        waveform
    )

    assert isinstance(
        result,
        torch.Tensor
    )

    assert result.shape == (
        1,
        16000
    )


def test_prepare_waveform_stereo():

    extractor = make_extractor()

    waveform = torch.rand(
        2,
        16000
    )

    result = extractor._prepare_waveform(
        waveform
    )

    assert result.shape == (
        1,
        16000
    )


def test_prepare_waveform_padding():

    extractor = make_extractor()

    waveform = np.random.rand(
        8000
    ).astype(np.float32)

    result = extractor._prepare_waveform(
        waveform
    )

    assert result.shape == (
        1,
        16000
    )


# ---------------------------------------------------------------------
# embed_file
# ---------------------------------------------------------------------

@patch("vocalid.embeddings.load_audio")
def test_embed_file_success(mock_load_audio):

    extractor = make_extractor()

    waveform = torch.rand(
        1,
        16000
    )

    mock_load_audio.return_value = (
        waveform,
        16000
    )

    extractor.model.encode_batch.return_value = torch.rand(
        1,
        192
    )

    result = extractor.embed_file(
        "dummy.wav"
    )

    assert isinstance(
        result,
        np.ndarray
    )

    assert result.shape == (
        192,
    )

    mock_load_audio.assert_called_once_with(
        "dummy.wav"
    )


@patch("vocalid.embeddings.load_audio")
def test_embed_file_exception(mock_load_audio):

    extractor = make_extractor()

    mock_load_audio.side_effect = RuntimeError(
        "Load failed"
    )

    with pytest.raises(RuntimeError):

        extractor.embed_file(
            "dummy.wav"
        )


# ---------------------------------------------------------------------
# emb_waveform
# ---------------------------------------------------------------------

def test_emb_waveform_success():

    extractor = make_extractor()

    waveform = torch.rand(
        1,
        16000
    )

    extractor.model.encode_batch.return_value = torch.rand(
        1,
        192
    )

    result = extractor.emb_waveform(
        waveform
    )

    assert isinstance(
        result,
        np.ndarray
    )

    assert result.shape == (
        192,
    )


def test_emb_waveform_exception():

    extractor = make_extractor()

    waveform = torch.rand(
        1,
        16000
    )

    extractor.model.encode_batch.side_effect = RuntimeError(
        "Encoding failed"
    )

    with pytest.raises(RuntimeError):

        extractor.emb_waveform(
            waveform
        )


# ---------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------

def test_extract_calls_emb_waveform():

    extractor = make_extractor()

    waveform = torch.rand(
        1,
        16000
    )

    expected = np.random.rand(
        192
    )

    with patch.object(
        extractor,
        "emb_waveform",
        return_value=expected,
    ) as mock_method:

        result = extractor.extract(
            waveform
        )

        mock_method.assert_called_once_with(
            waveform
        )

        assert np.array_equal(
            result,
            expected
        )