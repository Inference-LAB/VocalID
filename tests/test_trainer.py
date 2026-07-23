

import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from vocalid.trainer import (
    VoiceTrainer,
    normalize,
)


# ---------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------

def test_normalize_vector():

    vec = np.array([3.0, 4.0])

    result = normalize(vec)

    assert np.allclose(
        result,
        np.array([0.6, 0.8]),
    )


def test_normalize_zero_vector():

    vec = np.zeros(5)

    result = normalize(vec)

    assert np.array_equal(result, vec)


# ---------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------

def make_trainer():

    trainer = VoiceTrainer.__new__(VoiceTrainer)

    trainer.extractor = MagicMock()

    trainer.model = None

    return trainer


# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------

@patch("vocalid.trainer.save_model")
@patch("vocalid.trainer.LogisticRegression")
@patch("vocalid.trainer.EmbeddingExtractor")
def test_training_pipeline(
    mock_extractor,
    mock_lr,
    mock_save,
):

    extractor = MagicMock()

    extractor.embed_file.side_effect = (
        lambda _: np.random.rand(192)
    )

    mock_extractor.return_value = extractor

    model = MagicMock()

    mock_lr.return_value = model

    trainer = VoiceTrainer()

    positive = [
        "p1.wav",
        "p2.wav",
    ]

    negative = [
        "n1.wav",
        "n2.wav",
    ]

    path = trainer.train(
        positive,
        negative,
        "dummy.pkl",
    )

    assert path == "dummy.pkl"

    model.fit.assert_called_once()

    mock_save.assert_called_once()

    assert trainer.model == model


@patch("vocalid.trainer.save_model")
@patch("vocalid.trainer.LogisticRegression")
@patch("vocalid.trainer.EmbeddingExtractor")
def test_train_skips_none_embeddings(
    mock_extractor,
    mock_lr,
    mock_save,
):

    extractor = MagicMock()

    extractor.embed_file.side_effect = [
        np.random.rand(192),
        None,
        np.random.rand(192),
    ]

    mock_extractor.return_value = extractor

    mock_lr.return_value = MagicMock()

    trainer = VoiceTrainer()

    trainer.train(
        ["a.wav", "b.wav"],
        ["c.wav"],
    )

    trainer.model.fit.assert_called_once()


@patch("vocalid.trainer.EmbeddingExtractor")
def test_train_no_embeddings(
    mock_extractor,
):

    extractor = MagicMock()

    extractor.embed_file.return_value = None

    mock_extractor.return_value = extractor

    trainer = VoiceTrainer()

    with pytest.raises(ValueError):

        trainer.train(
            ["a.wav"],
            ["b.wav"],
        )


# ---------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------

def test_evaluate_success():

    trainer = make_trainer()

    trainer.model = MagicMock()

    trainer.model.predict.return_value = np.array([1, 0])

    trainer.extractor.embed_file.side_effect = [
        np.random.rand(192),
        np.random.rand(192),
    ]

    result = trainer.evaluate(
        ["p.wav"],
        ["n.wav"],
    )

    assert "accuracy" in result

    assert "report" in result


def test_evaluate_without_model():

    trainer = make_trainer()

    with pytest.raises(ValueError):

        trainer.evaluate(
            ["p.wav"],
            ["n.wav"],
        )


def test_evaluate_empty_embeddings():

    trainer = make_trainer()

    trainer.model = MagicMock()

    trainer.extractor.embed_file.return_value = None

    with pytest.raises(ValueError):

        trainer.evaluate(
            ["p.wav"],
            ["n.wav"],
        )


# ---------------------------------------------------------------------
# save
# ---------------------------------------------------------------------

@patch("vocalid.trainer.save_model")
def test_save_success(mock_save):

    trainer = make_trainer()

    trainer.model = MagicMock()

    trainer.save("abc.pkl")

    mock_save.assert_called_once_with(
        trainer.model,
        "abc.pkl",
    )


def test_save_without_model():

    trainer = make_trainer()

    with pytest.raises(ValueError):

        trainer.save("abc.pkl")


# ---------------------------------------------------------------------
# load
# ---------------------------------------------------------------------

@patch("vocalid.trainer.load_model")
def test_load_success(mock_load):

    trainer = make_trainer()

    model = MagicMock()

    mock_load.return_value = model

    trainer.load("abc.pkl")

    assert trainer.model == model


@patch("vocalid.trainer.load_model")
def test_load_failure(mock_load):

    trainer = make_trainer()

    mock_load.return_value = None

    with pytest.raises(ValueError):

        trainer.load("abc.pkl")