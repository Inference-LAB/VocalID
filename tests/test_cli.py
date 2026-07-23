 
import sys
from unittest.mock import MagicMock, patch

from vocalid.cli import main


# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------

@patch("vocalid.cli.VoiceTrainer")
def test_cli_train(mock_trainer):

    trainer = MagicMock()
    mock_trainer.return_value = trainer

    args = [
        "cli.py",
        "train",
        "--positive", "pos_folder",
        "--negative", "neg_folder",
    ]

    with patch.object(sys, "argv", args):
        main()

    trainer.train.assert_called_once()


# ---------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------

@patch("vocalid.cli.VoiceTrainer")
def test_cli_evaluate(mock_trainer):

    trainer = MagicMock()

    trainer.evaluate.return_value = {
        "accuracy": 0.95,
        "report": "classification report",
    }

    mock_trainer.return_value = trainer

    args = [
        "cli.py",
        "evaluate",
        "--model", "model.pkl",
        "--positive", "pos_folder",
        "--negative", "neg_folder",
    ]

    with patch.object(sys, "argv", args):
        main()

    trainer.load.assert_called_once_with("model.pkl")
    trainer.evaluate.assert_called_once()


# ---------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------

@patch("vocalid.cli.VoiceVerifier")
def test_cli_verify(mock_verifier):

    verifier = MagicMock()

    verifier.verify_file.return_value = (
        True,
        0.95,
    )

    mock_verifier.return_value = verifier

    args = [
        "cli.py",
        "verify",
        "dummy.wav",
    ]

    with patch.object(sys, "argv", args):
        main()

    verifier.verify_file.assert_called_once_with(
        "dummy.wav"
    )


# ---------------------------------------------------------------------
# live success
# ---------------------------------------------------------------------

@patch("vocalid.cli.record_audio")
@patch("vocalid.cli.VoiceVerifier")
def test_cli_live_success(
    mock_verifier,
    mock_record,
):

    verifier = MagicMock()

    verifier.verify_array.return_value = (
        True,
        0.91,
    )

    mock_verifier.return_value = verifier

    mock_record.return_value = MagicMock()

    args = [
        "cli.py",
        "live",
        "--seconds",
        "5",
    ]

    with patch.object(sys, "argv", args):
        main()

    mock_record.assert_called_once_with(5)

    verifier.verify_array.assert_called_once()


# ---------------------------------------------------------------------
# live microphone failure
# ---------------------------------------------------------------------

@patch("vocalid.cli.record_audio")
@patch("vocalid.cli.VoiceVerifier")
def test_cli_live_microphone_failure(
    mock_verifier,
    mock_record,
):

    mock_verifier.return_value = MagicMock()

    mock_record.side_effect = RuntimeError(
        "No microphone"
    )

    args = [
        "cli.py",
        "live",
    ]

    with patch.object(sys, "argv", args):
        main()


# ---------------------------------------------------------------------
# help
# ---------------------------------------------------------------------

@patch("argparse.ArgumentParser.print_help")
def test_cli_print_help(mock_help):

    args = [
        "cli.py",
    ]

    with patch.object(sys, "argv", args):
        main()

    mock_help.assert_called_once()