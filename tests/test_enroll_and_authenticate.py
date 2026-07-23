from unittest.mock import MagicMock, patch

from vocalid.enroll_and_authenticate import (
    main,
    DATASET_ROOT,
    MODEL_PATH,
)


@patch("vocalid.enroll_and_authenticate.print")
@patch("vocalid.enroll_and_authenticate.glob.glob")
@patch("vocalid.enroll_and_authenticate.VoiceAuthenticator")
@patch("vocalid.enroll_and_authenticate.VoiceTrainer")
@patch("vocalid.enroll_and_authenticate.EnrollmentSession")
def test_main(
    mock_session_cls,
    mock_trainer_cls,
    mock_auth_cls,
    mock_glob,
    mock_print,
):
    """
    Test the complete enroll -> train -> authenticate workflow.
    """

    # -----------------------------
    # Enrollment session
    # -----------------------------
    session = MagicMock()
    mock_session_cls.return_value = session

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = MagicMock()
    mock_trainer_cls.return_value = trainer

    # -----------------------------
    # Authenticator
    # -----------------------------
    authenticator = MagicMock()
    authenticator.authenticate_live.return_value = {
        "verified": True,
        "score": 0.96,
    }
    mock_auth_cls.return_value = authenticator

    # -----------------------------
    # Fake dataset files
    # -----------------------------
    mock_glob.side_effect = [
        [
            f"{DATASET_ROOT}/my_voice/a.wav",
            f"{DATASET_ROOT}/my_voice/b.wav",
        ],
        [
            f"{DATASET_ROOT}/other_voices/x.wav",
            f"{DATASET_ROOT}/other_voices/y.wav",
        ],
    ]

    # -----------------------------
    # Run
    # -----------------------------
    main()

    # -----------------------------
    # Enrollment
    # -----------------------------
    mock_session_cls.assert_called_once_with(
        dataset_root=DATASET_ROOT
    )

    session.enroll.assert_any_call(
        label="positive",
        count=10,
        seconds=5.0,
    )

    session.enroll.assert_any_call(
        label="negative",
        count=10,
        seconds=5.0,
    )

    assert session.enroll.call_count == 2

    # -----------------------------
    # Dataset lookup
    # -----------------------------
    assert mock_glob.call_count == 2

    # -----------------------------
    # Training
    # -----------------------------
    trainer.train.assert_called_once_with(
        [
            f"{DATASET_ROOT}/my_voice/a.wav",
            f"{DATASET_ROOT}/my_voice/b.wav",
        ],
        [
            f"{DATASET_ROOT}/other_voices/x.wav",
            f"{DATASET_ROOT}/other_voices/y.wav",
        ],
        save_path=MODEL_PATH,
    )

    # -----------------------------
    # Authentication
    # -----------------------------
    mock_auth_cls.assert_called_once_with(MODEL_PATH)

    authenticator.authenticate_live.assert_called_once_with(
        seconds=4.0
    )

    # -----------------------------
    # Output
    # -----------------------------
    assert mock_print.call_count >= 6