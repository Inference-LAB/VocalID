from unittest.mock import MagicMock, patch

from vocalid.model_store import save_model, load_model


# ---------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------

@patch("vocalid.model_store.joblib.dump")
def test_save_model(mock_dump):

    model = MagicMock()

    save_model(
        model,
        "dummy.pkl",
    )

    mock_dump.assert_called_once_with(
        model,
        "dummy.pkl",
    )


# ---------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------

@patch("vocalid.model_store.joblib.load")
def test_load_model(mock_load):

    fake_model = MagicMock()

    mock_load.return_value = fake_model

    model = load_model(
        "dummy.pkl",
    )

    mock_load.assert_called_once_with(
        "dummy.pkl",
    )

    assert model == fake_model