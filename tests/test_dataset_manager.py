"""
test_dataset_manager.py
"""

import os
import pytest
from vocalid.dataset_manager import DatasetManager


def test_creates_folder_layout(tmp_path):
    root = tmp_path / "dataset"
    DatasetManager(str(root))
    assert (root / "my_voice").is_dir()
    assert (root / "other_voices").is_dir()


def test_add_sample_copies_and_lists(tmp_path, write_wav):
    dm = DatasetManager(str(tmp_path / "dataset"))
    clip = write_wav("clip.wav")

    saved_path = dm.add_sample(clip, "positive")

    assert os.path.exists(saved_path)
    assert saved_path in dm.list_samples("positive")
    assert dm.list_samples("negative") == []


def test_add_sample_sequential_naming(tmp_path, write_wav):
    dm = DatasetManager(str(tmp_path / "dataset"))
    clip_a = write_wav("a.wav")
    clip_b = write_wav("b.wav")

    path1 = dm.add_sample(clip_a, "negative")
    path2 = dm.add_sample(clip_b, "negative")

    assert os.path.basename(path1) == "sample001.wav"
    assert os.path.basename(path2) == "sample002.wav"
    assert len(dm.list_samples("negative")) == 2


def test_rejects_bad_label(tmp_path, write_wav):
    dm = DatasetManager(str(tmp_path / "dataset"))
    clip = write_wav("clip.wav")

    with pytest.raises(ValueError):
        dm.add_sample(clip, "unknown_label")
