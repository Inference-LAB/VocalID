"""
dataset_manager.py
Keeps the on-disk dataset in the layout trainer.py expects:

    dataset/
        my_voice/       (positive samples)
        other_voices/   (negative samples)

Just handles copying accepted files in with sequential, collision-free
names - no metadata, no database.
"""
# src/vocalid/typing_compat.py

from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional,
    Union,
    Any,
    Callable,
)
import os
import shutil

POSITIVE_DIR = "my_voice"
NEGATIVE_DIR = "other_voices"


class DatasetManager:
    def __init__(self, root: str = "dataset"):
        self.root = root
        self.positive_dir = os.path.join(root, POSITIVE_DIR)
        self.negative_dir = os.path.join(root, NEGATIVE_DIR)
        os.makedirs(self.positive_dir, exist_ok=True)
        os.makedirs(self.negative_dir, exist_ok=True)

    def _target_dir(self, label: str) -> str:
        if label not in ("positive", "negative"):
            raise ValueError("label must be 'positive' or 'negative'")
        return self.positive_dir if label == "positive" else self.negative_dir

    def list_samples(self, label: str) -> List[str]:
        target_dir = self._target_dir(label)
        return [
            os.path.join(target_dir, f)
            for f in sorted(os.listdir(target_dir))
            if f.lower().endswith(".wav")
        ]

    def add_sample(self, src_path: str, label: str) -> str:
        """Copies src_path into the dataset with the next free index. Returns the new path."""
        target_dir = self._target_dir(label)
        existing = self.list_samples(label)
        next_index = len(existing) + 1
        dest_path = os.path.join(target_dir, f"sample{next_index:03d}.wav")
        shutil.copy2(src_path, dest_path)
        return dest_path
