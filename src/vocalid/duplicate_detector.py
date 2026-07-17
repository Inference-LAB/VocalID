"""
duplicate_detector.py
Flags a new clip as a duplicate if it's too similar (by embedding
cosine similarity) to a clip already in the dataset, or to another
clip recorded in the same enrollment batch. This is the same
embedding extractor the trainer/verifier already use, so no extra
model is loaded.
"""

import numpy as np
from .embeddings import EmbeddingExtractor
from .audio_utils import load_audio

DUPLICATE_THRESHOLD = 0.97  # cosine similarity above this = "same clip"


def _cosine(a, b) -> float:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


class DuplicateDetector:
    def __init__(self, threshold: float = DUPLICATE_THRESHOLD):
        self.extractor = EmbeddingExtractor()
        self.threshold = threshold

    def _embed_file(self, path: str):
        audio, sample_rate = load_audio(path)
        return self.extractor.extract(audio)

    def find_duplicate(self, new_path: str, existing_paths: list[str]):
        """
        Compares `new_path` against every clip in `existing_paths`.
        Returns the path of the first near-duplicate found, or None.
        """
        new_emb = self._embed_file(new_path)
        for path in existing_paths:
            sim = _cosine(new_emb, self._embed_file(path))
            if sim >= self.threshold:
                return path
        return None

    def is_duplicate(self, new_path: str, existing_paths: list[str]) -> bool:
        return self.find_duplicate(new_path, existing_paths) is not None
