"""
enrollment.py
Ties recorder + validator + duplicate_detector + dataset_manager
together into one guided session:

    record clip -> validate -> check duplicate -> save to dataset

Retries a rejected clip instead of silently skipping it, so the user
always ends up with the number of samples they asked for.
"""

import os
import tempfile

from .recorder import record_one
from .validator import check_audio
from .duplicate_detector import DuplicateDetector
from .dataset_manager import DatasetManager
from . import config


class EnrollmentSession:
    def __init__(self, dataset_root: str = "dataset", max_attempts_per_sample: int = 3):
        self.dataset = DatasetManager(dataset_root)
        self.dupe_detector = DuplicateDetector()
        self.max_attempts_per_sample = max_attempts_per_sample

    def enroll(self, label: str, count: int = 10, seconds: float = 5.0) -> list[str]:
        """
        Records and accepts `count` valid, non-duplicate clips for
        `label` ("positive" or "negative"). Returns the saved paths.
        """
        saved_paths = []

        for i in range(count):
            accepted = False

            for attempt in range(1, self.max_attempts_per_sample + 1):
                print(f"\nSample {i + 1}/{count} (attempt {attempt}) - label: {label}")
                audio = record_one(seconds, config.SAMPLE_RATE)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                from .audio_utils import save_audio
                save_audio(audio, tmp_path, config.SAMPLE_RATE)

                is_valid, reason = check_audio(tmp_path)
                if not is_valid:
                    print(f"Rejected: {reason}. Try again.")
                    os.remove(tmp_path)
                    continue

                existing = self.dataset.list_samples(label) + saved_paths
                dupe = self.dupe_detector.find_duplicate(tmp_path, existing)
                if dupe:
                    print(f"Rejected: too similar to {os.path.basename(dupe)}. Try again.")
                    os.remove(tmp_path)
                    continue

                saved_path = self.dataset.add_sample(tmp_path, label)
                os.remove(tmp_path)
                saved_paths.append(saved_path)
                print(f"Saved as {saved_path}")
                accepted = True
                break

            if not accepted:
                print(f"Giving up on sample {i + 1} after {self.max_attempts_per_sample} attempts.")

        return saved_paths
