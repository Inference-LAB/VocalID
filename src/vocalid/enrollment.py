"""
enrollment.py

Ties recorder + validator + dataset_manager
together into one guided session:

    record clip -> validate -> save to dataset

Retries a rejected clip instead of silently skipping it.
"""

import os
import tempfile

from .recorder import record_one
from .validator import check_audio
from .dataset_manager import DatasetManager
from . import config


class EnrollmentSession:

    def __init__(
        self,
        dataset_root: str = "dataset",
        max_attempts_per_sample: int = 3
    ):

        self.dataset = DatasetManager(dataset_root)
        self.max_attempts_per_sample = max_attempts_per_sample

    def enroll(
        self,
        label: str,
        count: int = 10,
        seconds: float = 5.0
    ) -> list[str]:

        """
        Records and accepts `count` valid clips for
        `label` ("positive" or "negative").

        Returns saved file paths.

        Label options:

            positive:
                Target/enrolled speaker voice samples

            negative:
                Other speaker voice samples

        Flow:

            record clip
                |
            validate audio quality
                |
            save accepted clip
        """

        # Validate label before doing anything
        if label not in ("positive", "negative"):
            raise ValueError(
                "label must be 'positive' or 'negative'"
            )

        print("""
==================================================
        VocalID Enrollment Instructions
==================================================

Label options:

    positive -> Target/enrolled speaker voice samples

    negative -> Other speaker voice samples


Enrollment flow:

    record clip
        |
    validate audio quality
        |
    save accepted clip


Recording guidelines:

    - Duration: 4-6.5 seconds
    - Speak naturally
    - Avoid silence
    - Avoid background noise

==================================================
""")

        saved_paths = []

        for i in range(count):

            accepted = False

            for attempt in range(
                1,
                self.max_attempts_per_sample + 1
            ):

                print(
                    f"\nSample {i + 1}/{count} "
                    f"(attempt {attempt}) "
                    f"- label: {label}"
                )

                audio = record_one(
                    seconds,
                    config.SAMPLE_RATE
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False
                ) as tmp:

                    tmp_path = tmp.name

                from .audio_utils import save_audio

                save_audio(
                    audio,
                    tmp_path,
                    config.SAMPLE_RATE
                )

                # Validate audio
                is_valid, reason = check_audio(
                    tmp_path
                )

                if not is_valid:

                    print(
                        f"Rejected: {reason}. Try again."
                    )

                    os.remove(tmp_path)
                    continue

                # Save accepted sample
                saved_path = self.dataset.add_sample(
                    tmp_path,
                    label
                )

                os.remove(tmp_path)

                saved_paths.append(
                    saved_path
                )

                print(
                    f"Saved as {saved_path}"
                )

                accepted = True
                break

            if not accepted:

                print(
                    f"Giving up on sample {i + 1} "
                    f"after {self.max_attempts_per_sample} attempts."
                )

        return saved_paths