"""
recorder.py

Records audio clips for enrollment.
"""


import os

from .audio_utils import record_audio, save_audio
from . import config



def record_one(
    seconds: float = 5.0,
    sample_rate: int = None
):

    """
    Record one audio clip.
    """

    sample_rate = (
        sample_rate
        or config.SAMPLE_RATE
    )


    print(
        f"Recording for {seconds:.1f}s... speak naturally."
    )


    audio = record_audio(
        duration=seconds,
        sample_rate=sample_rate
    )


    print(
        "Done."
    )


    return audio




def record_batch(
    out_dir: str,
    count: int = 10,
    seconds: float = 5.0,
    sample_rate: int = None,
    prefix: str = "sample"
):

    """
    Record multiple clips and save them.

    Returns:
        list of saved paths
    """


    os.makedirs(
        out_dir,
        exist_ok=True
    )


    sample_rate = (
        sample_rate
        or config.SAMPLE_RATE
    )


    paths = []


    for i in range(count):

        input(
            f"[{i+1}/{count}] Press Enter, then start speaking..."
        )


        audio = record_one(
            seconds,
            sample_rate
        )


        path = os.path.join(
            out_dir,
            f"{prefix}_{i+1:03d}.wav"
        )


        save_audio(
            audio,
            path,
            sample_rate
        )


        paths.append(
            path
        )


    return paths