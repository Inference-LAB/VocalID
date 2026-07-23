"""
End-to-end demo:
  1. Enroll positive (your voice) and negative (other voices) samples
  2. Train a model on the collected dataset
  3. Authenticate live from the microphone

Run: python examples/enroll_and_authenticate.py
"""

import glob

from vocalid.enrollment import EnrollmentSession
from vocalid.trainer import VoiceTrainer
from vocalid.auth_adapter import VoiceAuthenticator

DATASET_ROOT = "dataset"
MODEL_PATH = "my_voice_model.pkl"


def main():
    session = EnrollmentSession(dataset_root=DATASET_ROOT)

    print("=== Enrolling YOUR voice (positive samples) ===")
    session.enroll(label="positive", count=10, seconds=5.0)

    print("\n=== Enrolling OTHER voices (negative samples) ===")
    print("Have a few different people speak for this part.")
    session.enroll(label="negative", count=10, seconds=5.0)

    print("\n=== Training ===")
    pos_files = glob.glob(f"{DATASET_ROOT}/my_voice/*.wav")
    neg_files = glob.glob(f"{DATASET_ROOT}/other_voices/*.wav")
    trainer = VoiceTrainer()
    trainer.train(pos_files, neg_files, save_path=MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("\n=== Authenticate ===")
    authenticator = VoiceAuthenticator(MODEL_PATH)
    result = authenticator.authenticate_live(seconds=4.0)
    print(result)


if __name__ == "__main__":
    main()
