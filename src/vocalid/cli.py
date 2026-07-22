

import argparse
import logging
import os
from glob import glob

from .audio_utils import record_audio
from .trainer import VoiceTrainer
from .verifier import VoiceVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Voice Verifier CLI")
    sub = parser.add_subparsers(dest="commands")

    # training command
    train = sub.add_parser("train", help="Train a voice authentication model")
    train.add_argument("--positive", required=True, help="Folder with your voice samples")
    train.add_argument("--negative", required=True, help="Folder with other voices")
    train.add_argument("--output", default="voice_auth.pkl", help="Path to save model")

    # evaluate the model
    evaluate = sub.add_parser("evaluate", help="Evaluate the trained model")
    evaluate.add_argument("--model", required=True, help="Path to trained model")
    evaluate.add_argument("--positive", required=True, help="Folder with your voice samples")
    evaluate.add_argument("--negative", required=True, help="Folder with negative/random samples")

    # verify file command
    verify = sub.add_parser("verify", help="Verify a voice file")
    verify.add_argument("file", help="Path to .wav voice file to verify")
    verify.add_argument("--model", default="voice_auth.pkl", help="Trained model path")

    # live verification command
    live = sub.add_parser("live", help="Live microphone verification")
    live.add_argument("--model", default="voice_auth.pkl", help="Trained model path")
    live.add_argument("--seconds", type=int, default=4, help="Recording duration in seconds")

    args = parser.parse_args()

    if args.commands == "train":
        pos_files = glob(os.path.join(args.positive, "*.wav"))
        neg_files = glob(os.path.join(args.negative, "*.wav"))

        trainer = VoiceTrainer()
        trainer.train(pos_files, neg_files, args.output)

        logger.info("Model saved to %s", args.output)

    elif args.commands == "evaluate":
        pos_files = glob(os.path.join(args.positive, "*.wav"))
        neg_files = glob(os.path.join(args.negative, "*.wav"))

        trainer = VoiceTrainer()
        trainer.load(args.model)

        results = trainer.evaluate(pos_files, neg_files)

        logger.info("")
        logger.info("===== Evaluation Results =====")
        logger.info("")
        logger.info("Accuracy: %.4f", results["accuracy"])
        logger.info("")
        logger.info("Classification Report:")
        logger.info("%s", results["report"])

    elif args.commands == "verify":
        verifier = VoiceVerifier(args.model)

        ok, score = verifier.verify_file(args.file)

        logger.info("Verified: %s, Score: %.2f", ok, score)

    elif args.commands == "live":
        verifier = VoiceVerifier(args.model)

        try:
            audio_tensor = record_audio(args.seconds)

        except RuntimeError as e:
            logger.error("%s", e)
            return

        ok, score = verifier.verify_array(audio_tensor)

        logger.info("Verified: %s, Score: %.2f", ok, score)

    else:
        parser.print_help()