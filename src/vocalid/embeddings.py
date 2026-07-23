

import logging

import numpy as np
import torch

from .audio_utils import load_audio

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    def __init__(self, model_path="speechbrain/spkrec-ecapa-voxceleb"):
        # Lazy torchaudio compatibility patch
        try:
            import torchaudio

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["sox_io"]

        except Exception:
            # Older/newer torchaudio versions may not expose this API.
            # Safe to continue because it is only a compatibility patch.
            pass

        try:
            from speechbrain.inference import EncoderClassifier

        except Exception as e:
            raise ImportError(
                "SpeechBrain could not load.\n"
                f"Original error: {e}"
            ) from e

        self.model = EncoderClassifier.from_hparams(
            source=model_path,
            run_opts={"device": "cpu"},
            savedir="pretrained_models/ecapa",
        )

    def _prepare_waveform(self, waveform):
        """
        Convert waveform into the format expected by SpeechBrain.

        Expected shape:
            (1, T)
        """

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(
                waveform,
                dtype=torch.float32
            )

        # Convert (T,) -> (1, T)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Stereo -> Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Pad to at least 1 second
        min_len = 16000

        if waveform.shape[1] < min_len:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, min_len - waveform.shape[1])
            )

        return waveform

    def _normalize(self, embedding):
        embedding = np.asarray(embedding).squeeze()

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm

    def embed_file(self, path):
        """
        Load a WAV file and return a normalized speaker embedding.
        """

        try:
            waveform, _ = load_audio(path)
            waveform = self._prepare_waveform(waveform)

            with torch.no_grad():
                embedding = (
                    self.model
                    .encode_batch(waveform)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            return self._normalize(embedding)

        except Exception:
            logger.exception(
                "Embedding extraction failed for file: %s",
                path,
            )
            raise

    def emb_waveform(self, waveform):
        """
        Create an embedding directly from a waveform tensor.
        """

        try:
            waveform = self._prepare_waveform(waveform)

            with torch.no_grad():
                embedding = (
                    self.model
                    .encode_batch(waveform)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            return self._normalize(embedding)

        except Exception:
            logger.exception(
                "Embedding extraction failed from waveform."
            )
            raise

    def extract(self, waveform):
        """
        Compatibility wrapper.
        """
        return self.emb_waveform(waveform)