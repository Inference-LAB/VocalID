from .embeddings import EmbeddingExtractor
from .model_store import load_model
import torch
from .audio_utils import record_audio, load_audio
from .config import THRESHOLD

class VoiceVerifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # sklearn classifier
        self.extractor = EmbeddingExtractor()

    def verify_file(self, file_path, threshold=THRESHOLD):
        wav,_ = load_audio(file_path)                   # waveform tensor
        emb = self.extractor.emb_waveform(wav)        # embedding as numpy
        score = self.model.predict_proba([emb])[0][1]
        verified = score >= threshold
        return verified, float(score)
    
    def verify_array(self, audio_tensor, threshold=THRESHOLD):
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor).unsqueeze(0)
        emb = self.extractor.emb_waveform(audio_tensor)
        score = self.model.predict_proba([emb])[0][1]
        verified = score >= threshold
        return verified, float(score)

    def verify_live(self, seconds=4, threshold=THRESHOLD):
        waveform = record_audio(seconds)              # waveform tensor from mic
        return self.verify_array(waveform, threshold)
