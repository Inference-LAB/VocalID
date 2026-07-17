import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from .embeddings import EmbeddingExtractor
from .model_store import save_model, load_model


def normalize(emb):
    emb = np.asarray(emb).squeeze()
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


class VoiceTrainer:
    def __init__(self):
        self.extractor = EmbeddingExtractor()
        self.model = None


    def train(self, positive_paths, negative_paths, save_path="voice_auth.pkl"):
        X, y = [], []

        # Collect embeddings
        for p in positive_paths:
            emb = self.extractor.embed_file(p)
            if emb is None:
                continue
            X.append(normalize(emb))
            y.append(1)

        for p in negative_paths:
            print(f"Loading {p}")
            emb = self.extractor.embed_file(p)
            print(type(emb), None if emb is None else emb.shape)
            if emb is None:
                continue
            X.append(normalize(emb))
            y.append(0)

        if len(X) == 0:
            raise ValueError("No embeddings found to train on.")

        X = np.stack(X)
        y = np.array(y)

        self.model = LogisticRegression(max_iter=2000)
        self.model.fit(X, y)

        save_model(self.model, save_path)
        print(f"Model saved at: {save_path}")

        return save_path


    def evaluate(self, positive_files, negative_files):
        if self.model is None:
            raise ValueError("Model is not trained. Train before evaluating.")

        pos_embeds = []
        for f in positive_files:
            emb = self.extractor.embed_file(f)
            if emb is not None:
                pos_embeds.append(normalize(emb))

        neg_embeds = []
        for f in negative_files:
            emb = self.extractor.embed_file(f)
            if emb is not None:
                neg_embeds.append(normalize(emb))

        print(f"Loaded {len(pos_embeds)} positive embeddings")
        print(f"Loaded {len(neg_embeds)} negative embeddings")
        print("Evaluating...")

        if len(pos_embeds) == 0 or len(neg_embeds) == 0:
            raise ValueError("Not enough embeddings for evaluation.")

        X = np.vstack(pos_embeds + neg_embeds)
        y = np.array([1] * len(pos_embeds) + [0] * len(neg_embeds))

        preds = self.model.predict(X)

        acc = accuracy_score(y, preds)
        report = classification_report(y, preds)

        return {
            "accuracy": acc,
            "report": report
        }


    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)
        if self.model is None:
            raise ValueError(f"Failed to load model from {path}")