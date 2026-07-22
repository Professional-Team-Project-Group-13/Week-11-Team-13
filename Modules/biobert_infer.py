"""
 BioBERT inference adapter for explainability and the agent.

Loads fine-tuned BioBERT weights and exposes the small set of callables that
the rest of the system expects, so LIME, SHAP, FAISS, and the autonomous agent
all explain and route on the project's strongest model rather than a baseline.

Interfaces provided

    predict_proba(texts)  -> ndarray (n, 2)   # for LIME / SHAP
    classify_fn(text)     -> float            # P(ADE) for the agent
    embed(texts)          -> ndarray (n, h)   # CLS/mean vectors for FAISS

Performance note

"""

import numpy as np

DEFAULT_DIR = "biobert_out"


class BioBERTClassifier:
    """Wrap a fine-tuned BioBERT sequence classifier for inference."""

    def __init__(self, model_dir=DEFAULT_DIR, max_len=128, device=None,
                 batch_size=16):
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
        )

        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(self.device).eval()

    
    def _encode(self, texts):
        return self.tokenizer(
            [str(t) for t in texts], truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        ).to(self.device)

    def predict_proba(self, texts):
        """Return (n, 2) class probabilities. Signature matches sklearn/LIME."""
        import torch

        if isinstance(texts, str):
            texts = [texts]
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                enc = self._encode(batch)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(probs)
        return np.vstack(out) if out else np.empty((0, 2))

    def classify_fn(self, text):
        """Return P(ADE) for a single text, for the agent's classify tool."""
        return float(self.predict_proba([text])[0][1])

    def embed(self, texts, pooling="cls"):
        """
        Return fixed embeddings for FAISS retrieval over BioBERT space.

        pooling : {'cls', 'mean'}
        """
        import torch

        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                enc = self._encode(batch)
                hidden = self.model.base_model(**enc).last_hidden_state
                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    pooled = hidden[:, 0, :]
                vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors).astype("float32")


def load_biobert(model_dir=DEFAULT_DIR, max_len=128):
    """Convenience loader returning a ready BioBERTClassifier."""
    return BioBERTClassifier(model_dir=model_dir, max_len=max_len)


def biobert_embedder(clf):
    """Adapt a BioBERTClassifier into an embed_fn for the FAISS retriever."""
    def _fn(texts):
        return clf.embed(texts)
    return _fn
