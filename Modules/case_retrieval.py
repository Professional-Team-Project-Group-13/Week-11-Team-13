"""
FAISS Case Retrieval 

Provides *example-based* explanation to complement the feature-based methods
(LIME, SHAP) and the model-internal view (attention). Given a new input, it
retrieves the most similar previously seen ADE cases from an indexed corpus,
answering the clinically intuitive question: "what known cases does this
resemble?"

Mechanism
--------
    text -> embedding (TF-IDF or BERT) -> FAISS nearest-neighbour search
         -> k most similar training cases (with their labels and severity)

This supports clinical triage: a reviewer sees not only *why* the model
predicted an ADE (LIME/SHAP) but *what comparable cases* looked like and how
they were labelled, which aids trust and consistency.

FAISS is optional; if it is not installed the module falls back to exact
cosine similarity via scikit-learn, so the pipeline never breaks.
"""

import numpy as np

import eval_utils as ev

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class CaseRetriever:
    """
    Index a corpus of cases and retrieve nearest neighbours for new text.

    Parameters
    ----------
    embed_fn : callable
        Maps list[str] -> 2D float32 ndarray of embeddings. Supply a TF-IDF
        or BERT embedder. The same function is used for index and query so the
        vector spaces match.
    metric : {'cosine', 'l2'}
        'cosine' normalises vectors and uses inner-product search.
    """

    def __init__(self, embed_fn, metric="cosine"):
        self.embed_fn = embed_fn
        self.metric = metric
        self.index = None
        self.cases = None          # DataFrame of indexed cases
        self._matrix = None        # fallback storage when FAISS is absent

    
    def _embed(self, texts):
        vectors = np.asarray(self.embed_fn(list(texts)), dtype="float32")
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-9, None)
        return vectors

    def build(self, cases_df, text_col=ev.TEXT_COL):
        """
        Build the search index from a dataframe of cases.

        The dataframe is retained so retrieved neighbours carry their label,
        severity, domain and original text.
        """
        self.cases = cases_df.reset_index(drop=True)
        vectors = self._embed(self.cases[text_col])
        dim = vectors.shape[1]

        if _HAS_FAISS:
            # Inner product on normalised vectors == cosine similarity.
            self.index = (faiss.IndexFlatIP(dim) if self.metric == "cosine"
                          else faiss.IndexFlatL2(dim))
            self.index.add(vectors)
        else:
            self._matrix = vectors  # brute-force fallback
        return self

    def query(self, text, k=5, text_col=ev.TEXT_COL):
        """
        Return the k most similar indexed cases to `text`.

        Returns
       
        list[dict] with keys: rank, similarity, text, label, severity, domain.
        """
        if self.cases is None:
            raise RuntimeError("Call build() before query().")
        vector = self._embed([text])

        if _HAS_FAISS:
            scores, idxs = self.index.search(vector, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = (self._matrix @ vector[0]
                    if self.metric == "cosine"
                    else -np.linalg.norm(self._matrix - vector[0], axis=1))
            idxs = np.argsort(sims)[::-1][:k]
            scores = sims[idxs]

        results = []
        for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
            if i < 0:
                continue
            row = self.cases.iloc[int(i)]
            results.append({
                "rank": rank,
                "similarity": round(float(score), 4),
                "text": str(row.get(text_col, ""))[:300],
                "label": int(row.get(ev.LABEL_COL, -1)),
                "severity": row.get("severity", "n/a"),
                "domain": row.get(ev.DOMAIN_COL, "n/a"),
            })
        return results

    def retrieval_vote(self, text, k=5):
        """
        Simple k-NN label estimate from retrieved neighbours.

        Returns (predicted_label, confidence) where confidence is the fraction
        of neighbours sharing the majority label. Useful as an independent,
        non-parametric second opinion for the agent's decision logic.
        """
        neighbours = self.query(text, k=k)
        if not neighbours:
            return 0, 0.0
        labels = [n["label"] for n in neighbours if n["label"] in (0, 1)]
        if not labels:
            return 0, 0.0
        ones = sum(labels)
        pred = 1 if ones >= len(labels) / 2 else 0
        conf = max(ones, len(labels) - ones) / len(labels)
        return pred, round(conf, 3)


def tfidf_embedder(vectorizer):
    """Adapt a fitted TF-IDF vectorizer into a dense embed_fn for retrieval."""
    def _fn(texts):
        return vectorizer.transform(texts).toarray().astype("float32")
    return _fn


def format_neighbours(neighbours):
    """Render retrieved cases as a readable block for notebooks/logs."""
    lines = []
    for n in neighbours:
        lines.append(
            f"#{n['rank']} (sim={n['similarity']:.3f}) "
            f"[label={n['label']}, severity={n['severity']}, "
            f"domain={n['domain']}]\n    {n['text'][:140]}"
        )
    return "\n".join(lines)
