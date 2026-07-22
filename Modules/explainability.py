"""
Explainability & Clinical Triage.

Turns a trained adverse-drug-event classifier into an interpretable, triage-ready
system using three complementary methods:

    * LIME      - local, per-prediction word importances (model-agnostic).
    * SHAP      - additive feature attributions with consistency guarantees.
    * Attention - token weights read directly from the Bi-LSTM+attention model.

It also maps a model's confidence and predicted severity onto a four-level
clinical triage scale (None / Mild / Moderate / Severe) with confidence-based
routing, so uncertain or high-severity cases can be escalated to a human.

The methods are decoupled from any specific model: LIME and SHAP take a
`predict_proba`-style callable, so they work for the TF-IDF baselines, the
SVM-BERT hybrid, or a fine-tuned transformer alike.
"""

# Triage configuration: confidence bands and routing.
SEVERITY_LEVELS = ("none", "mild", "moderate", "severe")
LOW_CONFIDENCE = 0.60          # below this -> route to human review
HIGH_SEVERITY = ("moderate", "severe")



# LIME

def explain_with_lime(text, predict_proba_fn, class_names=("No ADE", "ADE"),
                      num_features=10, num_samples=1000):
    """
    Produce a LIME explanation for a single text.

    Parameters
    
    text : str
        The input to explain.
    predict_proba_fn : callable
        Maps a list[str] -> ndarray of shape (n, 2) of class probabilities.
    num_features : int
        How many words to include in the explanation.

    Returns
    
    list[tuple[str, float]]
        (word, weight) pairs; positive weight pushes toward the ADE class.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=list(class_names))
    explanation = explainer.explain_instance(
        text, predict_proba_fn,
        num_features=num_features, num_samples=num_samples,
    )
    return explanation.as_list()


def lime_predict_proba_sklearn(model, vectorizer):
    """
    Build a LIME-compatible predict_proba from an sklearn model + vectorizer.

    Returns a callable list[str] -> (n, 2) probability array.
    """
    def _fn(texts):
        features = vectorizer.transform(texts)
        return model.predict_proba(features)
    return _fn



# SHAP

def explain_with_shap(texts, predict_proba_fn, max_evals=200):
    """
    Compute SHAP values for a batch of texts using a model-agnostic explainer.

    Returns the shap.Explanation object (use shap plots to visualise).
    """
    import shap

    masker = shap.maskers.Text()
    explainer = shap.Explainer(predict_proba_fn, masker)
    return explainer(list(texts), max_evals=max_evals)


def top_shap_words(shap_values, index=0, top_k=10):
    """
    Extract the top-k most influential words for one example from SHAP values.

    Returns list[tuple[str, float]] sorted by absolute contribution.
    """
    values = shap_values[index].values
    tokens = shap_values[index].data
    # For binary output SHAP may return per-class columns; take the ADE column.
    if values.ndim > 1:
        values = values[:, -1]
    pairs = list(zip(tokens, values))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs[:top_k]


def shap_top_words_for_text(text, predict_proba_fn, top_k=8, max_evals=200):
    """
    Convenience wrapper: SHAP top words for one text as (word, weight) tuples.

    Mirrors the shape of explain_with_lime's output so the two methods can be
    used interchangeably (for example, together at an agent's explain step).
    """
    shap_values = explain_with_shap([text], predict_proba_fn,
                                    max_evals=max_evals)
    return top_shap_words(shap_values, index=0, top_k=top_k)



# Attention (Bi-LSTM)

def attention_weights_for_text(model, vocab, text, max_len=80, device=None):
    """
    Return per-token attention weights from the Bi-LSTM+attention model.

    Returns
   
    list[tuple[str, float]]
        (token, weight) for the real (non-padding) tokens, in order.
    """
    import torch
    from bilstm_attention import encode, PAD

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ids = encode(text, vocab, max_len)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, weights = model(tensor)
    weights = weights.squeeze(0).cpu().numpy()

    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    pad_idx = vocab[PAD]
    tokens = [(inv_vocab.get(i, "<unk>"), float(w))
              for i, w in zip(ids, weights) if i != pad_idx]
    return tokens


def normalise_weights(token_weights):
    """Scale a list of (token, weight) so weights sum to 1 for display."""
    total = sum(abs(w) for _, w in token_weights) or 1.0
    return [(tok, abs(w) / total) for tok, w in token_weights]


# Clinical triage

def triage(ade_probability, severity="mild",
           low_confidence=LOW_CONFIDENCE):
    """
    Map a prediction to a clinical triage decision.

    Parameters
    ----------
    ade_probability : float
        Model probability that an ADE is present (0-1).
    severity : str
        Predicted/derived severity in SEVERITY_LEVELS.

    Returns
    -------
    dict with keys: triage_level, route, confidence, rationale.
    """
    severity = severity if severity in SEVERITY_LEVELS else "mild"
    confident = ade_probability >= low_confidence or ade_probability <= (
        1 - low_confidence
    )

    if ade_probability < 0.5:
        level = "none"
    else:
        level = severity if severity != "none" else "mild"

    needs_human = (not confident) or (level in HIGH_SEVERITY)
    route = "human_review" if needs_human else "auto_logged"

    if not confident:
        rationale = "Low model confidence — escalated for human review."
    elif level in HIGH_SEVERITY:
        rationale = f"Predicted {level} severity — escalated for clinician."
    elif level == "none":
        rationale = "No adverse event detected."
    else:
        rationale = "Mild adverse event, auto-logged for monitoring."

    return {
        "triage_level": level,
        "route": route,
        "confidence": round(float(ade_probability), 3),
        "rationale": rationale,
    }


def explain_and_triage(text, predict_proba_fn, severity="mild",
                       num_features=8):
    """
    Convenience: run LIME + triage for one text and return a combined report.
    """
    proba = float(predict_proba_fn([text])[0][1])
    decision = triage(proba, severity)
    try:
        words = explain_with_lime(text, predict_proba_fn,
                                  num_features=num_features)
    except Exception as exc:  # explainability must not break the prediction
        words = [("(LIME unavailable)", 0.0)]
        decision["rationale"] += f" [LIME error: {exc}]"
    return {
        "text": text,
        "ade_probability": round(proba, 3),
        "top_words": words,
        **decision,
    }


def format_token_heatmap(token_weights, width=60):
    """
    Render token weights as a simple text heatmap for notebooks/logs.

    Heavier-weighted tokens are shown in upper case with a bar.
    """
    norm = normalise_weights(token_weights)
    lines = []
    for tok, w in norm:
        bar = "#" * int(round(w * width))
        marker = tok.upper() if w > (1.5 / max(len(norm), 1)) else tok
        lines.append(f"{marker:>20s} | {bar} {w:.3f}")
    return "\n".join(lines)
