"""
SVM-BERT Hybrid 

A hybrid model that uses frozen pre-trained BERT embeddings as a fixed feature
extractor and a Support Vector Machine as the classifier. This decouples
representation (contextual embeddings) from decision boundary (max-margin SVM),
which is fast to train and a strong, well-understood baseline above the
classical TF-IDF models.


The embedding step is cached per split so repeated SVM fits are cheap.
Device selection is centralised in eval_utils.get_device().
Requires `transformers` and `torch`; both are preinstalled on Colab.

"""

import argparse

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eval_utils as ev

DEFAULT_MODEL = "bert-base-uncased"


def embed_texts(texts, model_name=DEFAULT_MODEL, batch_size=16,
                max_len=128, pooling="mean", device=None):
    """
    Encode a list of strings into fixed BERT embeddings.

    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = device or ev.get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    vectors = []
    texts = [str(t) for t in texts]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            ).to(device)
            out = model(**enc).last_hidden_state  # (B, T, H)
            if pooling == "cls":
                pooled = out[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def build_classifier(c=1.0, gamma="scale"):
    """Return a scaled RBF-SVM pipeline with probability estimates enabled."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=c, gamma=gamma, kernel="rbf",
                    class_weight="balanced", probability=True,
                    random_state=ev.RANDOM_STATE)),
    ])


def train_and_eval(train_df, test_df, model_name=DEFAULT_MODEL,
                   pooling="mean", cache=None):
    """
    Embed train/test text with BERT, fit the SVM, and evaluate.

    """
    cache = cache if cache is not None else {}

    def _embed(split_key, texts):
        if split_key not in cache:
            cache[split_key] = embed_texts(texts, model_name, pooling=pooling)
        return cache[split_key]

    x_train = _embed("train", train_df[ev.TEXT_COL])
    x_test = _embed("test", test_df[ev.TEXT_COL])
    y_train = train_df[ev.LABEL_COL].values
    y_test = test_df[ev.LABEL_COL].values

    clf = build_classifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]
    return ev.compute_metrics(y_test, preds, proba), clf


def predict_proba_on_df(clf, df, model_name=DEFAULT_MODEL, pooling="mean"):
    """
    Return positive-class probabilities for a dataframe split.

    Used by the ensemble meta-learner, which consumes per-model probabilities.
    """
    features = embed_texts(df[ev.TEXT_COL], model_name, pooling=pooling)
    return clf.predict_proba(features)[:, 1]


def run_cross_domain(full_df, train_domain="Formal", test_domain="Informal",
                     model_name=DEFAULT_MODEL, pooling="mean"):
    """Train on one domain, test on the other (Objective 3)."""
    train_part = ev.split_by_domain(full_df, train_domain)
    test_part = ev.split_by_domain(full_df, test_domain)
    if len(train_part) == 0 or len(test_part) == 0:
        raise ValueError("Empty domain split — check the domain labels.")
    metrics, _ = train_and_eval(train_part, test_part, model_name, pooling)
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SVM-BERT hybrid")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--full", default="data/processed/unified_full.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    train_df, _, test_df, full_df = ev.load_splits(
        args.train, args.val, args.test, args.full
    )

    print(" SVM-BERT (standard) ")
    standard, _ = train_and_eval(train_df, test_df, args.model)
    print({k: round(v, 3) for k, v in standard.items()})

    print("\n SVM-BERT cross-domain (Formal -> Informal) ")
    cross = run_cross_domain(full_df, "Formal", "Informal", args.model)
    print({k: round(v, 3) for k, v in cross.items()})

    print("\nGap:", ev.degradation_row("SVM-BERT", standard["f1"], cross["f1"]))


if __name__ == "__main__":
    main()
