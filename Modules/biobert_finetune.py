"""
BioBERT Fine-Tuning 

Fine-tunes a domain-specific biomedical transformer (BioBERT) end-to-end for
adverse-drug-event detection. Unlike the SVM-BERT hybrid (frozen features),
here the transformer weights are updated, which typically gives the strongest
single-model performance in the project's model ladder.

Uses the HuggingFace Trainer API for a concise, well-tested training loop.
The Trainer manages device placement itself (GPU when available). Requires
`transformers`, `torch`, and `datasets` (preinstalled on Colab).

Default checkpoint: 'dmis-lab/biobert-base-cased-v1.1'.
"""

import argparse

import numpy as np

import eval_utils as ev

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
# Clean-loading biomedical alternative (proper fast-tokenizer files, no
# sentencepiece needed). Same domain-pretraining idea as BioBERT.
PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def _load_tokenizer(model_name):
    """ Load the tokenizer robustly.

    """
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"[biobert_finetune] tokenizer ok: {type(tok).__name__} (slow)")
        return tok
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        print(f"[biobert_finetune] AutoTokenizer failed ({exc}); "
              "falling back to BertTokenizer.")
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(model_name)
        print("[biobert_finetune] tokenizer ok: BertTokenizer (fallback)")
        return tok


def _load_model(model_name, num_labels=2):
    """Load the sequence-classification model robustly.
    """
    from transformers import AutoModelForSequenceClassification
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        print(f"[biobert_finetune] model ok: {type(mdl).__name__} (auto)")
        return mdl
    except (ValueError, OSError, KeyError) as exc:
        print(f"[biobert_finetune] AutoModel failed ({exc}); falling back "
              "to BertForSequenceClassification.")
        from transformers import BertForSequenceClassification
        mdl = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        print("[biobert_finetune] model ok: BertForSequenceClassification "
              "(fallback)")
        return mdl


def _softmax(logits):
    """Numerically stable row-wise softmax."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _tokenize_dataset(df, tokenizer, max_len):
    """Build a HuggingFace Dataset of tokenised text + labels from a frame."""
    from datasets import Dataset

    ds = Dataset.from_dict({
        "text": df[ev.TEXT_COL].astype(str).tolist(),
        "label": df[ev.LABEL_COL].astype(int).tolist(),
    })

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=max_len)

    return ds.map(_tok, batched=True)


def _metric_fn(eval_pred):
    """Compute metrics for the HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    proba = _softmax(logits)[:, 1]
    return ev.compute_metrics(labels, preds, proba)


def fine_tune(train_df, val_df, model_name=DEFAULT_MODEL, max_len=128,
              epochs=3, batch_size=16, lr=2e-5, output_dir="biobert_out"):
    """
    Fine-tune BioBERT and return (trainer, tokenizer).

    The best checkpoint by validation F1 is loaded at the end.
    """
    import torch
    from transformers import TrainingArguments, Trainer

    # BioBERT ships only a slow (WordPiece) tokenizer and a config.json with
    # no model_type; the robust loaders handle both via BertTokenizer /
    # BertForSequenceClassification fallbacks (no sentencepiece needed).
    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name, num_labels=2)

    train_ds = _tokenize_dataset(train_df, tokenizer, max_len)
    val_ds = _tokenize_dataset(val_df, tokenizer, max_len)

    # Class weights to counter imbalance, injected via a custom Trainer.
    labels = train_df[ev.LABEL_COL].values
    pos = max(int(np.sum(labels)), 1)
    neg = max(len(labels) - pos, 1)
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float)

    class WeightedTrainer(Trainer):
        """Trainer using a class-weighted cross-entropy loss."""

        def compute_loss(self, model, inputs, return_outputs=False,
                         **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(outputs.logits.device)
            )
            loss = loss_fct(outputs.logits, labels_)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=ev.RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_metric_fn,
    )
    trainer.train()
    return trainer, tokenizer


def _predict_logits(trainer, tokenizer, df, max_len=128):
    """Return raw logits for a dataframe split using a fine-tuned trainer."""
    ds = _tokenize_dataset(df, tokenizer, max_len)
    return trainer.predict(ds).predictions


def evaluate_on_df(trainer, tokenizer, df, max_len=128):
    """Evaluate a fine-tuned trainer on a dataframe split."""
    logits = _predict_logits(trainer, tokenizer, df, max_len)
    preds = np.argmax(logits, axis=1)
    proba = _softmax(logits)[:, 1]
    return ev.compute_metrics(df[ev.LABEL_COL].values, preds, proba)


def predict_proba_on_df(trainer, tokenizer, df, max_len=128):
    """
    Return positive-class probabilities for a dataframe split.

    Used by the ensemble meta-learner, which consumes per-model probabilities.
    """
    logits = _predict_logits(trainer, tokenizer, df, max_len)
    return _softmax(logits)[:, 1]


def embed_texts(trainer, tokenizer, texts, max_len=128, batch_size=16,
                pooling="cls"):
    """
    Produce fixed-length embeddings from the fine-tuned BioBERT encoder.

    These domain-specific, task-tuned vectors are what the FAISS case-retrieval
    module indexes, so "similar" means similar in ways that matter
    for adverse-event detection.

    """
    import numpy as np
    import torch

    # The classification model wraps the BioBERT encoder; reach it via base_model
    # so we read the contextual embeddings rather than the classification logits.
    model = trainer.model
    encoder = model.base_model
    device = model.device

    texts = [str(t) for t in texts]
    vectors = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt",
            ).to(device)
            hidden = encoder(**enc).last_hidden_state  # (B, T, H)
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:  # 'cls'
                pooled = hidden[:, 0, :]
            vectors.append(pooled.cpu().numpy())

    embeddings = np.vstack(vectors).astype("float32")
    # L2-normalise so inner-product search in FAISS == cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def run_cross_domain(full_df, train_domain="Formal", test_domain="Informal",
                     model_name=DEFAULT_MODEL, epochs=3):
    """Fine-tune on one domain, evaluate on the other (Objective 3)."""
    train_part = ev.split_by_domain(full_df, train_domain)
    test_part = ev.split_by_domain(full_df, test_domain)
    if len(train_part) == 0 or len(test_part) == 0:
        raise ValueError("Empty domain split — check the domain labels.")
    val_part = train_part.sample(frac=0.15, random_state=ev.RANDOM_STATE)
    train_part = train_part.drop(val_part.index)
    trainer, tokenizer = fine_tune(train_part, val_part, model_name,
                                   epochs=epochs)
    return evaluate_on_df(trainer, tokenizer, test_part)


def save_model(trainer, tokenizer, directory):
    """Save the fine-tuned model weights and tokenizer to ``directory``."""
    import os
    os.makedirs(directory, exist_ok=True)
    trainer.save_model(directory)
    tokenizer.save_pretrained(directory)


def load_model(directory):
    """
    Reload a fine-tuned model + tokenizer for inference.

    Returns (trainer, tokenizer) where the trainer wraps the reloaded model,
    ready for evaluate_on_df / predict_proba_on_df / embed_texts.
    """
    from transformers import Trainer
    tokenizer = _load_tokenizer(directory)
    model = _load_model(directory)
    trainer = Trainer(model=model, compute_metrics=_metric_fn)
    return trainer, tokenizer


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BioBERT fine-tuning")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--full", default="data/processed/unified_full.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_df, val_df, test_df, full_df = ev.load_splits(
        args.train, args.val, args.test, args.full
    )

    print("=== BioBERT fine-tuning (standard) ===")
    trainer, tokenizer = fine_tune(train_df, val_df, args.model,
                                   epochs=args.epochs)
    standard = evaluate_on_df(trainer, tokenizer, test_df)
    print({k: round(v, 3) for k, v in standard.items()})

    print("\n=== BioBERT cross-domain (Formal -> Informal) ===")
    cross = run_cross_domain(full_df, "Formal", "Informal", args.model,
                             epochs=args.epochs)
    print({k: round(v, 3) for k, v in cross.items()})

    print("\nGap:", ev.degradation_row("BioBERT", standard["f1"], cross["f1"]))


if __name__ == "__main__":
    main()
