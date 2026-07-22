"""
 Bi-LSTM with Attention (neural tier).

* Pure PyTorch, no heavy framework lock-in.
* A small, self-contained vocabulary is built from the training split only
  (no leakage from val/test).
* Class imbalance is handled with a positive-class weight in the loss.
* Device selection is centralised in eval_utils.get_device().
* The module is import-safe: training only runs via train_model() / main().

"""

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import eval_utils as ev

PAD, UNK = "<pad>", "<unk>"
RANDOM_STATE = ev.RANDOM_STATE

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Vocabulary + dataset
# --------------------------------------------------------------------------- #
def build_vocab(texts, min_freq=2, max_size=20000):
    """Build a {token: index} mapping from an iterable of whitespace strings."""
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    vocab = {PAD: 0, UNK: 1}
    for token, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    """Encode one string into a fixed-length list of token indices."""
    ids = [vocab.get(tok, vocab[UNK]) for tok in str(text).split()][:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class ADEDataset(Dataset):
    """Torch dataset yielding (token_ids, label) pairs."""

    def __init__(self, texts, labels, vocab, max_len):
        self.x = [encode(t, vocab, max_len) for t in texts]
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float),
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Additive attention over Bi-LSTM hidden states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        """Return (context_vector, attention_weights)."""
        scores = self.context(torch.tanh(self.proj(encoder_outputs)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights.squeeze(-1)


class BiLSTMAttention(nn.Module):
    """Embedding -> Bi-LSTM -> attention -> sigmoid classifier."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """Return (logit, attention_weights)."""
        emb = self.dropout(self.embedding(x))
        outputs, _ = self.lstm(emb)
        context, weights = self.attention(outputs)
        logit = self.fc(self.dropout(context)).squeeze(-1)
        return logit, weights


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _make_loader(texts, labels, vocab, max_len, batch_size, shuffle):
    dataset = ADEDataset(texts, labels, vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _pos_weight(labels):
    """Positive-class weight = (#neg / #pos) to counter class imbalance."""
    pos = max(int(np.sum(labels)), 1)
    neg = max(len(labels) - pos, 1)
    return torch.tensor([neg / pos], dtype=torch.float)


def train_model(train_df, val_df, vocab=None, max_len=80, embed_dim=128,
                hidden_dim=128, epochs=8, batch_size=32, lr=1e-3,
                device=None, verbose=True):
    """
    Train the Bi-LSTM+attention model.

    Returns
    -------
    model : trained BiLSTMAttention
    vocab : dict used for encoding
    """
    device = device or ev.get_device()
    vocab = vocab or build_vocab(train_df[ev.TEXT_COL])

    train_loader = _make_loader(
        train_df[ev.TEXT_COL], train_df[ev.LABEL_COL], vocab,
        max_len, batch_size, shuffle=True,
    )
    val_loader = _make_loader(
        val_df[ev.TEXT_COL], val_df[ev.LABEL_COL], vocab,
        max_len, batch_size, shuffle=False,
    )

    model = BiLSTMAttention(len(vocab), embed_dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=_pos_weight(train_df[ev.LABEL_COL].values).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit, _ = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:2d} | loss {epoch_loss / len(train_loader):.4f} "
                  f"| val_f1 {val_metrics['f1']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, vocab


@torch.no_grad()
def _predict(model, loader, device):
    """Return (y_true, y_pred, y_proba) lists for a loader."""
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit, _ = model(xb)
        proba = torch.sigmoid(logit).cpu().numpy()
        y_proba.extend(proba.tolist())
        y_pred.extend((proba >= 0.5).astype(int).tolist())
        y_true.extend(yb.numpy().astype(int).tolist())
    return y_true, y_pred, y_proba


def evaluate(model, loader, device):
    """Run the model over a loader and return the standard metric dict."""
    y_true, y_pred, y_proba = _predict(model, loader, device)
    return ev.compute_metrics(y_true, y_pred, y_proba)


def evaluate_on_df(model, vocab, df, max_len=80, batch_size=32, device=None):
    """Convenience wrapper to evaluate a trained model on a dataframe split."""
    device = device or ev.get_device()
    loader = _make_loader(
        df[ev.TEXT_COL], df[ev.LABEL_COL], vocab, max_len, batch_size, False
    )
    return evaluate(model, loader, device)


def predict_proba_on_df(model, vocab, df, max_len=80, batch_size=32, device=None):
    """
    Return positive-class probabilities for a dataframe split.

    Used by the ensemble meta-learner, which consumes per-model probabilities.
    """
    device = device or ev.get_device()
    loader = _make_loader(
        df[ev.TEXT_COL], df[ev.LABEL_COL], vocab, max_len, batch_size, False
    )
    _, _, y_proba = _predict(model, loader, device)
    return np.asarray(y_proba)


def run_cross_domain(full_df, train_domain="Formal", test_domain="Informal",
                     **train_kwargs):
    """Train on one domain, test on the other (Objective 3). Returns metrics."""
    train_part = ev.split_by_domain(full_df, train_domain)
    test_part = ev.split_by_domain(full_df, test_domain)
    if len(train_part) == 0 or len(test_part) == 0:
        raise ValueError("Empty domain split — check the domain labels.")
    # carve a small val set from the training domain
    val_part = train_part.sample(frac=0.15, random_state=RANDOM_STATE)
    train_part = train_part.drop(val_part.index)
    model, vocab = train_model(train_part, val_part, verbose=False,
                               **train_kwargs)
    return evaluate_on_df(model, vocab, test_part)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bi-LSTM + attention baseline")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--full", default="data/processed/unified_full.csv")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_df, val_df, test_df, full_df = ev.load_splits(
        args.train, args.val, args.test, args.full
    )

    print("=== Training Bi-LSTM + attention (standard) ===")
    model, vocab = train_model(train_df, val_df, epochs=args.epochs)
    standard = evaluate_on_df(model, vocab, test_df)
    print("Standard test metrics:", {k: round(v, 3) for k, v in standard.items()})

    print("\n=== Cross-domain (Formal -> Informal) ===")
    cross = run_cross_domain(full_df, "Formal", "Informal", epochs=args.epochs)
    print("Cross-domain metrics:", {k: round(v, 3) for k, v in cross.items()})

    gap = ev.degradation_row("Bi-LSTM+Attention", standard["f1"], cross["f1"])
    print("\nGeneralisation gap:", gap)


if __name__ == "__main__":
    main()
