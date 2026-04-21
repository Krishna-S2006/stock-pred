from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# SEQUENCE CREATION (FIXED)
# =========================
def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = 5,
):
    xs = []
    ys = []

    if len(features) <= seq_len:
        return torch.empty(0), torch.empty(0)

    for i in range(len(features) - seq_len):
        xs.append(features[i : i + seq_len])
        ys.append(targets[i + seq_len])

    if len(xs) == 0:
        return torch.empty(0), torch.empty(0)

    x_arr = np.stack(xs)
    y_arr = np.array(ys)

    return torch.from_numpy(x_arr).float(), torch.from_numpy(y_arr).long()


# =========================
# MODEL
# =========================
class PriceMovementLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# =========================
# RESULT CONTAINER
# =========================
@dataclass
class LSTMTrainingResult:
    model: PriceMovementLSTM
    scaler: StandardScaler
    feature_columns: Sequence[str]
    seq_len: int


# =========================
# TRAINING (FULL FIXED)
# =========================
def train_lstm_on_dataframe(
    df_supervised: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "target_up",
    seq_len: int = 5,
    test_size: float = 0.2,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> LSTMTrainingResult:

    df = df_supervised.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, shuffle=False
    )

    # Create sequences
    x_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len=seq_len)
    x_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len=seq_len)

    # 🔥 HANDLE EMPTY TRAIN DATA (CRITICAL)
    if x_train_seq.size(0) == 0:
        raise ValueError("Not enough training data to create sequences. Increase dataset size.")

    # 🔥 HANDLE EMPTY TEST DATA
    if x_test_seq.size(0) == 0:
        print("⚠️ Not enough test data. Skipping evaluation.")
        x_test_seq = None
        y_test_seq = None

    model = PriceMovementLSTM(input_size=len(feature_columns)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
        num_samples = x.size(0)
        for i in range(0, num_samples, batch_size):
            yield x[i : i + batch_size], y[i : i + batch_size]

    # =========================
    # TRAIN LOOP
    # =========================
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in _iterate_batches(x_train_seq, y_train_seq, batch_size):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        train_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {train_acc:.4f}")

    # =========================
    # EVALUATION
    # =========================
    if x_test_seq is not None:
        model.eval()
        with torch.no_grad():
            logits = model(x_test_seq.to(DEVICE))
            preds = logits.argmax(dim=1).cpu()
            test_acc = (preds == y_test_seq).float().mean().item()
            print(f"Test Accuracy: {test_acc:.4f}")

    return LSTMTrainingResult(
        model=model,
        scaler=scaler,
        feature_columns=list(feature_columns),
        seq_len=seq_len,
    )


# =========================
# PREDICTION
# =========================
def predict_next_movement(
    training_result: LSTMTrainingResult,
    recent_features: pd.DataFrame,
) -> Tuple[int, float]:

    seq_len = training_result.seq_len
    feat_cols = list(training_result.feature_columns)

    df = recent_features.dropna(subset=feat_cols).reset_index(drop=True)

    if len(df) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows to make a prediction")

    last_seq = df[feat_cols].values.astype(np.float32)[-seq_len:]
    last_seq_scaled = training_result.scaler.transform(last_seq)

    x = torch.from_numpy(last_seq_scaled).unsqueeze(0).float().to(DEVICE)

    model = training_result.model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = int(np.argmax(probs))
    prob_up = float(probs[1])

    return pred_label, prob_up


__all__ = [
    "PriceMovementLSTM",
    "LSTMTrainingResult",
    "train_lstm_on_dataframe",
    "predict_next_movement",
]
