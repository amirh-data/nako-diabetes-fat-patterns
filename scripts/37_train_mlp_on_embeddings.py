"""
scripts/37_train_mlp_on_embeddings.py

Train a small MLP classifier on VAT/SAT encoder embeddings
to predict T2D vs normal.

This tests whether the proxy-task trained encoder
produces better image representations for T2D classification
than direct supervised training.

Usage:
    python -u -m scripts.37_train_mlp_on_embeddings \
        --model_name convnext3d \
        --input_mode fat_water
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",  type=str, default="convnext3d",
                    choices=["convnext3d", "resnet10_3d"])
parser.add_argument("--input_mode",  type=str, default="fat_water",
                    choices=["fat_water", "fat"])
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--lr",          type=float, default=1e-3)
args = parser.parse_args()

EMBEDDING_CSV = (
    BASE /
    f"vatsat_encoder/{args.model_name}_{args.input_mode}/vatsat_embeddings_matched.csv"
)
OUT_DIR = BASE / f"vatsat_encoder/{args.model_name}_{args.input_mode}"


def train_mlp(X_train, y_train, X_val, y_val, device):
    """Train small MLP classifier on embeddings."""

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2),
            )
        def forward(self, x):
            return self.net(x)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long)

    n0          = (y_tr == 0).sum().item()
    n1          = (y_tr == 1).sum().item()
    cls_weights = torch.tensor([1.0/n0, 1.0/n1], dtype=torch.float32).to(device)
    cls_weights = cls_weights / cls_weights.sum() * 2

    mlp       = MLPClassifier(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc          = 0.0
    best_state        = None
    epochs_no_improve = 0
    patience          = 20

    for epoch in range(args.epochs):
        mlp.train()
        optimizer.zero_grad()
        logits = mlp(X_tr)
        loss   = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(X_vl.to(device))
            val_prob   = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_pred   = torch.argmax(val_logits, dim=1).cpu().numpy()

        auc = roc_auc_score(y_vl.numpy(), val_prob)

        if auc > best_auc:
            best_auc          = auc
            best_state        = {k: v.clone() for k, v in mlp.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best
    mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        val_logits = mlp(X_vl.to(device))
        val_prob   = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_pred   = torch.argmax(val_logits, dim=1).cpu().numpy()

    return val_prob, val_pred, best_auc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:        {device}")
    print(f"Model:         {args.model_name}")
    print(f"Input mode:    {args.input_mode}")
    print(f"Embedding CSV: {EMBEDDING_CSV}")

    if not EMBEDDING_CSV.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDING_CSV}\n"
            f"Run script 36 first:\n"
            f"python -u -m scripts.36_extract_vatsat_embeddings "
            f"--model_name {args.model_name} --input_mode {args.input_mode}"
        )

    # Load embeddings
    df = pd.read_csv(EMBEDDING_CSV)
    print(f"Loaded embeddings: {df.shape}")
    print(df["label"].value_counts())

    emb_cols = [c for c in df.columns if c.startswith("vatsat_emb_")]
    X = df[emb_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    print(f"Embedding dim: {X.shape[1]}")

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    results = []

    # -------------------------------------------------------------
    # Method 1: Logistic Regression on embeddings
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Method 1: Logistic Regression on VAT/SAT embeddings")

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    lr_model.fit(X_train, y_train)
    lr_prob = lr_model.predict_proba(X_val)[:, 1]
    lr_pred = lr_model.predict(X_val)

    lr_auc     = roc_auc_score(y_val, lr_prob)
    lr_bal_acc = balanced_accuracy_score(y_val, lr_pred)
    lr_cm      = confusion_matrix(y_val, lr_pred)

    print(f"AUC:               {lr_auc:.4f}")
    print(f"Balanced accuracy: {lr_bal_acc:.4f}")
    print(f"Confusion matrix:\n{lr_cm}")

    results.append({
        "method":    "logistic_regression",
        "model":     args.model_name,
        "input":     args.input_mode,
        "auc":       lr_auc,
        "bal_acc":   lr_bal_acc,
    })

    # -------------------------------------------------------------
    # Method 2: MLP classifier on embeddings
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Method 2: MLP classifier on VAT/SAT embeddings")

    mlp_prob, mlp_pred, mlp_best_auc = train_mlp(
        X_train, y_train, X_val, y_val, device
    )

    mlp_auc     = roc_auc_score(y_val, mlp_prob)
    mlp_bal_acc = balanced_accuracy_score(y_val, mlp_pred)
    mlp_cm      = confusion_matrix(y_val, mlp_pred)

    print(f"Best AUC:          {mlp_best_auc:.4f}")
    print(f"Final AUC:         {mlp_auc:.4f}")
    print(f"Balanced accuracy: {mlp_bal_acc:.4f}")
    print(f"Confusion matrix:\n{mlp_cm}")
    print(classification_report(y_val, mlp_pred, digits=4))

    results.append({
        "method":    "mlp_classifier",
        "model":     args.model_name,
        "input":     args.input_mode,
        "best_auc":  mlp_best_auc,
        "auc":       mlp_auc,
        "bal_acc":   mlp_bal_acc,
    })

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY: VAT/SAT Proxy Encoder → T2D Classification")
    print(f"Model:        {args.model_name}")
    print(f"Input mode:   {args.input_mode}")
    print(f"Encoder loss: ~0.039 (best val_loss during training)")
    print()
    print("Comparison with baselines:")
    print(f"  clinical only (matched):            AUC 0.481")
    print(f"  global biomarkers only (matched):   AUC 0.555")
    print(f"  image CNN only (matched):            AUC 0.551")
    print(f"  image + biomarkers (matched):        AUC 0.543")
    print()
    print(f"  VAT/SAT encoder → LR (matched):     AUC {lr_auc:.4f}")
    print(f"  VAT/SAT encoder → MLP (matched):    AUC {mlp_auc:.4f}")
    print("=" * 60)

    # Save results
    results_df = pd.DataFrame(results)
    out_path   = OUT_DIR / "t2d_classification_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()