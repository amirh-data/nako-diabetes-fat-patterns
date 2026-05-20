"""
scripts/40_train_clip.py

CLIP-style contrastive learning for MRI + biomarkers + distribution features.

Architecture:
    Image encoder:   ResNet10_3D → 128-dim image embedding
    Tabular encoder: MLP → 128-dim embedding
                     Input: clinical + global biomarkers + vertebral distribution

Loss:
    InfoNCE contrastive loss
    Same subject → embeddings close
    Different subjects → embeddings far apart

Phases:
    train:    Phase 1 - CLIP contrastive pretraining
    evaluate: Phase 2 - T2D MLP classifier on frozen image embeddings
    both:     Run both phases sequentially

Usage:
    # Matched cohort (fast baseline):
    python -u -m scripts.40_train_clip \
        --phase both \
        --cohort matched \
        --input_mode fat_water_seg_trunk_binary \
        --epochs 50 --epochs_cls 30 --batch_size 64

    # Full cohort (better contrastive learning):
    python -u -m scripts.40_train_clip \
        --phase both \
        --cohort full \
        --input_mode fat_water_seg_trunk_binary \
        --epochs 50 --epochs_cls 30 --batch_size 128
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from src.models.supervised_3d_models import ResNet10_3D


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

ANALYSIS_CSV  = BASE / "quantification_full/analysis_table_with_diabetes_labels.csv"
MATCHED_CSV   = BASE / "cohorts/nako_t2d_normal_matched_cohort.csv"
MATCHED_CACHE = BASE / "cached_nifti128/matched_fat_water_trunkseg"
FULL_CACHE    = BASE / "cached_nifti128/full_fat_water_trunkseg"


# ---------------------------------------------------------------------
# Feature columns
# All combined into one tabular encoder for richer representation
# No separation needed for CLIP - more features = richer learning signal
# ---------------------------------------------------------------------

TABULAR_COLS = [
    # Clinical
    "basis_age",
    "basis_sex",
    "a_anthro_bmi",

    # Global biomarkers - volumes
    "VAT_ml",
    "SAT_abd_ml",
    "SAT_tho_ml",
    "SAT_ml",
    "VAT_SAT_ratio",
    "CardiacAT_ml",
    "Liver_ml",
    "Pancreas_ml",
    "BackMuscle_ml",

    # Global biomarkers - fat fractions
    "Liver_ff_pct",
    "Pancreas_ff_pct",
    "BackMuscle_IMAT_ml",
    "BackMuscle_ff_pct",
    "VertebralMarrow_mean_ff_pct",

    # Vertebral fat distribution - lumbar
    "L5_ff_pct",
    "L4_ff_pct",
    "L3_ff_pct",
    "L2_ff_pct",
    "L1_ff_pct",

    # Vertebral fat distribution - thoracic
    "Th12_ff_pct",
    "Th11_ff_pct",
    "Th10_ff_pct",
    "Th9_ff_pct",
    "Th8_ff_pct",
    "Th7_ff_pct",
    "Th6_ff_pct",
    "Th5_ff_pct",
    "Th4_ff_pct",
    "Th3_ff_pct",
    "Th2_ff_pct",
    "Th1_ff_pct",
]


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--phase",         type=str,   default="both",
                    choices=["train", "evaluate", "both"])
parser.add_argument("--cohort",        type=str,   default="matched",
                    choices=["matched", "full"])
parser.add_argument("--input_mode",    type=str,   default="fat_water_seg_trunk_binary")
parser.add_argument("--epochs",        type=int,   default=50)
parser.add_argument("--epochs_cls",    type=int,   default=30)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--lr_cls",        type=float, default=1e-3)
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--embedding_dim", type=int,   default=128)
parser.add_argument("--temperature",   type=float, default=0.07)
parser.add_argument("--patience",      type=int,   default=10)
parser.add_argument("--resume_from", type=str, default="",
                    help="Resume from checkpoint path")
parser.add_argument("--epochs_ft",  type=int,   default=30)
parser.add_argument("--lr_ft",      type=float, default=1e-5)
args = parser.parse_args()

OUT_DIR   = BASE / f"clip_experiments/{args.cohort}_{args.input_mode}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = MATCHED_CACHE if args.cohort == "matched" else FULL_CACHE


# ---------------------------------------------------------------------
# Tabular encoder
# ---------------------------------------------------------------------

class TabularEncoder(nn.Module):
    """
    MLP encoding clinical + biomarker + distribution features
    into the shared embedding space.
    """
    def __init__(self, n_features, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, tabular_encoder, embedding_dim=128):
        super().__init__()
        self.image_encoder   = image_encoder
        self.tabular_encoder = tabular_encoder
        self.image_proj      = nn.Linear(embedding_dim, embedding_dim)
        self.tabular_proj    = nn.Linear(embedding_dim, embedding_dim)

    def encode_image(self, x):
        _, emb = self.image_encoder(x)
        return F.normalize(self.image_proj(emb), dim=1)

    def encode_tabular(self, t):
        emb = self.tabular_encoder(t)
        return F.normalize(self.tabular_proj(emb), dim=1)

    def forward(self, x, t):
        return self.encode_image(x), self.encode_tabular(t)


# ---------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------

def info_nce_loss(img_emb, tab_emb, temperature=0.07):
    batch_size = img_emb.shape[0]
    logits     = torch.matmul(img_emb, tab_emb.T) / temperature
    labels     = torch.arange(batch_size, device=img_emb.device)
    loss_i2t   = F.cross_entropy(logits,   labels)
    loss_t2i   = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CLIPDataset(Dataset):
    CHANNEL_MAP = {
        "fat":                        slice(0, 1),
        "fat_water":                  slice(0, 2),
        "seg_trunk_binary":           slice(2, 3),
        "seg_trunk_label":            slice(3, 4),
        "fat_water_seg_trunk_binary": [0, 1, 2],
        "fat_water_seg_trunk_label":  [0, 1, 3],
    }

    def __init__(self, df, cache_dir, input_mode, tabular_cols):
        self.df           = df.reset_index(drop=True)
        self.cache_dir    = Path(cache_dir)
        self.input_mode   = input_mode
        self.tabular_cols = tabular_cols

        if input_mode not in self.CHANNEL_MAP:
            raise ValueError(f"Unknown input_mode: {input_mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        nako_id = int(row["NAKO_ID"])
        label   = int(row["label"])

        path   = self.cache_dir / f"NAKO751_{nako_id}.pt"
        sample = torch.load(path, map_location="cpu", weights_only=False)
        x      = sample["x"].float()

        ch = self.CHANNEL_MAP[self.input_mode]
        x  = x[ch] if isinstance(ch, list) else x[ch]

        tab = torch.tensor(
            row[self.tabular_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        return x, tab, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def prepare_dataframe(df, name="df"):
    """
    Convert tabular columns to numeric, impute missing values with median,
    and keep only subjects with cached .pt files.
    """
    for col in TABULAR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    medians = {}
    for col in TABULAR_COLS:
        med = df[col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)

    print(
        f"{name}: after median imputation: {len(df)} subjects "
        f"(removed {before - len(df)})",
        flush=True,
    )

    before_cache = len(df)
    df = df[
        df["NAKO_ID"].apply(
            lambda x: (CACHE_DIR / f"NAKO751_{int(x)}.pt").exists()
        )
    ].copy().reset_index(drop=True)

    print(
        f"{name}: with cached .pt files: {len(df)} "
        f"(removed {before_cache - len(df)})",
        flush=True,
    )
    print(f"{name} label counts:")
    print(df["label"].value_counts(), flush=True)

    return df


def load_matched_data():
    """
    Load the matched cohort for final T2D train/validation evaluation.
    """
    ana = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
    matched = pd.read_csv(MATCHED_CSV)[["NAKO_ID", "label"]]

    df = matched.merge(
        ana[["NAKO_ID"] + TABULAR_COLS],
        on="NAKO_ID",
        how="left",
    )

    df = prepare_dataframe(df, name="matched_df")
    return df


def load_full_data_excluding(exclude_ids):
    """
    Load full normal/T2D cohort for CLIP pretraining,
    excluding selected matched validation subjects to prevent leakage.
    """
    ana = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

    df = ana[ana["diabetes_group"].isin(["normal", "T2D"])].copy()
    df["label"] = (df["diabetes_group"] == "T2D").astype(int)
    df = df[["NAKO_ID", "label"] + TABULAR_COLS].copy()

    df = prepare_dataframe(df, name="full_df_before_exclusion")

    before = len(df)
    exclude_ids = set(int(x) for x in exclude_ids)

    df = df[
        ~df["NAKO_ID"].astype(int).isin(exclude_ids)
    ].copy().reset_index(drop=True)

    print(
        f"full_df_pretrain: after excluding matched validation subjects: "
        f"{len(df)} (removed {before - len(df)})",
        flush=True,
    )
    print("full_df_pretrain label counts:")
    print(df["label"].value_counts(), flush=True)

    return df

# ---------------------------------------------------------------------
# Phase 1: Scaler fit
# ---------------------------------------------------------------------
def fit_tabular_scaler(df):
    """
    Fit scaler using only the training/pretraining data.
    """
    train_mean = df[TABULAR_COLS].mean()
    train_std = df[TABULAR_COLS].std().replace(0, 1)

    pd.DataFrame({
        "feature": TABULAR_COLS,
        "mean": train_mean.values,
        "std": train_std.values,
    }).to_csv(OUT_DIR / "tabular_scaler.csv", index=False)

    return train_mean, train_std


def apply_tabular_scaler(df, mean, std):
    """
    Apply saved mean/std to a dataframe.
    """
    df = df.copy()
    df[TABULAR_COLS] = (df[TABULAR_COLS] - mean) / std
    return df
# ---------------------------------------------------------------------
# Phase 1: CLIP training
# ---------------------------------------------------------------------

def train_clip(df, train_idx, val_idx, device):
    print("=" * 80, flush=True)
    print("PHASE 1: CLIP CONTRASTIVE TRAINING", flush=True)
    print(f"Cohort:      {args.cohort} ({len(df)} subjects)", flush=True)
    print(f"Input mode:  {args.input_mode}", flush=True)
    print(f"Tabular:     {len(TABULAR_COLS)} features", flush=True)
    print(f"Batch size:  {args.batch_size}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Epochs:      {args.epochs}", flush=True)
    print(f"Output:      {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    in_channels = {
        "fat": 1, "fat_water": 2,
        "seg_trunk_binary": 1, "seg_trunk_label": 1,
        "fat_water_seg_trunk_binary": 3,
        "fat_water_seg_trunk_label":  3,
    }[args.input_mode]

    dataset      = CLIPDataset(df, CACHE_DIR, args.input_mode, TABULAR_COLS)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    image_encoder   = ResNet10_3D(in_channels=in_channels,
                                  embedding_dim=args.embedding_dim)
    tabular_encoder = TabularEncoder(n_features=len(TABULAR_COLS),
                                     embedding_dim=args.embedding_dim)
    clip_model      = CLIPModel(image_encoder, tabular_encoder,
                                args.embedding_dim).to(device)

    n_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        clip_model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch       = 0
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from: {args.resume_from}", flush=True)
        res = torch.load(args.resume_from, map_location=device,
                        weights_only=False)
        clip_model.load_state_dict(res["model_state_dict"])
        optimizer.load_state_dict(res["optimizer_state_dict"])
        scheduler.load_state_dict(res["scheduler_state_dict"])
        start_epoch       = res["epoch"]
        best_val_loss     = res["best_val_loss"]
        epochs_no_improve = res["epochs_no_improve"]
        print(f"Resumed from epoch {start_epoch}, "
            f"best_val_loss={best_val_loss:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        clip_model.train()
        train_losses = []
        for x, tab, _ in train_loader:
            x, tab = x.to(device), tab.to(device)
            optimizer.zero_grad()
            img_emb, tab_emb = clip_model(x, tab)
            loss = info_nce_loss(img_emb, tab_emb, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        clip_model.eval()
        val_losses = []
        with torch.no_grad():
            for x, tab, _ in val_loader:
                x, tab = x.to(device), tab.to(device)
                img_emb, tab_emb = clip_model(x, tab)
                val_losses.append(
                    info_nce_loss(img_emb, tab_emb, args.temperature).item()
                )

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_dict = {
                "model_state_dict":     clip_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch + 1,
                "best_val_loss":        best_val_loss,
                "epochs_no_improve":    epochs_no_improve,
                "args":                 vars(args),
                "tabular_cols":         TABULAR_COLS,
                "in_channels":          in_channels,
            }
            torch.save(save_dict, best_ckpt)

            # Always save resume checkpoint
            torch.save(save_dict, OUT_DIR / "resume_checkpoint.pt")
            print(f"Saved best: val_loss={best_val_loss:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"No improvement {epochs_no_improve}/{args.patience}",
                flush=True)
            # Still save resume checkpoint
            torch.save({
                "model_state_dict":     clip_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch + 1,
                "best_val_loss":        best_val_loss,
                "epochs_no_improve":    epochs_no_improve,
                "tabular_cols":         TABULAR_COLS,
                "in_channels":          in_channels,
            }, OUT_DIR / "resume_checkpoint.pt")
            if epochs_no_improve >= args.patience:
                print("Early stopping.", flush=True)
                break

    print(f"\nBest CLIP val_loss: {best_val_loss:.4f}", flush=True)
    return best_ckpt


# ---------------------------------------------------------------------
# Phase 2: T2D classification
# ---------------------------------------------------------------------

def evaluate_t2d(df, train_idx, val_idx, best_ckpt, device):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 2: T2D CLASSIFICATION FROM CLIP IMAGE EMBEDDINGS", flush=True)
    print("=" * 80, flush=True)

    in_channels = {
        "fat": 1, "fat_water": 2,
        "seg_trunk_binary": 1, "seg_trunk_label": 1,
        "fat_water_seg_trunk_binary": 3,
        "fat_water_seg_trunk_label":  3,
    }[args.input_mode]

    image_encoder   = ResNet10_3D(in_channels=in_channels,
                                  embedding_dim=args.embedding_dim)
    tabular_encoder = TabularEncoder(n_features=len(TABULAR_COLS),
                                     embedding_dim=args.embedding_dim)
    clip_model      = CLIPModel(image_encoder, tabular_encoder,
                                args.embedding_dim).to(device)

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["model_state_dict"])
    clip_model.eval()
    print(f"Loaded CLIP from epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['best_val_loss']:.4f}", flush=True)

    # Extract image embeddings
    dataset = CLIPDataset(df, CACHE_DIR, args.input_mode, TABULAR_COLS)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    all_embs, all_labels = [], []
    with torch.no_grad():
        for i, (x, tab, y) in enumerate(loader):
            x = x.to(device)
            all_embs.append(clip_model.encode_image(x).cpu())
            all_labels.append(y)

            if (i + 1) % 20 == 0:
                print(f"Extracted embeddings batch {i+1}/{len(loader)}", flush=True)

    all_embs   = torch.cat(all_embs,   dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    print(f"Extracted embeddings: {all_embs.shape}", flush=True)

    # Save embeddings
    emb_df = pd.DataFrame(
        all_embs,
        columns=[f"clip_emb_{i}" for i in range(all_embs.shape[1])]
    )
    emb_df.insert(0, "NAKO_ID", df["NAKO_ID"].values)
    emb_df.insert(1, "label",   all_labels)
    emb_df.to_csv(OUT_DIR / "clip_image_embeddings.csv", index=False)

    # MLP classifier
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.net(x)

    X_train = torch.tensor(all_embs[train_idx],   dtype=torch.float32)
    y_train = torch.tensor(all_labels[train_idx], dtype=torch.long)
    X_val   = torch.tensor(all_embs[val_idx],     dtype=torch.float32)
    y_val   = torch.tensor(all_labels[val_idx],   dtype=torch.long)

    n0          = (y_train == 0).sum().item()
    n1          = (y_train == 1).sum().item()
    cls_weights = torch.tensor([1.0/n0, 1.0/n1], dtype=torch.float32).to(device)
    cls_weights = cls_weights / cls_weights.sum() * 2

    mlp       = MLPClassifier(args.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = torch.optim.AdamW(mlp.parameters(),
                                  lr=args.lr_cls, weight_decay=1e-4)

    best_auc          = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs_cls):
        mlp.train()
        optimizer.zero_grad()
        logits = mlp(X_train.to(device))
        loss   = criterion(logits, y_train.to(device))
        loss.backward()
        optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(X_val.to(device))
            val_prob   = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_pred   = torch.argmax(val_logits, dim=1).cpu().numpy()

        auc     = roc_auc_score(y_val.numpy(), val_prob)
        bal_acc = balanced_accuracy_score(y_val.numpy(), val_pred)

        print(
            f"Epoch {epoch+1}/{args.epochs_cls} | "
            f"train_loss={loss.item():.4f} | "
            f"val_auc={auc:.4f} | "
            f"val_bal_acc={bal_acc:.4f}",
            flush=True,
        )

        if auc > best_auc:
            best_auc          = auc
            epochs_no_improve = 0
            torch.save(mlp.state_dict(), OUT_DIR / "best_mlp_classifier.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping.", flush=True)
                break

    # Final evaluation
    mlp.load_state_dict(
        torch.load(OUT_DIR / "best_mlp_classifier.pt", weights_only=False)
    )
    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(X_val.to(device))
        val_prob   = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_pred   = torch.argmax(val_logits, dim=1).cpu().numpy()

    final_auc     = roc_auc_score(y_val.numpy(), val_prob)
    final_bal_acc = balanced_accuracy_score(y_val.numpy(), val_pred)
    cm            = confusion_matrix(y_val.numpy(), val_pred)

    print("\n" + "=" * 80, flush=True)
    print("FINAL T2D CLASSIFICATION (CLIP image embeddings)", flush=True)
    print(f"Cohort:            {args.cohort}", flush=True)
    print(f"Best AUC:          {best_auc:.4f}", flush=True)
    print(f"Final AUC:         {final_auc:.4f}", flush=True)
    print(f"Balanced accuracy: {final_bal_acc:.4f}", flush=True)
    print(f"Confusion matrix:\n{cm}", flush=True)
    print("=" * 80, flush=True)

    pd.DataFrame([{
        "cohort":         args.cohort,
        "input_mode":     args.input_mode,
        "n_tabular":      len(TABULAR_COLS),
        "best_clip_loss": ckpt["best_val_loss"],
        "best_auc":       best_auc,
        "final_auc":      final_auc,
        "final_bal_acc":  final_bal_acc,
    }]).to_csv(OUT_DIR / "clip_t2d_results.csv", index=False)
    print(f"Saved: {OUT_DIR / 'clip_t2d_results.csv'}", flush=True)

# ---------------------------------------------------------------------
# Phase 3: Fine-tuning
# ---------------------------------------------------------------------

def finetune_t2d(df, train_idx, val_idx, best_ckpt, device):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 3: END-TO-END FINE-TUNING ON T2D LABELS", flush=True)
    print("=" * 80, flush=True)

    in_channels = {
        "fat": 1, "fat_water": 2,
        "seg_trunk_binary": 1, "seg_trunk_label": 1,
        "fat_water_seg_trunk_binary": 3,
        "fat_water_seg_trunk_label":  3,
    }[args.input_mode]

    # Load CLIP model
    image_encoder   = ResNet10_3D(in_channels=in_channels,
                                  embedding_dim=args.embedding_dim)
    tabular_encoder = TabularEncoder(n_features=len(TABULAR_COLS),
                                     embedding_dim=args.embedding_dim)
    clip_model      = CLIPModel(image_encoder,
                                tabular_encoder,args.embedding_dim).to(device)

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    clip_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded CLIP from epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['best_val_loss']:.4f}", flush=True)

    # Fine-tuning classifier on top of image encoder
    class FineTuneModel(nn.Module):
        def __init__(self, clip_model, embedding_dim):
            super().__init__()
            self.image_encoder = clip_model.image_encoder
            self.image_proj    = clip_model.image_proj
            self.classifier    = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            _, emb = self.image_encoder(x)
            emb    = F.normalize(self.image_proj(emb), dim=1)
            return self.classifier(emb)

    ft_model = FineTuneModel(clip_model, args.embedding_dim).to(device)

    # Use image-only dataset for fine-tuning
    eval_cache = MATCHED_CACHE if args.cohort == "full" else CACHE_DIR
    dataset    = CLIPDataset(df, eval_cache, args.input_mode, TABULAR_COLS)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=16, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=16, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Class weights
    y_train     = torch.tensor(df["label"].iloc[train_idx].values)
    n0          = (y_train == 0).sum().item()
    n1          = (y_train == 1).sum().item()
    cls_weights = torch.tensor([1.0/n0, 1.0/n1],
                                dtype=torch.float32).to(device)
    cls_weights = cls_weights / cls_weights.sum() * 2

    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=args.lr_ft, weight_decay=1e-4
    )

    best_auc          = -1.0
    epochs_no_improve = 0
    ft_ckpt           = OUT_DIR / "best_finetune_model.pt"

    for epoch in range(args.epochs_ft):
        ft_model.train()
        train_losses = []
        for x, tab, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = ft_model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        ft_model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, tab, y in val_loader:
                x = x.to(device)
                logits = ft_model(x)
                prob   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred   = torch.argmax(logits, dim=1).cpu().numpy()
                all_probs.extend(prob)
                all_preds.extend(pred)
                all_labels.extend(y.numpy())

        auc     = roc_auc_score(all_labels, all_probs)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        print(
            f"Epoch {epoch+1}/{args.epochs_ft} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_auc={auc:.4f} | "
            f"val_bal_acc={bal_acc:.4f}",
            flush=True,
        )

        if auc > best_auc:
            best_auc          = auc
            epochs_no_improve = 0
            torch.save(ft_model.state_dict(), ft_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping.", flush=True)
                break

    # Final evaluation
    ft_model.load_state_dict(
        torch.load(ft_ckpt, map_location=device, weights_only=False)
    )
    ft_model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, tab, y in val_loader:
            x      = x.to(device)
            logits = ft_model(x)
            prob   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred   = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)
            all_labels.extend(y.numpy())

    final_auc     = roc_auc_score(all_labels, all_probs)
    final_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm            = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 80, flush=True)
    print("FINAL FINE-TUNING RESULTS", flush=True)
    print(f"Best AUC:          {best_auc:.4f}", flush=True)
    print(f"Final AUC:         {final_auc:.4f}", flush=True)
    print(f"Balanced accuracy: {final_bal_acc:.4f}", flush=True)
    print(f"Confusion matrix:\n{cm}", flush=True)
    print("=" * 80, flush=True)

    pd.DataFrame([{
        "cohort":        args.cohort,
        "input_mode":    args.input_mode,
        "phase":         "finetune",
        "best_auc":      best_auc,
        "final_auc":     final_auc,
        "final_bal_acc": final_bal_acc,
    }]).to_csv(OUT_DIR / "finetune_t2d_results.csv", index=False)
    print(f"Saved: {OUT_DIR / 'finetune_t2d_results.csv'}", flush=True)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}", flush=True)
    print(f"Cohort:  {args.cohort}", flush=True)
    print(f"Cache:   {CACHE_DIR}", flush=True)

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_DIR}\n"
            f"Run the cache script first."
        )

    # ============================================================
    # CASE 1: matched-only CLIP
    # Train CLIP on matched_train, validate on matched_val,
    # then train/evaluate T2D classifier on matched embeddings.
    # ============================================================
    if args.cohort == "matched":
        df = load_matched_data()

        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )

        mean, std = fit_tabular_scaler(df.iloc[train_idx])
        df = apply_tabular_scaler(df, mean, std)

        if args.phase in ["train", "both"]:
            best_ckpt = train_clip(df, train_idx, val_idx, device)

        if args.phase in ["evaluate", "both"]:
            best_ckpt = OUT_DIR / "best_clip_model.pt"
            evaluate_t2d(df, train_idx, val_idx, best_ckpt, device)
            finetune_t2d(df, train_idx, val_idx, best_ckpt, device)
            
    # ============================================================
    # CASE 2: full pretrain -> matched evaluation
    # Train CLIP on full cohort excluding matched validation subjects.
    # Then evaluate frozen image embeddings on matched train/val.
    # ============================================================
    elif args.cohort == "full":
        print("=" * 80, flush=True)
        print("FULL PRETRAIN -> MATCHED EVALUATION MODE", flush=True)
        print("=" * 80, flush=True)

        # 1. Load matched cohort for final T2D evaluation
        matched_df = load_matched_data()

        matched_train_idx, matched_val_idx = train_test_split(
            np.arange(len(matched_df)),
            test_size=0.2,
            random_state=42,
            stratify=matched_df["label"],
        )

        matched_val_ids = matched_df.iloc[matched_val_idx]["NAKO_ID"].astype(int).values

        print(f"Matched train subjects: {len(matched_train_idx)}", flush=True)
        print(f"Matched val subjects:   {len(matched_val_idx)}", flush=True)

        # 2. Load full cohort, but remove matched validation subjects
        pretrain_df = load_full_data_excluding(exclude_ids=matched_val_ids)

        # 3. Split full cohort into CLIP train/val
        pretrain_train_idx, pretrain_val_idx = train_test_split(
            np.arange(len(pretrain_df)),
            test_size=0.1,
            random_state=42,
            stratify=pretrain_df["label"],
        )

        print(f"CLIP pretrain train subjects: {len(pretrain_train_idx)}", flush=True)
        print(f"CLIP pretrain val subjects:   {len(pretrain_val_idx)}", flush=True)

        # 4. Normalize tabular features using full-pretrain train statistics
        mean, std = fit_tabular_scaler(pretrain_df.iloc[pretrain_train_idx])

        pretrain_df = apply_tabular_scaler(pretrain_df, mean, std)
        matched_df = apply_tabular_scaler(matched_df, mean, std)

        # 5. Train CLIP on full cohort excluding matched validation subjects
        if args.phase in ["train", "both"]:
            best_ckpt = train_clip(
                pretrain_df,
                pretrain_train_idx,
                pretrain_val_idx,
                device,
            )

        # 6. Evaluate T2D on matched train/validation embeddings
        if args.phase in ["evaluate", "both"]:
            best_ckpt = OUT_DIR / "best_clip_model.pt"
            evaluate_t2d(
                matched_df,
                matched_train_idx,
                matched_val_idx,
                best_ckpt,
                device,
            )
            finetune_t2d(matched_df, matched_train_idx,    
                         matched_val_idx, best_ckpt, device)
    else:
        raise ValueError(f"Unknown cohort: {args.cohort}")


if __name__ == "__main__":
    main()