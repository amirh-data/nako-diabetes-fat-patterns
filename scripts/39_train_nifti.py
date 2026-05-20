"""
scripts/39_train_nifti.py

Unified direct NIfTI training script.

Supports:
    input_mode:
        fat
        fat_water
        seg
        seg_fat_water
        seg_trunk_binary
        seg_trunk_label
        fat_water_seg_trunk_binary
        fat_water_seg_trunk_label
        fat_water_trunk_masked
        fat_water_trunk_masked_seg_binary
        fat_water_trunk_masked_seg_label
        
    model_name:
        resnet10_3d
        convnext3d

    cohort:
        matched
        full

    feature_mode:
        none
        clinical
        biomarkers
        clinical_biomarkers

Examples:
    python -u -m scripts.39_train_nifti \
        --input_mode fat_water \
        --model_name resnet10_3d \
        --cohort matched \
        --feature_mode none

    python -u -m scripts.39_train_nifti \
        --input_mode seg \
        --model_name resnet10_3d \
        --cohort matched \
        --feature_mode none

    python -u -m scripts.39_train_nifti \
        --input_mode fat_water \
        --model_name resnet10_3d \
        --cohort matched \
        --feature_mode clinical_biomarkers
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data.nifti_dataset import (
    NiftiDataset,
    CachedNiftiDataset,
    load_nifti_dataset,
    get_feature_cols,
)
from src.models.supervised_3d_models import ResNet10_3D, ConvNeXt3D


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_mode",
    type=str,
    default="fat_water",
    choices=[
        "fat",
        "fat_water",
        "seg",
        "seg_fat_water",
        "seg_trunk_binary",
        "seg_trunk_label",
        "fat_water_seg_trunk_binary",
        "fat_water_seg_trunk_label",
        "fat_water_trunk_masked",
        "fat_water_trunk_masked_seg_binary",
        "fat_water_trunk_masked_seg_label",
        "fat_water_body_masked",
        "fat_water_body_masked_seg_binary",
        "fat_water_body_masked_seg_label",
    ],
)

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet10_3d",
    choices=["resnet10_3d", "convnext3d"],
)

parser.add_argument(
    "--feature_mode",
    type=str,
    default="none",
    choices=["none", "clinical", "biomarkers", "distribution",
         "biomarkers_distribution", "clinical_biomarkers"],
)

parser.add_argument(
    "--cohort",
    type=str,
    default="matched",
    choices=["matched", "full"],
)

parser.add_argument(
    "--cache_dir",
    type=str,
    default="/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/cached_nifti128/matched_fat_water_trunkseg",
)

parser.add_argument(
    "--data_backend",
    type=str,
    default="nifti",
    choices=["nifti", "cached"],
)

parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--val_every", type=int, default=5)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--augment", action="store_true")
parser.add_argument("--accum_steps", type=int, default=1)
parser.add_argument("--overfit_n", type=int, default=0)
parser.add_argument("--resume_from", type=str, default="")

args = parser.parse_args()


# ---------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------

BASE_OUT = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/"
    "supervised_image_results"
)

aug_tag = "aug" if args.augment else "noaug"

OUT_DIR = BASE_OUT / (
    f"nifti{args.resolution}_"
    f"{args.data_backend}_"
    f"{args.model_name}_"
    f"{args.input_mode}_"
    f"{args.cohort}_"
    f"features-{args.feature_mode}_"
    f"{aug_tag}"
)

if args.overfit_n > 0:
    OUT_DIR = OUT_DIR.parent / (OUT_DIR.name + f"_overfit{args.overfit_n}")
    


# ---------------------------------------------------------------------
# Model wrapper for image + tabular features
# ---------------------------------------------------------------------

class ImageTabularClassifier(nn.Module):
    """
    Wraps image encoder and combines image embedding with tabular features.
    """

    def __init__(self, image_model, embedding_dim, n_features, n_classes=2):
        super().__init__()

        self.image_model = image_model

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + n_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, features=None):
        if features is None:
            raise ValueError("features must be provided for ImageTabularClassifier")

        _, emb = self.image_model(x)
        combined = torch.cat([emb, features], dim=1)
        logits = self.classifier(combined)

        return logits, combined


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, loader, device, feature_cols=None, split_name="VAL"):
    feature_cols = feature_cols or []

    model.eval()

    all_y = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for batch in loader:
            if len(feature_cols) > 0:
                x, features, y = batch
                x = x.to(device)
                features = features.to(device)
                logits, _ = model(x, features)
            else:
                x, y = batch
                x = x.to(device)
                logits, _ = model(x)

            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            all_y.extend(y.numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    auc = roc_auc_score(all_y, all_prob)
    bal_acc = balanced_accuracy_score(all_y, all_pred)
    cm = confusion_matrix(all_y, all_pred)

    normal_probs = all_prob[all_y == 0]
    t2d_probs = all_prob[all_y == 1]

    print(
        f"  [{split_name}] prob_std={all_prob.std():.4f} | "
        f"mean_normal={normal_probs.mean():.4f} | "
        f"mean_T2D={t2d_probs.mean():.4f}",
        flush=True,
    )

    return auc, bal_acc, cm, all_y, all_pred


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_cols = get_feature_cols(args.feature_mode)

    in_channels = {
    "fat": 1,
    "fat_water": 2,
    "seg": 1,
    "seg_fat_water": 3,
    "seg_trunk_binary": 1,
    "seg_trunk_label": 1,
    "fat_water_seg_trunk_binary": 3,
    "fat_water_seg_trunk_label": 3,
    "fat_water_trunk_masked": 2,
    "fat_water_trunk_masked_seg_binary": 3,
    "fat_water_trunk_masked_seg_label": 3,
    "fat_water_body_masked": 2,
    "fat_water_body_masked_seg_binary": 3,
    "fat_water_body_masked_seg_label": 3,
    }[args.input_mode]

    print("=" * 80, flush=True)
    print("UNIFIED NIFTI TRAINING", flush=True)
    print(f"Input mode:     {args.input_mode}", flush=True)
    print(f"Input channels: {in_channels}", flush=True)
    print(f"Model:          {args.model_name}", flush=True)
    print(f"Cohort:         {args.cohort}", flush=True)
    print(f"Feature mode:   {args.feature_mode}", flush=True)
    print(f"Feature cols:   {feature_cols}", flush=True)
    print(f"Data backend:    {args.data_backend}", flush=True)
    print(f"Cache dir:       {args.cache_dir}", flush=True)
    print(f"Accum steps:     {args.accum_steps}", flush=True)
    print(f"Overfit N:       {args.overfit_n}", flush=True)
    print(f"Resolution:     {args.resolution}³", flush=True)
    print(f"Epochs:         {args.epochs}", flush=True)
    print(f"LR:             {args.lr}", flush=True)
    print(f"Batch size:     {args.batch_size}", flush=True)
    print(f"Augment:        {args.augment}", flush=True)
    print(f"Device:         {device}", flush=True)
    print(f"Output:         {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    load_input_mode = args.input_mode

    if args.data_backend == "cached":
        load_input_mode = "seg_fat_water"
    
    df = load_nifti_dataset(
        cohort=args.cohort,
        input_mode=load_input_mode,
        feature_mode=args.feature_mode,
    )
    
    if args.data_backend == "cached":
        cache_dir = Path(args.cache_dir)

        before = len(df)
        df = df[
            df["NAKO_ID"].apply(
                lambda x: (cache_dir / f"NAKO751_{int(x)}.pt").exists()
            )
        ].copy()

        print(
            f"Cached backend: {len(df)}/{before} subjects have cached .pt files",
            flush=True,
        )

        if len(df) == 0:
            raise RuntimeError(f"No cached files found in {cache_dir}")

    if args.overfit_n > 0:
    # take balanced small subset if possible
        df0 = df[df["label"] == 0].sample(
            n=args.overfit_n // 2,
            random_state=42,
        )
        df1 = df[df["label"] == 1].sample(
            n=args.overfit_n - len(df0),
            random_state=42,
        )

        df = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)

        print("=" * 80, flush=True)
        print(f"OVERFIT DEBUG MODE: using only {len(df)} samples", flush=True)
        print(df["label"].value_counts().to_dict(), flush=True)
        print("=" * 80, flush=True)
    
    if args.overfit_n > 0:
        train_idx = np.arange(len(df))
        val_idx = np.arange(len(df))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )

    # Normalize tabular features using TRAIN statistics only.
    if len(feature_cols) > 0:
        train_mean = df.iloc[train_idx][feature_cols].mean()
        train_std = df.iloc[train_idx][feature_cols].std().replace(0, 1)

        df.loc[:, feature_cols] = (df[feature_cols] - train_mean) / train_std

        scaler_df = pd.DataFrame({
            "feature": feature_cols,
            "mean": train_mean.values,
            "std": train_std.values,
        })
        scaler_df.to_csv(OUT_DIR / "feature_scaler.csv", index=False)

        print("\nSaved feature scaler:", OUT_DIR / "feature_scaler.csv", flush=True)

    if args.data_backend == "cached":
        dataset_train_full = CachedNiftiDataset(
            df=df,
            cache_dir=args.cache_dir,
            input_mode=args.input_mode,
            feature_mode=args.feature_mode,
            augment=args.augment,
        )

        dataset_val_full = CachedNiftiDataset(
            df=df,
            cache_dir=args.cache_dir,
            input_mode=args.input_mode,
            feature_mode=args.feature_mode,
            augment=False,
        )

    else:
        dataset_train_full = NiftiDataset(
            df=df,
            input_mode=args.input_mode,
            feature_mode=args.feature_mode,
            resolution=args.resolution,
            augment=args.augment,
        )

        dataset_val_full = NiftiDataset(
            df=df,
            input_mode=args.input_mode,
            feature_mode=args.feature_mode,
            resolution=args.resolution,
            augment=False,
        )

    train_ds = Subset(dataset_train_full, train_idx)
    val_ds = Subset(dataset_val_full, val_idx)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)}", flush=True)
    print(
        "Train labels:",
        df.iloc[train_idx]["label"].value_counts().to_dict(),
        flush=True,
    )
    print(
        "Val labels:",
        df.iloc[val_idx]["label"].value_counts().to_dict(),
        flush=True,
    )

    # -----------------------------------------------------------------
    # Sampler
    # -----------------------------------------------------------------

    train_labels = df.iloc[train_idx]["label"].values
    class_counts = np.bincount(train_labels)

    print(f"Train class counts: {class_counts}", flush=True)

    sample_weights = (1.0 / class_counts)[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    if args.overfit_n > 0:
        print("OVERFIT MODE: using shuffle=True, no WeightedRandomSampler", flush=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    if args.model_name == "resnet10_3d":
        image_model = ResNet10_3D(
            in_channels=in_channels,
            embedding_dim=args.embedding_dim,
        )

    elif args.model_name == "convnext3d":
        image_model = ConvNeXt3D(
            in_channels=in_channels,
            embedding_dim=args.embedding_dim,
        )

    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    n_features = len(feature_cols)

    if n_features > 0:
        model = ImageTabularClassifier(
            image_model=image_model,
            embedding_dim=args.embedding_dim,
            n_features=n_features,
        ).to(device)
    else:
        model = image_model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}", flush=True)
    
    # -------------------------
    # Resume / fine-tune from checkpoint
    # -------------------------
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=torch.device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model weights from: {args.resume_from}", flush=True)

    # -----------------------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------------------

    all_counts = np.bincount(df["label"].values)
    cls_weights = torch.tensor(
        1.0 / np.sqrt(all_counts),
        dtype=torch.float32,
    ).to(device)
    cls_weights = cls_weights / cls_weights.sum() * 2.0

    print(f"Loss class weights: {cls_weights.detach().cpu().numpy()}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    weight_decay = 0.0 if args.overfit_n > 0 else 1e-4

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=weight_decay,
    )

    print(f"Weight decay: {weight_decay}", flush=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    best_auc = 0.0
    best_path = OUT_DIR / "best_model.pt"
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            if len(feature_cols) > 0:
                x, features, y = batch
                x = x.to(device)
                features = features.to(device)
                y = y.to(device)

                logits, _ = model(x, features)

            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                logits, _ = model(x)

            loss = criterion(logits, y)

            # divide loss so accumulated gradient has correct scale
            loss_for_backward = loss / args.accum_steps
            loss_for_backward.backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        # Batch-level debug print.
        # Keep this commented out unless detailed batch monitoring is needed.
        # if i % 50 == 0:
        #     print(
        #         f"Epoch {epoch + 1}/{args.epochs} | "
        #         f"Batch {i}/{len(train_loader)} | "
        #         f"loss={loss.item():.4f}",
        #         flush=True,
        #     )

        scheduler.step()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f}",
            flush=True,
        )

        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_auc, val_bal_acc, cm, y_true, y_pred = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                feature_cols=feature_cols,
                split_name="VAL",
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"val_auc={val_auc:.4f} | "
                f"val_bal_acc={val_bal_acc:.4f}",
                flush=True,
            )
            print(f"Confusion matrix:\n{cm}", flush=True)

            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "best_auc": best_auc,
                        "epoch": epoch + 1,
                        "feature_cols": feature_cols,
                    },
                    best_path,
                )

                print(f"Saved best model: AUC={best_auc:.4f}", flush=True)

            else:
                epochs_no_improve += 1
                print(
                    f"No improvement {epochs_no_improve}/{args.patience}",
                    flush=True,
                )

                if epochs_no_improve >= args.patience:
                    print("Early stopping.", flush=True)
                    break

    # -----------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------

    final_path = OUT_DIR / "final_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "best_auc": best_auc,
            "feature_cols": feature_cols,
        },
        final_path,
    )

    val_auc, val_bal_acc, cm, y_true, y_pred = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        feature_cols=feature_cols,
        split_name="FINAL",
    )

    print("\nFinal validation:", flush=True)
    print(f"AUC:      {val_auc:.4f}", flush=True)
    print(f"Bal acc:  {val_bal_acc:.4f}", flush=True)
    print(f"Best AUC: {best_auc:.4f}", flush=True)
    print(cm, flush=True)
    print(classification_report(y_true, y_pred, digits=4), flush=True)

    results = pd.DataFrame([{
    "experiment": "nifti_direct_or_cached",
    "data_backend": args.data_backend,
    "input_mode": args.input_mode,
    "model": args.model_name,
    "cohort": args.cohort,
    "feature_mode": args.feature_mode,
    "n_features": len(feature_cols),
    "resolution": args.resolution,
    "augment": args.augment,
    "overfit_n": args.overfit_n,
    "val_auc": val_auc,
    "best_auc": best_auc,
    "val_bal_acc": val_bal_acc,
    "cache_dir": args.cache_dir if args.data_backend == "cached" else "",
    "out_dir": str(OUT_DIR),
    }])

    results.to_csv(OUT_DIR / "results.csv", index=False)

    print(f"\nSaved results: {OUT_DIR / 'results.csv'}", flush=True)
    print(f"Saved final model: {final_path}", flush=True)


if __name__ == "__main__":
    main()