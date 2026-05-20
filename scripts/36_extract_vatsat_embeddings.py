"""
scripts/36_extract_vatsat_embeddings.py

Extract 128-dim embeddings from trained VAT/SAT encoder
for matched cohort subjects.

Usage:
    python -u -m scripts.36_extract_vatsat_embeddings \
        --model_name convnext3d \
        --input_mode fat_water
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models.vatsat_encoder import build_vatsat_encoder
from src.data.supervised_npz_dataset import SupervisedNPZDataset

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",  type=str, default="convnext3d")
parser.add_argument("--input_mode",  type=str, default="fat_water")
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

ENCODER_PATH = BASE / f"vatsat_encoder/{args.model_name}_{args.input_mode}/best_encoder.pt"
NPZ_CSV      = BASE / f"supervised_image_datasets/npz_96_{args.input_mode}_t2d_3000/prepared_npz_matched_cohort.csv"
OUT_DIR      = BASE / f"vatsat_encoder/{args.model_name}_{args.input_mode}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:       {device}")
    print(f"Encoder:      {ENCODER_PATH}")
    print(f"NPZ CSV:      {NPZ_CSV}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    if not NPZ_CSV.exists():
        raise FileNotFoundError(f"NPZ CSV not found: {NPZ_CSV}")

    # Load dataset - uses matched cohort NPZ directly
    df      = pd.read_csv(NPZ_CSV)
    print(f"Subjects: {len(df)}")
    print(df["label"].value_counts())

    dataset = SupervisedNPZDataset(NPZ_CSV, augment=False)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build and load model
    in_channels = 2 if args.input_mode == "fat_water" else 1
    model       = build_vatsat_encoder(
        model_name=args.model_name,
        in_channels=in_channels,
        num_targets=9,
    )
    state_dict = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Extract embeddings
    all_embs   = []
    all_labels = []

    print("Extracting embeddings...", flush=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            _, emb, _ = model(x)
            all_embs.append(emb.cpu())
            all_labels.append(y)
            if i % 20 == 0:
                print(f"  Batch {i}/{len(loader)}", flush=True)

    all_embs   = torch.cat(all_embs,   dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    print(f"Embeddings shape: {all_embs.shape}")

    # Save with NAKO_IDs
    emb_cols = [f"vatsat_emb_{i}" for i in range(all_embs.shape[1])]
    emb_df   = pd.DataFrame(all_embs, columns=emb_cols)
    emb_df.insert(0, "NAKO_ID", df["NAKO_ID"].values)
    emb_df.insert(1, "label",   all_labels)

    out_path = OUT_DIR / "vatsat_embeddings_matched.csv"
    emb_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {emb_df.shape}")
    print(emb_df["label"].value_counts())


if __name__ == "__main__":
    main()
