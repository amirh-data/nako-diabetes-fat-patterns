from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.regression_npz_dataset import RegressionNPZDataset, NUM_TARGETS
from src.models.vatsat_encoder import build_vatsat_encoder

parser = argparse.ArgumentParser()
parser.add_argument("--input_mode", type=str, default="fat_water")
parser.add_argument("--model_name", type=str, default="resnet10_3d")
parser.add_argument("--epochs",     type=int, default=50)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers",type=int, default=4)
parser.add_argument("--resume", action="store_true",
                    help="Resume from best_encoder.pt checkpoint")
parser.add_argument("--augment",    action="store_true")
args = parser.parse_args()

CSV_PATH = Path(
    f"/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/supervised_image_datasets/npz_96_{args.input_mode}_t2d_3000/prepared_npz_with_vatsat.csv"
)

OUT_DIR = Path(
    f"/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/vatsat_encoder/{args.model_name}_{args.input_mode}"
)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion  = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for x, targets, mask, _ in loader:
            x, targets, mask = x.to(device), targets.to(device), mask.to(device)
            _, emb, reg_out  = model(x)
            loss_raw         = criterion(reg_out, targets)
            masked           = loss_raw * mask.float()
            denom            = mask.float().sum().clamp(min=1)
            total_loss      += (masked.sum() / denom).item()

    return total_loss / len(loader)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",     device,     flush=True)
    print("Model:",      args.model_name, flush=True)
    print("Input mode:", args.input_mode, flush=True)
    print("NUM_TARGETS:", NUM_TARGETS,   flush=True)

    df = pd.read_csv(CSV_PATH)
    print("Dataset:", len(df), "cases", flush=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_ds = Subset(RegressionNPZDataset(CSV_PATH, augment=args.augment), train_idx)
    val_ds   = Subset(RegressionNPZDataset(CSV_PATH, augment=False),        val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # model with regression head
    sample_x, _, _, _ = RegressionNPZDataset(CSV_PATH, augment=False)[0]
    in_channels = sample_x.shape[0]
    print("Input channels:", in_channels, flush=True)

    model = build_vatsat_encoder(
        model_name  = args.model_name,
        in_channels = in_channels,
        num_targets = NUM_TARGETS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    best_path = OUT_DIR / "best_encoder.pt"
    if args.resume and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Resumed from: {best_path}", flush=True)
    elif args.resume:
        print("WARNING: --resume specified but no checkpoint found. Starting fresh.", flush=True)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_path     = OUT_DIR / "best_encoder.pt"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for i, (x, targets, mask, _) in enumerate(train_loader):
            x, targets, mask = x.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()
            _, emb, reg_out = model(x)

            loss_raw = criterion(reg_out, targets)
            masked   = loss_raw * mask.float()
            denom    = mask.float().sum().clamp(min=1)
            loss     = masked.sum() / denom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {i}/{len(train_loader)} | "
                      f"loss={loss.item():.4f}", flush=True)

        train_loss /= len(train_loader)
        val_loss    = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best encoder: val_loss={best_val_loss:.4f}", flush=True)

    torch.save(model.state_dict(), OUT_DIR / "final_encoder.pt")
    print("Done. Best val_loss:", best_val_loss, flush=True)


if __name__ == "__main__":
    main()
