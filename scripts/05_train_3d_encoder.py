from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.dataset_3d import Nako3DDataset
from src.models.cnn3d_mlp import CNN3DMLPStrongOverfit


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_balanced_t2d_3000_labels/prepared_npz_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/checkpoints_t2d_3000_labels"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # Load full dataset
    dataset = Nako3DDataset(CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("Loaded dataset:", len(dataset), flush=True)
    print("Full label counts:")
    print(df["label"].value_counts(), flush=True)

    # Stratified train/validation split
    train_idx, val_idx = train_test_split(
        list(range(len(df))),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    print("Train size:", len(train_ds), "Val size:", len(val_ds), flush=True)
    print("Train label counts:")
    print(df.iloc[train_idx]["label"].value_counts(), flush=True)
    print("Val label counts:")
    print(df.iloc[val_idx]["label"].value_counts(), flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN3DMLPStrongOverfit().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0,
    )

    num_epochs = 30

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, emb = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

            if i % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Batch {i}/{len(train_loader)} | "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits, emb = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "cnn3d_mlp_best.pt")
            print(f"Saved new best model with val_acc={best_val_acc:.4f}", flush=True)

    # Save final model
    torch.save(model.state_dict(), OUT_DIR / "cnn3d_mlp_final.pt")

    print("Training finished.", flush=True)
    print(f"Best val_acc: {best_val_acc:.4f}", flush=True)
    print(f"Saved final model to: {OUT_DIR / 'cnn3d_mlp_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
