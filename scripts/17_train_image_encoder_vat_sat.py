from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from src.data.dataset_3d import Nako3DDataset
from src.models.cnn3d_mlp import CNN3DMLPStrongOverfit


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_vat_sat_extremes/prepared_npz_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/checkpoints_vat_sat_96"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    dataset = Nako3DDataset(CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("Dataset size:", len(dataset), flush=True)
    print("Label counts:")
    print(df["label"].value_counts(), flush=True)

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model = CNN3DMLPStrongOverfit().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, emb = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(train_loader)} | loss={loss.item():.4f}",
                    flush=True,
                )

        train_loss /= len(train_loader)

        # validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{num_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), OUT_DIR / "cnn3d_vatsat.pt")
    print("Saved model.", flush=True)


if __name__ == "__main__":
    main()
