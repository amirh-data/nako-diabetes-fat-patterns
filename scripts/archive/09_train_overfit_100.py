from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset_3d import Nako3DDataset
from src.models.cnn3d_mlp import CNN3DMLPStrongOverfit


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_tiny_overfit_100/prepared_npz_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/checkpoints_overfit_100"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    dataset = Nako3DDataset(CSV_PATH)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN3DMLPStrongOverfit().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, emb = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(loader)
        train_acc = correct / total

        print(
            f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), OUT_DIR / "cnn3d_mlp_overfit_100.pt")
    print("Saved model.", flush=True)


if __name__ == "__main__":
    main()
