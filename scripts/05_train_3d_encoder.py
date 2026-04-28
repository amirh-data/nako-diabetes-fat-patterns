from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset_3d import Nako3DDataset
from src.models.cnn3d_mlp import CNN3DMLP


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96/prepared_npz_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/checkpoints"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = Nako3DDataset(CSV_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = CNN3DMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, emb = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits, emb = model(x)
                pred = logits.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), OUT_DIR / "cnn3d_mlp_test.pt")
    print("Saved model.")


if __name__ == "__main__":
    main()
