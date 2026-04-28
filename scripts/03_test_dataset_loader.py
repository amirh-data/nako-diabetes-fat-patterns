from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset_3d import Nako3DDataset


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96/prepared_npz_cases.csv"
)


def main():
    ds = Nako3DDataset(CSV_PATH)

    print("Dataset size:", len(ds))

    x, y = ds[0]
    print("Single x shape:", x.shape)
    print("Single y:", y)

    loader = DataLoader(ds, batch_size=2, shuffle=True)

    xb, yb = next(iter(loader))
    print("Batch x shape:", xb.shape)
    print("Batch y shape:", yb.shape)
    print("Batch y:", yb)


if __name__ == "__main__":
    main()
