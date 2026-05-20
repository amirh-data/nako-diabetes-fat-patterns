from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset

REGRESSION_TARGETS = [
    "VAT_ml",
    "SAT_abd_ml",
    "SAT_tho_ml",
    "VAT_SAT_ratio",
    "BackMuscle_ml",
    "BackMuscle_ff_pct",
    "CardiacAT_ml",
    "Liver_ff_pct",
    "VertebralMarrow_mean_ff_pct",
]

NUM_TARGETS = len(REGRESSION_TARGETS)


class RegressionNPZDataset(Dataset):
    """
    Returns image + z-scored regression targets + mask.
    Used to train image encoder on VAT/SAT/tissue prediction.
    """

    def __init__(self, csv_path, augment=False):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment

        # compute z-score stats per target
        self.means = {}
        self.stds  = {}
        for col in REGRESSION_TARGETS:
            vals          = pd.to_numeric(self.df[col], errors="coerce")
            self.means[col] = float(vals.mean())
            self.stds[col]  = float(vals.std()) + 1e-8

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        npz_path = Path(row["npz_path"])
        data     = np.load(npz_path)
        x        = torch.tensor(data["x"], dtype=torch.float32)
        y        = torch.tensor(int(data["y"]), dtype=torch.long)

        if self.augment:
            transform = tio.Compose([
                tio.RandomFlip(axes=['LR']),
                tio.RandomFlip(axes=['AP']),
                tio.RandomAffine(scales=0.1, degrees=10, translation=5),
                tio.RandomBiasField(coefficients=0.3),
                tio.RandomNoise(std=0.02),
                tio.RandomGamma(log_gamma=0.2),
            ])
            subject     = tio.Subject(image=tio.ScalarImage(tensor=x))
            transformed = transform(subject)
            x           = transformed["image"].data
            x           = torch.clamp(x, 0.0, 1.0)

        # build regression targets
        targets = []
        mask    = []
        for col in REGRESSION_TARGETS:
            raw   = pd.to_numeric(row[col], errors="coerce")
            valid = not np.isnan(float(raw)) if raw == raw else False
            z     = (float(raw) - self.means[col]) / self.stds[col] if valid else 0.0
            targets.append(z)
            mask.append(valid)

        targets = torch.tensor(targets, dtype=torch.float32)
        mask    = torch.tensor(mask,    dtype=torch.bool)

        return x, targets, mask, y
