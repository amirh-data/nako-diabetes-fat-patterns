from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset


class SupervisedNPZDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment

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
                tio.RandomAffine(
                    scales=0.1,
                    degrees=10,
                    translation=5,
                ),
                tio.RandomBiasField(coefficients=0.3),
                tio.RandomNoise(std=0.02),
                tio.RandomGamma(log_gamma=0.2),
            ])
            subject     = tio.Subject(image=tio.ScalarImage(tensor=x))
            transformed = transform(subject)
            x           = transformed["image"].data
            x           = torch.clamp(x, 0.0, 1.0)

        return x, y