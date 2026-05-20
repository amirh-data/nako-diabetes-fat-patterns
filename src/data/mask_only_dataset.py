from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MaskOnlyDataset(Dataset):
    """
    Loads only the segmentation mask channel (channel 2)
    from fat_water_mask NPZ files.

    Input to model: (1, 96, 96, 96) binary mask
    Label: T2D (0/1)
    """

    def __init__(self, csv_path, augment=False):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, x):
        # only spatial flips — no intensity augmentation on binary mask
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])   # LR flip
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])   # AP flip
        return x

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        data     = np.load(Path(row["npz_path"]))
        x_full   = data["x"]              # (3, 96, 96, 96)

        # extract mask channel only
        mask     = x_full[2:3]            # (1, 96, 96, 96)
        x        = torch.tensor(mask, dtype=torch.float32)
        y        = torch.tensor(int(data["y"]), dtype=torch.long)

        if self.augment:
            x = self._augment(x)

        return x, y
