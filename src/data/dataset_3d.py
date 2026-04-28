from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Nako3DDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = Path(row["npz_path"])

        data = np.load(npz_path)
        x = data["x"].astype(np.float32)
        y = int(data["y"])

        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
