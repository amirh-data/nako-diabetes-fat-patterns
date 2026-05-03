from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset_3d import Nako3DDataset
from src.models.cnn3d_mlp import CNN3DMLPStrongOverfit


# T2D/normal image dataset
CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_balanced_t2d_3000/prepared_npz_cases.csv"
)

# VAT/SAT-trained model
MODEL_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/checkpoints_vat_sat_96/cnn3d_vatsat.pt"
)

OUT_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/t2d_embeddings_from_vatsat_encoder.csv"
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    dataset = Nako3DDataset(CSV_PATH)
    df_cases = pd.read_csv(CSV_PATH)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN3DMLPStrongOverfit().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)

            logits, emb = model(x)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(y.numpy())

            if i % 50 == 0:
                print(f"Processed batch {i}/{len(loader)}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    emb_cols = [f"vatsat_emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols)

    df_out = pd.concat(
        [
            df_cases[["NAKO_ID"]].reset_index(drop=True),
            pd.Series(labels, name="label"),
            df_emb,
        ],
        axis=1,
    )

    df_out.to_csv(OUT_PATH, index=False)

    print("Saved embeddings:")
    print(OUT_PATH)
    print("Shape:", df_out.shape)


if __name__ == "__main__":
    main()
