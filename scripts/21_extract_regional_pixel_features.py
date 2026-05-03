from pathlib import Path

import numpy as np
import pandas as pd


CASE_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_balanced_t2d_3000/prepared_npz_cases.csv"
)

OUT_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/regional_pixel_features/regional_pixel_features_t2d_3000.csv"
)

# 96 x 96 x 96 will be divided into:
# z: 8 regions
# y: 4 regions
# x: 4 regions
# total = 8 * 4 * 4 = 128 spatial clusters
Z_BINS = 8
Y_BINS = 4
X_BINS = 4


def get_slices(size, n_bins):
    edges = np.linspace(0, size, n_bins + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_bins)]


def extract_features_from_npz(npz_path):
    data = np.load(npz_path)

    x = data["x"]
    fat = x[0]  # z, y, x
    mask = x[1]  # z, y, x

    z_slices = get_slices(fat.shape[0], Z_BINS)
    y_slices = get_slices(fat.shape[1], Y_BINS)
    x_slices = get_slices(fat.shape[2], X_BINS)

    features = {}

    for zi, zs in enumerate(z_slices):
        for yi, ys in enumerate(y_slices):
            for xi, xs in enumerate(x_slices):
                region_fat = fat[zs, ys, xs]
                region_mask = mask[zs, ys, xs]

                region_name = f"z{zi}_y{yi}_x{xi}"

                mask_fraction = float(region_mask.mean())

                if region_mask.sum() > 0:
                    fat_inside = region_fat[region_mask > 0]
                    fat_mean_inside = float(fat_inside.mean())
                    fat_std_inside = float(fat_inside.std())
                else:
                    fat_mean_inside = 0.0
                    fat_std_inside = 0.0

                features[f"{region_name}_mask_fraction"] = mask_fraction
                features[f"{region_name}_fat_mean_inside_mask"] = fat_mean_inside
                features[f"{region_name}_fat_std_inside_mask"] = fat_std_inside

    # Global simple features
    features["global_mask_fraction"] = float(mask.mean())
    features["global_fat_mean"] = float(fat.mean())
    features["global_fat_std"] = float(fat.std())

    return features


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df_cases = pd.read_csv(CASE_CSV)

    print("Loaded cases:", df_cases.shape)
    print("Label counts:")
    print(df_cases["label"].value_counts())

    rows = []

    for i, row in df_cases.iterrows():
        if i % 100 == 0:
            print(f"Processing {i}/{len(df_cases)}", flush=True)

        npz_path = Path(row["npz_path"])

        feats = extract_features_from_npz(npz_path)
        feats["NAKO_ID"] = row["NAKO_ID"]
        feats["label"] = int(row["label"])

        rows.append(feats)

    df_out = pd.DataFrame(rows)

    # Put ID and label first
    first_cols = ["NAKO_ID", "label"]
    other_cols = [c for c in df_out.columns if c not in first_cols]
    df_out = df_out[first_cols + other_cols]

    df_out.to_csv(OUT_CSV, index=False)

    print("\nSaved regional pixel features to:")
    print(OUT_CSV)
    print("Shape:", df_out.shape)


if __name__ == "__main__":
    main()
