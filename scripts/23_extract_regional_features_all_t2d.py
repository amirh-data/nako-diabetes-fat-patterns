from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom


ANALYSIS_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

FAT_IMAGE_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_repaired_input"
)

SEGMENTATION_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_full_output"
)

OUT_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/regional_pixel_features/regional_features_all_normal_t2d.csv"
)

TARGET_SHAPE = (96, 96, 96)

Z_BINS = 8
Y_BINS = 4
X_BINS = 4

MAX_CASES = None


def load_nifti(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def resize_3d(arr, target_shape, order):
    factors = (
        target_shape[0] / arr.shape[0],
        target_shape[1] / arr.shape[1],
        target_shape[2] / arr.shape[2],
    )
    return zoom(arr, factors, order=order)


def normalize_fat(arr):
    arr = arr.astype(np.float32)
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)

    mean = arr.mean()
    std = arr.std()

    if std > 0:
        arr = (arr - mean) / std

    return arr.astype(np.float32)


def get_slices(size, n_bins):
    edges = np.linspace(0, size, n_bins + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_bins)]


def extract_region_features(fat, mask):
    z_slices = get_slices(fat.shape[0], Z_BINS)
    y_slices = get_slices(fat.shape[1], Y_BINS)
    x_slices = get_slices(fat.shape[2], X_BINS)

    features = {}

    for zi, zs in enumerate(z_slices):
        for yi, ys in enumerate(y_slices):
            for xi, xs in enumerate(x_slices):
                region_fat = fat[zs, ys, xs]
                region_mask = mask[zs, ys, xs]

                name = f"z{zi}_y{yi}_x{xi}"

                features[f"{name}_mask_fraction"] = float(region_mask.mean())

                if region_mask.sum() > 0:
                    fat_inside = region_fat[region_mask > 0]
                    features[f"{name}_fat_mean_inside_mask"] = float(fat_inside.mean())
                    features[f"{name}_fat_std_inside_mask"] = float(fat_inside.std())
                else:
                    features[f"{name}_fat_mean_inside_mask"] = 0.0
                    features[f"{name}_fat_std_inside_mask"] = 0.0

    features["global_mask_fraction"] = float(mask.mean())
    features["global_fat_mean"] = float(fat.mean())
    features["global_fat_std"] = float(fat.std())

    return features


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

    # Clean normal-vs-T2D task
    df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()
    df["label"] = df["diabetes_group"].map({"normal": 0, "T2D": 1})

    df["fat_path"] = df["NAKO_ID"].apply(
        lambda x: FAT_IMAGE_DIR / f"NAKO751_{int(x)}_0000.nii.gz"
    )
    df["seg_path"] = df["NAKO_ID"].apply(
        lambda x: SEGMENTATION_DIR / f"NAKO751_{int(x)}.nii.gz"
    )

    df["fat_exists"] = df["fat_path"].apply(lambda p: p.exists())
    df["seg_exists"] = df["seg_path"].apply(lambda p: p.exists())

    df = df[df["fat_exists"] & df["seg_exists"]].copy()
    df = df.reset_index(drop=True)

    if MAX_CASES is not None:
        df = df.head(MAX_CASES)

    print("Cases:", df.shape)
    print("Label counts:")
    print(df["label"].value_counts())

    rows = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing {i}/{len(df)}", flush=True)

        nako_id = int(row["NAKO_ID"])
        label = int(row["label"])

        fat = load_nifti(row["fat_path"])
        seg = load_nifti(row["seg_path"])

        fat = resize_3d(fat, TARGET_SHAPE, order=1)
        seg = resize_3d(seg, TARGET_SHAPE, order=0)

        fat = normalize_fat(fat)
        mask = (seg > 0).astype(np.float32)

        feats = extract_region_features(fat, mask)
        feats["NAKO_ID"] = nako_id
        feats["label"] = label

        rows.append(feats)

    out = pd.DataFrame(rows)

    first_cols = ["NAKO_ID", "label"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    out.to_csv(OUT_CSV, index=False)

    print("\nSaved to:")
    print(OUT_CSV)
    print("Shape:", out.shape)


if __name__ == "__main__":
    main()
