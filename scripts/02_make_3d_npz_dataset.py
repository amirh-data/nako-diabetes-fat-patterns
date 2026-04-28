from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom


CASE_LIST = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/encoder_3d_cases_balanced_2000.csv"
)

OUTPUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/npz_96_balanced_2000"
)

TARGET_SHAPE = (96, 96, 96)  # z, y, x
MAX_CASES = None  # start small for testing


def load_nifti(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return arr


def resize_3d(arr, target_shape):
    factors = (
        target_shape[0] / arr.shape[0],
        target_shape[1] / arr.shape[1],
        target_shape[2] / arr.shape[2],
    )
    return zoom(arr, factors, order=1)


def resize_mask(mask, target_shape):
    factors = (
        target_shape[0] / mask.shape[0],
        target_shape[1] / mask.shape[1],
        target_shape[2] / mask.shape[2],
    )
    return zoom(mask, factors, order=0)


def normalize_fat(arr):
    arr = arr.astype(np.float32)
    arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
    mean = arr.mean()
    std = arr.std()
    if std > 0:
        arr = (arr - mean) / std
    return arr.astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CASE_LIST)

    if MAX_CASES is not None:
        df = df.head(MAX_CASES)

    print(f"Preparing {len(df)} cases")
    print(f"Output: {OUTPUT_DIR}")

    rows = []

    for i, row in df.iterrows():
        nako_id = row["NAKO_ID"]
        label = int(row["label"])
        fat_path = Path(row["fat_path"])
        seg_path = Path(row["seg_path"])

        out_file = OUTPUT_DIR / f"{nako_id}.npz"

        if out_file.exists():
            print(f"[SKIP] {nako_id}")
            rows.append({
                "NAKO_ID": nako_id,
                "label": label,
                "npz_path": out_file,
            })
            continue

        print(f"[{i+1}/{len(df)}] Processing {nako_id}")

        fat = load_nifti(fat_path)
        seg = load_nifti(seg_path)

        fat = resize_3d(fat, TARGET_SHAPE)
        seg = resize_mask(seg, TARGET_SHAPE)

        fat = normalize_fat(fat)

        # For first experiment: use binary segmentation mask
        seg = (seg > 0).astype(np.float32)

        # Shape: channels, z, y, x
        x = np.stack([fat, seg], axis=0).astype(np.float32)

        np.savez_compressed(
            out_file,
            x=x,
            y=np.array(label, dtype=np.int64),
            nako_id=str(nako_id),
        )

        rows.append({
            "NAKO_ID": nako_id,
            "label": label,
            "npz_path": out_file,
        })

    out_csv = OUTPUT_DIR / "prepared_npz_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nDone.")
    print(f"Saved prepared case list to:\n{out_csv}")


if __name__ == "__main__":
    main()
