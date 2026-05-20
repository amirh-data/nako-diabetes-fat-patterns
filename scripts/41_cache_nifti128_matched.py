"""
scripts/41_cache_nifti128_matched.py

Cache matched cohort NIfTI files to 128³ .pt tensors.

Each output .pt file contains:
    x: tensor [4, 128, 128, 128]

Channels:
    0 = fat
    1 = water
    2 = seg_trunk_binary
    3 = seg_trunk_label

Trunk labels used:
    1  SAT
    2  VAT
    4  Cardiac adipose tissue
    9  Back muscles
    14 Vertebral bodies
    16 Liver
    18 Pancreas
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F


BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

IMG_DIR = BASE / "dataset301_repaired_input"
SEG_DIR = BASE / "dataset301_full_output"

ANALYSIS_CSV = BASE / "quantification_full/analysis_table_with_diabetes_labels.csv"

OUT_DIR_DEFAULT = BASE / "cached_nifti128/full_fat_water_trunkseg"


# Original segmentation labels
TRUNK_LABELS = {
    1: "SAT",
    2: "VAT",
    4: "CardiacAT",
    9: "BackMuscle",
    14: "VertebralBodies",
    16: "Liver",
    18: "Pancreas",
}

# Relabel to compact 1–7 values
TRUNK_RELABEL = {
    1: 1,   # SAT
    2: 2,   # VAT
    4: 3,   # CardiacAT
    9: 4,   # BackMuscle
    14: 5,  # VertebralBodies
    16: 6,  # Liver
    18: 7,  # Pancreas
}


def load_nifti_arr(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def resize_volume(arr, target_shape=(128, 128, 128), is_seg=False):
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    if is_seg:
        t = F.interpolate(t, size=target_shape, mode="nearest")
    else:
        t = F.interpolate(
            t,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    return t.squeeze().numpy()


def normalize_01(arr):
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    arr = np.clip(arr, p1, p99)

    denom = p99 - p1
    if denom > 0:
        arr = (arr - p1) / denom
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.float32)


def make_trunk_masks(seg_raw):
    """
    seg_raw contains original labels 0–19.

    Returns:
        seg_trunk_binary: 0/1 mask for selected trunk labels
        seg_trunk_label: compact relabeled map normalized to [0,1]
                         0 = other/background
                         1–7 = selected trunk structures
    """
    seg_raw = np.rint(seg_raw).astype(np.int16)

    selected = np.isin(seg_raw, list(TRUNK_LABELS.keys()))
    seg_trunk_binary = selected.astype(np.float32)

    seg_relabel = np.zeros_like(seg_raw, dtype=np.float32)

    for original_label, new_label in TRUNK_RELABEL.items():
        seg_relabel[seg_raw == original_label] = float(new_label)

    # Normalize 0–7 to 0–1
    seg_trunk_label = seg_relabel / 7.0

    return seg_trunk_binary.astype(np.float32), seg_trunk_label.astype(np.float32)


def process_one(row_dict, out_dir, resolution, overwrite=False):
    nako_id = int(row_dict["NAKO_ID"])
    label = int(row_dict["label"])

    out_path = out_dir / f"NAKO751_{nako_id}.pt"

    if out_path.exists() and not overwrite:
        return {
            "NAKO_ID": nako_id,
            "status": "exists",
            "path": str(out_path),
        }

    fat_path = IMG_DIR / f"NAKO751_{nako_id}_0000.nii.gz"
    water_path = IMG_DIR / f"NAKO751_{nako_id}_0001.nii.gz"
    seg_path = SEG_DIR / f"NAKO751_{nako_id}.nii.gz"

    if not fat_path.exists():
        return {"NAKO_ID": nako_id, "status": "missing_fat"}
    if not water_path.exists():
        return {"NAKO_ID": nako_id, "status": "missing_water"}
    if not seg_path.exists():
        return {"NAKO_ID": nako_id, "status": "missing_seg"}

    target_shape = (resolution, resolution, resolution)

    try:
        fat = load_nifti_arr(fat_path)
        fat = resize_volume(fat, target_shape, is_seg=False)
        fat = normalize_01(fat)

        water = load_nifti_arr(water_path)
        water = resize_volume(water, target_shape, is_seg=False)
        water = normalize_01(water)

        seg_raw = load_nifti_arr(seg_path)
        seg_raw = resize_volume(seg_raw, target_shape, is_seg=True)

        seg_trunk_binary, seg_trunk_label = make_trunk_masks(seg_raw)

        x = np.stack(
            [
                fat,
                water,
                seg_trunk_binary,
                seg_trunk_label,
            ],
            axis=0,
        )

        # Save as float16 to reduce disk size.
        x = torch.tensor(x, dtype=torch.float16)

        sample = {
            "x": x,
            "label": label,
            "NAKO_ID": nako_id,
            "channel_info": {
                "0": "fat",
                "1": "water",
                "2": "seg_trunk_binary",
                "3": "seg_trunk_label",
            },
            "trunk_labels": TRUNK_LABELS,
            "trunk_relabel": TRUNK_RELABEL,
        }

        torch.save(sample, out_path)

        return {
            "NAKO_ID": nako_id,
            "status": "ok",
            "path": str(out_path),
        }

    except Exception as e:
        return {
            "NAKO_ID": nako_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("CACHE FULL COHORT NIFTI: FAT + WATER + TRUNK SEG", flush=True)
    print(f"Output dir:  {out_dir}", flush=True)
    print(f"Resolution:  {args.resolution}³", flush=True)
    print(f"Workers:     {args.num_workers}", flush=True)
    print("=" * 80, flush=True)

    # Load full cohort
    ana = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
    df = ana[ana["diabetes_group"].isin(["normal", "T2D"])].copy()
    df["label"] = (df["diabetes_group"] == "T2D").astype(int)
    df = df[["NAKO_ID", "label"]].copy()

    print(f"Total subjects: {len(df)}", flush=True)
    print(df["label"].value_counts().to_string(), flush=True)

    rows = df.to_dict("records")
    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                row,
                out_dir,
                args.resolution,
                args.overwrite,
            )
            for row in rows
        ]

        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)

            if i % 200 == 0 or res["status"] not in ["ok", "exists"]:
                print(
                    f"[{i}/{len(rows)}] "
                    f"NAKO_ID={res.get('NAKO_ID')} "
                    f"status={res.get('status')}",
                    flush=True,
                )

    results_df = pd.DataFrame(results)
    results_csv = out_dir / "cache_report.csv"
    results_df.to_csv(results_csv, index=False)

    print("\nDone.", flush=True)
    print(results_df["status"].value_counts().to_string(), flush=True)
    ok_count = (results_df["status"].isin(["ok", "exists"])).sum()
    print(f"Usable cached files: {ok_count}/{len(df)}", flush=True)


if __name__ == "__main__":
    main()
