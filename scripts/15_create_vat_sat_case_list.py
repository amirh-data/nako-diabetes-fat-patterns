from pathlib import Path

import pandas as pd


ANALYSIS_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

FAT_IMAGE_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_repaired_input"
)

SEGMENTATION_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_full_output"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

    df = df[["NAKO_ID", "VAT_SAT_ratio"]].copy()
    df["VAT_SAT_ratio"] = pd.to_numeric(df["VAT_SAT_ratio"], errors="coerce")
    df = df.dropna(subset=["VAT_SAT_ratio"])

    low_thr = df["VAT_SAT_ratio"].quantile(0.25)
    high_thr = df["VAT_SAT_ratio"].quantile(0.75)

    print("VAT/SAT thresholds:")
    print("low 25% <=", low_thr)
    print("high 25% >=", high_thr)

    low = df[df["VAT_SAT_ratio"] <= low_thr].copy()
    high = df[df["VAT_SAT_ratio"] >= high_thr].copy()

    low["label"] = 0
    high["label"] = 1

    N_PER_CLASS = 1500

    low = low.sample(n=min(N_PER_CLASS, len(low)), random_state=42)
    high = high.sample(n=min(N_PER_CLASS, len(high)), random_state=42)

    out = pd.concat([low, high], axis=0)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    out["fat_path"] = out["NAKO_ID"].apply(
        lambda x: FAT_IMAGE_DIR / f"NAKO751_{int(x)}_0000.nii.gz"
    )

    out["seg_path"] = out["NAKO_ID"].apply(
        lambda x: SEGMENTATION_DIR / f"NAKO751_{int(x)}.nii.gz"
    )

    out["fat_exists"] = out["fat_path"].apply(lambda p: p.exists())
    out["seg_exists"] = out["seg_path"].apply(lambda p: p.exists())

    print("\nBefore file filtering:")
    print(out["label"].value_counts())

    out = out[out["fat_exists"] & out["seg_exists"]].copy()

    print("\nAfter file filtering:")
    print(out["label"].value_counts())

    out_csv = OUT_DIR / "encoder_3d_cases_vat_sat_extremes.csv"

    out[["NAKO_ID", "VAT_SAT_ratio", "label", "fat_path", "seg_path"]].to_csv(
        out_csv, index=False
    )

    print("\nSaved to:")
    print(out_csv)


if __name__ == "__main__":
    main()

