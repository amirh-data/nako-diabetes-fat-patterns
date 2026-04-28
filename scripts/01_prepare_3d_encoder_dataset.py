from pathlib import Path
import pandas as pd


ANALYSIS_TABLE = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

FAT_IMAGE_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_repaired_input"
)

SEGMENTATION_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_full_output"
)

OUTPUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_TABLE)

    print("Loaded table:")
    print(df.shape)

    # Keep only normal and T2D for first experiment
    df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()

    print("After keeping normal and T2D only:")
    print(df["diabetes_group"].value_counts())

    # Binary label
    df["label"] = df["diabetes_group"].map({
        "normal": 0,
        "T2D": 1,
    })

    # Create expected paths
    df["fat_path"] = df["NAKO_ID"].apply(
        lambda x: FAT_IMAGE_DIR / f"NAKO751_{int(x)}_0000.nii.gz"
    )

    df["seg_path"] = df["NAKO_ID"].apply(
        lambda x: SEGMENTATION_DIR / f"NAKO751_{int(x)}.nii.gz"
    )

    # Check available files
    df["fat_exists"] = df["fat_path"].apply(lambda p: p.exists())
    df["seg_exists"] = df["seg_path"].apply(lambda p: p.exists())

    print("\nFile availability:")
    print(df[["fat_exists", "seg_exists"]].value_counts())

    df_valid = df[df["fat_exists"] & df["seg_exists"]].copy()

    print("\nValid cases:")
    print(df_valid.shape)
    print(df_valid["diabetes_group"].value_counts())

    out_csv = OUTPUT_DIR / "encoder_3d_cases.csv"
    df_valid[["NAKO_ID", "diabetes_group", "label", "fat_path", "seg_path"]].to_csv(
        out_csv, index=False
    )

    print(f"\nSaved case list to:\n{out_csv}")


if __name__ == "__main__":
    main()
