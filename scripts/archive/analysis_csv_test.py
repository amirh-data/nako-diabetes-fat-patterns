from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
# Keep only clean normal-vs-T2D cases
df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()

# Create clean binary label
df["label"] = df["diabetes_group"].map({
    "normal": 0,
    "T2D": 1,
})
print("Shape:", df.shape)
print("\nDiabetes group counts:")
print(df["diabetes_group"].value_counts(dropna=False))

print("\nT2D binary counts:")
print(df["t2d_binary"].value_counts(dropna=False))

print("\nPreview:")
print(df[["NAKO_ID", "diabetes_group", "t2d_binary", "VAT_ml", "SAT_ml", "VAT_SAT_ratio"]].head())
