from pathlib import Path
import pandas as pd

OUT = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/experiment_summary"
)
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "experiment": "3D image encoder - T2D yes/no",
        "input": "fat image + binary Dataset301 mask, 96^3",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.5000,
        "balanced_accuracy": 0.5000,
        "roc_auc": 0.5000,
        "interpretation": "Image-only model did not generalize for T2D.",
    },
    {
        "experiment": "3D image encoder - VAT/SAT extremes",
        "input": "fat image + binary Dataset301 mask, 96^3",
        "target": "low vs high VAT/SAT ratio",
        "accuracy_or_val_acc": 0.9983,
        "balanced_accuracy": None,
        "roc_auc": None,
        "interpretation": "Image pipeline can learn obvious fat-distribution phenotype.",
    },
    {
        "experiment": "Image embedding only",
        "input": "256-d image embedding",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.5000,
        "balanced_accuracy": 0.5000,
        "roc_auc": 0.5000,
        "interpretation": "Extracted image embedding did not predict T2D.",
    },
    {
        "experiment": "Image embedding + tabular",
        "input": "256-d image embedding + clinical/imaging tabular features",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.7233,
        "balanced_accuracy": 0.7233,
        "roc_auc": 0.7911,
        "interpretation": "Image embedding did not improve over tabular features.",
    },
    {
        "experiment": "Clinical only",
        "input": "age + sex + BMI",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.7201,
        "balanced_accuracy": 0.7494,
        "roc_auc": 0.8221,
        "interpretation": "Clinical variables carry most predictive signal.",
    },
    {
        "experiment": "Imaging biomarkers only",
        "input": "VAT/SAT/organ/muscle/marrow imaging biomarkers",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.5609,
        "balanced_accuracy": 0.5141,
        "roc_auc": 0.5215,
        "interpretation": "Imaging biomarkers alone weak for T2D.",
    },
    {
        "experiment": "Clinical + imaging biomarkers",
        "input": "age + sex + BMI + imaging biomarkers",
        "target": "normal vs T2D",
        "accuracy_or_val_acc": 0.7205,
        "balanced_accuracy": 0.7511,
        "roc_auc": 0.8215,
        "interpretation": "Adding imaging biomarkers did not improve over clinical-only.",
    },
]

df = pd.DataFrame(rows)
out_csv = OUT / "experiment_summary.csv"
df.to_csv(out_csv, index=False)

print(df)
print(f"\nSaved to:\n{out_csv}")
