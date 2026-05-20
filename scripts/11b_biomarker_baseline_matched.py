from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

MATCHED_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/supervised_image_datasets/npz_96_fat_water_t2d_3000/prepared_npz_matched_cohort.csv"
)

FEATURES = [
    "VAT_ml", "SAT_abd_ml", "SAT_tho_ml", "SAT_ml",
    "VAT_SAT_ratio", "CardiacAT_ml", "Liver_ff_pct",
    "Pancreas_ff_pct", "BackMuscle_IMAT_ml",
    "BackMuscle_ff_pct", "VertebralMarrow_mean_ff_pct",
]

# Load matched cohort IDs
matched = pd.read_csv(MATCHED_CSV)

# Load analysis table
analysis = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

# Merge
df = matched.merge(analysis[["NAKO_ID"] + FEATURES], on="NAKO_ID", how="left")

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Matched cohort shape:", df.shape)
print("Label counts:")
print(df["label"].value_counts())

X = df[FEATURES]
y = df["label"].astype(int)

train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2,
    random_state=42, stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X.iloc[train_idx], y.iloc[train_idx])
y_pred = model.predict(X.iloc[test_idx])
y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]

auc     = roc_auc_score(y.iloc[test_idx], y_prob)
bal_acc = balanced_accuracy_score(y.iloc[test_idx], y_pred)
cm      = confusion_matrix(y.iloc[test_idx], y_pred)

print(f"\nMatched cohort biomarker baseline:")
print(f"ROC AUC:           {auc:.4f}")
print(f"Balanced accuracy: {bal_acc:.4f}")
print("Confusion matrix:")
print(cm)

# Also test age+sex+BMI on matched cohort
DEMO = ["basis_age", "a_anthro_bmi", "basis_sex"]
analysis2 = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
for col in DEMO:
    analysis2[col] = pd.to_numeric(analysis2[col], errors="coerce")

df2 = matched.merge(analysis2[["NAKO_ID"] + DEMO], on="NAKO_ID", how="left")
X2  = df2[DEMO]

model2 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     LogisticRegression(max_iter=5000, class_weight="balanced")),
])
model2.fit(X2.iloc[train_idx], y.iloc[train_idx])
y_prob2 = model2.predict_proba(X2.iloc[test_idx])[:, 1]
auc2    = roc_auc_score(y.iloc[test_idx], y_prob2)

print(f"\nAge+sex+BMI on matched cohort: AUC={auc2:.4f}")
print("(Should be ~0.50 if matching worked correctly)")
