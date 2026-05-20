"""
scripts/42_distribution_features_matched.py

Compare tabular feature groups on matched cohort:
  1. clinical_only
  2. global_biomarkers_only
  3. distribution_only
  4. global_plus_distribution
  5. clinical_plus_global_plus_distribution

Answers: Is T2D signal in total fat amount, fat distribution, or both?
"""

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
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/cohorts/nako_t2d_normal_matched_cohort.csv"
)
OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/distribution_feature_comparison"
)

CLINICAL = [
    "basis_sex",
    "basis_age",
    "a_anthro_bmi",
]

GLOBAL_BIOMARKERS = [
    "VAT_ml",
    "SAT_abd_ml",
    "SAT_tho_ml",
    "SAT_ml",
    "VAT_SAT_ratio",
    "CardiacAT_ml",
    "Liver_ff_pct",
    "Pancreas_ff_pct",
    "BackMuscle_IMAT_ml",
    "BackMuscle_ff_pct",
    "VertebralMarrow_mean_ff_pct",
]

DISTRIBUTION = [
    "SAT_abd_ml",
    "SAT_tho_ml",
    "VAT_SAT_ratio",
    "L5_ff_pct", "L4_ff_pct", "L3_ff_pct",
    "L2_ff_pct", "L1_ff_pct",
    "Th12_ff_pct", "Th11_ff_pct", "Th10_ff_pct",
    "Th9_ff_pct", "Th8_ff_pct", "Th7_ff_pct",
    "Th6_ff_pct", "Th5_ff_pct", "Th4_ff_pct",
    "Th3_ff_pct", "Th2_ff_pct", "Th1_ff_pct",
]

ALL_FEATURES = sorted(set(CLINICAL + GLOBAL_BIOMARKERS + DISTRIBUTION))


def run_experiment(df, name, features, train_idx, test_idx):
    y = df["label"].astype(int)
    X = df[features].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42
        )),
    ])

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    y_pred = model.predict(X.iloc[test_idx])

    auc     = roc_auc_score(y.iloc[test_idx], y_prob)
    bal_acc = balanced_accuracy_score(y.iloc[test_idx], y_pred)
    cm      = confusion_matrix(y.iloc[test_idx], y_pred)

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Features ({len(features)}): {features[:5]}{'...' if len(features)>5 else ''}")
    print(f"AUC:               {auc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # Feature importance
    coefs = pd.DataFrame({
        "feature": features,
        "coef": model.named_steps["clf"].coef_[0],
        "abs_coef": abs(model.named_steps["clf"].coef_[0]),
    }).sort_values("abs_coef", ascending=False)
    coefs.to_csv(OUT_DIR / f"{name}_coefs.csv", index=False)

    return {"experiment": name, "n_features": len(features),
            "auc": auc, "bal_acc": bal_acc}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(MATCHED_CSV)[["NAKO_ID", "label"]]
    ana = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

    df = matched.merge(ana[["NAKO_ID"] + ALL_FEATURES], on="NAKO_ID", how="left")

    for col in ALL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Dataset shape: {df.shape}")
    print(df["label"].value_counts())

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2,
        random_state=42, stratify=df["label"]
    )

    experiments = {
        "clinical_only": CLINICAL,
        "global_biomarkers_only": GLOBAL_BIOMARKERS,
        "distribution_only": DISTRIBUTION,
        "global_plus_distribution": list(set(GLOBAL_BIOMARKERS + DISTRIBUTION)),
        "clinical_plus_global_plus_distribution": ALL_FEATURES,
    }

    results = []
    for name, features in experiments.items():
        results.append(run_experiment(df, name, features, train_idx, test_idx))

    df_results = pd.DataFrame(results).sort_values("auc", ascending=False)
    df_results.to_csv(OUT_DIR / "results.csv", index=False)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON:")
    print(df_results.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
