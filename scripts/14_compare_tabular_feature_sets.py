from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/tabular_feature_set_comparison"
)


CLINICAL_FEATURES = [
    "basis_sex",
    "basis_age",
    "a_anthro_bmi",
]


IMAGING_FEATURES = [
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


def run_experiment(df, name, features):
    print("\n" + "=" * 80)
    print("Experiment:", name)
    print("Features:", features)

    X = df[features].copy()
    y = df["label"].astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"ROC AUC:           {auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    clf = model.named_steps["clf"]
    coefs = pd.DataFrame({
        "feature": features,
        "coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    coef_path = OUT_DIR / f"{name}_coefficients.csv"
    coefs.to_csv(coef_path, index=False)

    return {
        "experiment": name,
        "n_features": len(features),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")

    # Clean diabetes yes/no task
    df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()
    df["label"] = df["diabetes_group"].map({
        "normal": 0,
        "T2D": 1,
    })

    all_features = CLINICAL_FEATURES + IMAGING_FEATURES

    df = df[["NAKO_ID", "diabetes_group", "label"] + all_features].copy()

    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Dataset shape:", df.shape)
    print("Label counts:")
    print(df["label"].value_counts())

    experiments = {
        "clinical_only": CLINICAL_FEATURES,
        "imaging_biomarkers_only": IMAGING_FEATURES,
        "clinical_plus_imaging_biomarkers": all_features,
    }

    results = []

    for name, features in experiments.items():
        results.append(run_experiment(df, name, features))

    results_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "tabular_feature_set_results.csv"
    results_df.to_csv(out_csv, index=False)

    print("\nFinal comparison:")
    print(results_df)
    print("\nSaved to:")
    print(out_csv)


if __name__ == "__main__":
    main()
