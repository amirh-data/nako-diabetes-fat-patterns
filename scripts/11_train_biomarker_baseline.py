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


CSV_PATH = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/biomarker_baseline_imaging_only"
)


FEATURES = [
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH, sep=";", decimal=",")

    # Clean binary task: normal vs T2D only
    df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()

    df["label"] = df["diabetes_group"].map({
        "normal": 0,
        "T2D": 1,
    })

    # Keep only columns we need
    df = df[["NAKO_ID", "diabetes_group", "label"] + FEATURES].copy()

    # Convert features to numeric just in case
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Dataset shape:", df.shape)
    print("Label counts:")
    print(df["label"].value_counts())

    X = df[FEATURES]
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

    print("\nTrain counts:")
    print(y_train.value_counts())

    print("\nTest counts:")
    print(y_test.value_counts())

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

    print("\nResults:")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"ROC AUC:           {auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save coefficients for interpretation
    clf = model.named_steps["clf"]
    coefs = pd.DataFrame({
        "feature": FEATURES,
        "coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    out_coef = OUT_DIR / "logistic_regression_coefficients.csv"
    coefs.to_csv(out_coef, index=False)

    print(f"\nSaved coefficients to:\n{out_coef}")


if __name__ == "__main__":
    main()
