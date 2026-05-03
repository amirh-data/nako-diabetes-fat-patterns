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


FEATURE_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/regional_pixel_features/regional_features_cropped128_balanced_t2d_3000.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/regional_pixel_features"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_CSV)

    y = df["label"].astype(int)
    feature_cols = [c for c in df.columns if c not in ["NAKO_ID", "label"]]
    X = df[feature_cols]

    print("Dataset shape:", df.shape)
    print("Number of features:", len(feature_cols))
    print("Label counts:")
    print(y.value_counts())

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

    experiments = {
        "l2_C1": LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "l1_C001": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.01,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "l1_C01": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.1,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "l1_C1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    rows = []

    for name, clf in experiments.items():
        print("\n" + "=" * 80)
        print("Experiment:", name)

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
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
        print(classification_report(y_test, y_pred, digits=4))

        rows.append({
            "experiment": name,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": auc,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "regional_cropped128_balanced_t2d_classifier_results.csv"
    out.to_csv(out_path, index=False)

    print("\nFinal comparison:")
    print(out)
    print("\nSaved to:")
    print(out_path)


if __name__ == "__main__":
    main()
