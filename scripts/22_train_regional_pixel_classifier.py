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
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/regional_pixel_features/regional_pixel_features_t2d_3000.csv"
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

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.1,
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

    result = pd.DataFrame([
        {
            "experiment": "regional_pixel_cluster_features",
            "n_features": len(feature_cols),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": auc,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        }
    ])

    result_path = OUT_DIR / "regional_pixel_classifier_results.csv"
    result.to_csv(result_path, index=False)

    # Save coefficients
    clf = model.named_steps["clf"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    coef_path = OUT_DIR / "regional_pixel_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print("\nSaved results to:")
    print(result_path)

    print("\nTop regional features:")
    print(coef_df.head(20))


if __name__ == "__main__":
    main()
