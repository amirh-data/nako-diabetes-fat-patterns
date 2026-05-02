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

EMBEDDING_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/image_embeddings_t2d_3000.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/combined_embedding_biomarker"
)


TABULAR_FEATURES = [
    "basis_sex",
    "basis_age",
    "a_anthro_bmi",
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

    df_tab = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
    df_emb = pd.read_csv(EMBEDDING_CSV)

    # Clean normal-vs-T2D only
    df_tab = df_tab[df_tab["diabetes_group"].isin(["normal", "T2D"])].copy()
    df_tab["label_clean"] = df_tab["diabetes_group"].map({
        "normal": 0,
        "T2D": 1,
    })

    # Merge embeddings with tabular features
    df = df_emb.merge(
        df_tab[["NAKO_ID", "label_clean"] + TABULAR_FEATURES],
        on="NAKO_ID",
        how="inner",
    )

    # Make sure labels match
    print("Merged shape:", df.shape)
    print("Label counts from embeddings:")
    print(df["label"].value_counts())
    print("Label counts from analysis table:")
    print(df["label_clean"].value_counts())

    # Use label from analysis table
    y = df["label_clean"].astype(int)

    emb_features = [c for c in df.columns if c.startswith("img_emb_")]

    feature_sets = {
        "image_embedding_only": emb_features,
        "tabular_only": TABULAR_FEATURES,
        "image_embedding_plus_tabular": emb_features + TABULAR_FEATURES,
    }

    results = []

    for name, features in feature_sets.items():
        print("\n" + "=" * 80)
        print("Experiment:", name)
        print("Number of features:", len(features))

        X = df[features].copy()

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

        results.append({
            "experiment": name,
            "n_features": len(features),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": auc,
        })

    df_results = pd.DataFrame(results)
    out_results = OUT_DIR / "combined_model_results.csv"
    df_results.to_csv(out_results, index=False)

    print("\nFinal comparison:")
    print(df_results)
    print("\nSaved results to:")
    print(out_results)


if __name__ == "__main__":
    main()
