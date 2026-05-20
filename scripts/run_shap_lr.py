import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")
ANALYSIS_CSV = BASE / "quantification_full/analysis_table_with_diabetes_labels.csv"
MATCHED_CSV  = BASE / "cohorts/nako_t2d_normal_matched_cohort.csv"
OUT_DIR      = BASE / "shap_results"
OUT_DIR.mkdir(exist_ok=True)

# In run_shap_lr.py, use only:
FEATURES = [
    "VAT_ml", "SAT_abd_ml", "SAT_tho_ml", "SAT_ml",
    "VAT_SAT_ratio", "CardiacAT_ml", "Liver_ff_pct",
    "Pancreas_ff_pct", "BackMuscle_IMAT_ml",
    "BackMuscle_ff_pct",
]

matched = pd.read_csv(MATCHED_CSV)[["NAKO_ID", "label"]]
ana     = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
df      = matched.merge(ana[["NAKO_ID"] + FEATURES], on="NAKO_ID", how="left")
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

y = df["label"].astype(int)
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train = scaler.fit_transform(imputer.fit_transform(df[FEATURES].iloc[train_idx]))
X_test  = scaler.transform(imputer.transform(df[FEATURES].iloc[test_idx]))

clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
clf.fit(X_train, y.iloc[train_idx])

# SHAP
explainer  = shap.LinearExplainer(clf, X_train)
shap_vals  = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals, X_test,
    feature_names=FEATURES,
    show=False,
    plot_type="bar"
)
plt.title("SHAP Feature Importance — T2D Biomarkers (Matched Cohort)")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Beeswarm plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals, X_test,
    feature_names=FEATURES,
    show=False
)
plt.title("SHAP Values — T2D Biomarkers (Matched Cohort)")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved SHAP figures to {OUT_DIR}")
print("\nFeature importance ranking:")
mean_shap = np.abs(shap_vals).mean(axis=0)
for feat, val in sorted(zip(FEATURES, mean_shap),
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat:<35} {val:.4f}")
