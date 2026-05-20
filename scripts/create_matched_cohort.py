import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load full dataset
df = pd.read_csv(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/quantification_full/analysis_table_with_diabetes_labels.csv",
    sep=";", decimal=","
)

df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()
df["label"] = (df["diabetes_group"] == "T2D").astype(int)

# Convert columns
for col in ["basis_age", "a_anthro_bmi", "basis_sex"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["basis_age", "a_anthro_bmi", "basis_sex"])
df = df.reset_index(drop=True)

print("Before matching:")
print(df["label"].value_counts())
print(f"Age  — T2D: {df[df.label==1]['basis_age'].mean():.1f}  Normal: {df[df.label==0]['basis_age'].mean():.1f}")
print(f"BMI  — T2D: {df[df.label==1]['a_anthro_bmi'].mean():.1f}  Normal: {df[df.label==0]['a_anthro_bmi'].mean():.1f}")
print(f"Sex  — T2D: {df[df.label==1]['basis_sex'].mean():.2f}  Normal: {df[df.label==0]['basis_sex'].mean():.2f}")

# Separate
t2d    = df[df["label"] == 1].reset_index(drop=True)
normal = df[df["label"] == 0].reset_index(drop=True)

# Normalize matching features
features = ["basis_age", "a_anthro_bmi", "basis_sex"]
scaler       = StandardScaler()
normal_scaled = scaler.fit_transform(normal[features])
t2d_scaled    = scaler.transform(t2d[features])

# Match WITHOUT replacement
# For each T2D subject find closest unmatched normal
from scipy.spatial.distance import cdist

dist_matrix = cdist(t2d_scaled, normal_scaled)  # (n_t2d, n_normal)

matched_normal_indices = []
used_normal_indices    = set()

for t2d_idx in range(len(t2d)):
    # sort normals by distance to this T2D subject
    sorted_normal_idx = np.argsort(dist_matrix[t2d_idx])
    
    # pick closest unused normal
    for normal_idx in sorted_normal_idx:
        if normal_idx not in used_normal_indices:
            matched_normal_indices.append(normal_idx)
            used_normal_indices.add(normal_idx)
            break

matched_normals = normal.iloc[matched_normal_indices].copy()

print(f"\nAfter matching WITHOUT replacement:")
print(f"T2D:    {len(t2d)}")
print(f"Normal: {len(matched_normals)}")

# Check matching quality
age_diff = np.abs(t2d["basis_age"].values - matched_normals["basis_age"].values)
bmi_diff = np.abs(t2d["a_anthro_bmi"].values - matched_normals["a_anthro_bmi"].values)
sex_diff = np.abs(t2d["basis_sex"].values - matched_normals["basis_sex"].values)

print(f"\nMatching quality:")
print(f"Age difference  — mean: {age_diff.mean():.1f} years, max: {age_diff.max():.0f}")
print(f"BMI difference  — mean: {bmi_diff.mean():.2f}, max: {bmi_diff.max():.1f}")
print(f"Sex mismatch    — fraction: {(sex_diff > 0).mean():.2f}")

# Combine
matched = pd.concat([t2d, matched_normals]).reset_index(drop=True)
print(f"\nFinal matched cohort: {len(matched)} subjects")
print(matched["label"].value_counts())

print(f"\nAfter matching:")
print(f"Age  — T2D: {matched[matched.label==1]['basis_age'].mean():.1f}  Normal: {matched[matched.label==0]['basis_age'].mean():.1f}")
print(f"BMI  — T2D: {matched[matched.label==1]['a_anthro_bmi'].mean():.1f}  Normal: {matched[matched.label==0]['a_anthro_bmi'].mean():.1f}")
print(f"Sex  — T2D: {matched[matched.label==1]['basis_sex'].mean():.2f}  Normal: {matched[matched.label==0]['basis_sex'].mean():.2f}")

# Check NPZ files
npz_dir = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/supervised_image_datasets/npz_96_fat_water_t2d_3000"
)

rows = []
missing = []
for _, row in matched.iterrows():
    p = npz_dir / f"{int(row['NAKO_ID'])}.npz"
    if p.exists():
        rows.append({
            "NAKO_ID":  int(row["NAKO_ID"]),
            "label":    int(row["label"]),
            "npz_path": str(p),
        })
    else:
        missing.append(int(row["NAKO_ID"]))

out = pd.DataFrame(rows)
print(f"\nWith NPZ files: {len(out)}")
print(f"Missing NPZ:    {len(missing)}")
print(out["label"].value_counts())

out_path = npz_dir / "prepared_npz_matched_cohort.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
