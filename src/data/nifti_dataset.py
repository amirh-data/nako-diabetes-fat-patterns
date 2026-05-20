"""
src/data/nifti_dataset.py

Dataset utilities for direct NIfTI training and cached .pt training.

Direct NIfTI input modes:
    fat              -> fat only
    fat_water        -> fat + water
    seg              -> full segmentation label map, normalized
    seg_fat_water    -> fat + water + full segmentation

Cached trunk input modes:
    fat
    fat_water
    seg_trunk_binary
    seg_trunk_label
    fat_water_seg_trunk_binary
    fat_water_seg_trunk_label

Cohorts:
    matched          -> 3,040 matched cohort
    full             -> all T2D vs normal subjects

Feature modes:
    none
    clinical
    biomarkers
    clinical_biomarkers
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = Path("/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1")

IMG_DIR = BASE / "dataset301_repaired_input"
SEG_DIR = BASE / "dataset301_full_output"

ANALYSIS_CSV = BASE / "quantification_full/analysis_table_with_diabetes_labels.csv"

MATCHED_CSV = BASE / "cohorts/nako_t2d_normal_matched_cohort.csv"


# ---------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------

CLINICAL_COLS = [
    "basis_age",
    "basis_sex",
    "a_anthro_bmi",
]

BIOMARKER_COLS = [
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

DISTRIBUTION_COLS = [
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


def get_feature_cols(feature_mode):
    if feature_mode == "none":
        return []
    elif feature_mode == "clinical":
        return CLINICAL_COLS
    elif feature_mode == "biomarkers":
        return BIOMARKER_COLS
    elif feature_mode == "distribution":
        return DISTRIBUTION_COLS
    elif feature_mode == "biomarkers_distribution":
        return list(set(BIOMARKER_COLS + DISTRIBUTION_COLS))
    elif feature_mode == "clinical_biomarkers":
        return CLINICAL_COLS + BIOMARKER_COLS
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


# ---------------------------------------------------------------------
# Input mode definitions
# ---------------------------------------------------------------------

DIRECT_NIFTI_MODES = [
    "fat",
    "fat_water",
    "seg",
    "seg_fat_water",
]

CACHED_TRUNK_MODES = [
    "fat",
    "fat_water",
    "seg_trunk_binary",
    "seg_trunk_label",
    "fat_water_seg_trunk_binary",
    "fat_water_seg_trunk_label",
    
    # fat/water intensities only inside selected trunk labels
    "fat_water_trunk_masked",
    "fat_water_trunk_masked_seg_binary",
    "fat_water_trunk_masked_seg_label",
    
    # fat/water with near-zero background removed, whole body kept
    "fat_water_body_masked",
    "fat_water_body_masked_seg_binary",
    "fat_water_body_masked_seg_label",
]

ALL_INPUT_MODES = sorted(set(DIRECT_NIFTI_MODES + CACHED_TRUNK_MODES))


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def load_nifti_arr(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def resize_volume(arr, target_shape=(128, 128, 128), is_seg=False):
    """
    Resize volume to target shape.

    Images use trilinear interpolation.
    Segmentations use nearest-neighbor interpolation.
    """
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    if is_seg:
        t = F.interpolate(t, size=target_shape, mode="nearest")
    else:
        t = F.interpolate(
            t,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    return t.squeeze().numpy()


def normalize_01(arr):
    """
    Percentile clipping and scaling to [0, 1].
    """
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    arr = np.clip(arr, p1, p99)

    denom = p99 - p1
    if denom > 0:
        arr = (arr - p1) / denom
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.float32)


def normalize_seg(arr, n_labels=19):
    """
    Normalize full segmentation labels from 0-19 to 0-1.

    Note:
    This is only for direct NIfTI fallback.
    For the cached trunk setup, we use selected trunk labels separately.
    """
    arr = np.rint(arr)
    arr = np.clip(arr, 0, n_labels)
    return (arr / n_labels).astype(np.float32)


# ---------------------------------------------------------------------
# Direct NIfTI Dataset
# ---------------------------------------------------------------------

class NiftiDataset(Dataset):
    """
    Directly loads raw NIfTI files every epoch.

    This is slower but useful as fallback/debug.
    """

    def __init__(
        self,
        df,
        input_mode="fat_water",
        feature_mode="none",
        resolution=128,
        augment=False,
    ):
        self.df = df.reset_index(drop=True)
        self.input_mode = input_mode
        self.feature_mode = feature_mode
        self.feature_cols = get_feature_cols(feature_mode)
        self.target_shape = (resolution, resolution, resolution)
        self.augment = augment

        if self.input_mode not in DIRECT_NIFTI_MODES:
            raise ValueError(
                f"Unknown direct NIfTI input_mode: {self.input_mode}. "
                f"Allowed: {DIRECT_NIFTI_MODES}"
            )

    def __len__(self):
        return len(self.df)

    def _load_fat(self, nako_id):
        path = IMG_DIR / f"NAKO751_{nako_id}_0000.nii.gz"
        arr = load_nifti_arr(path)
        arr = resize_volume(arr, self.target_shape, is_seg=False)
        arr = normalize_01(arr)
        return arr

    def _load_water(self, nako_id):
        path = IMG_DIR / f"NAKO751_{nako_id}_0001.nii.gz"
        arr = load_nifti_arr(path)
        arr = resize_volume(arr, self.target_shape, is_seg=False)
        arr = normalize_01(arr)
        return arr

    def _load_seg(self, nako_id):
        path = SEG_DIR / f"NAKO751_{nako_id}.nii.gz"
        arr = load_nifti_arr(path)
        arr = resize_volume(arr, self.target_shape, is_seg=True)
        arr = normalize_seg(arr, n_labels=19)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nako_id = int(row["NAKO_ID"])
        label = int(row["label"])

        if self.input_mode == "fat":
            channels = [self._load_fat(nako_id)]

        elif self.input_mode == "fat_water":
            channels = [
                self._load_fat(nako_id),
                self._load_water(nako_id),
            ]

        elif self.input_mode == "seg":
            channels = [self._load_seg(nako_id)]

        elif self.input_mode == "seg_fat_water":
            channels = [
                self._load_fat(nako_id),
                self._load_water(nako_id),
                self._load_seg(nako_id),
            ]

        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        x = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        if self.augment:
            x = apply_torchio_augmentation(x, input_mode=self.input_mode)

        if len(self.feature_cols) > 0:
            features = torch.tensor(
                row[self.feature_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
            return x, features, y

        return x, y


# ---------------------------------------------------------------------
# Cached .pt Dataset
# ---------------------------------------------------------------------

class CachedNiftiDataset(Dataset):
    """
    Loads cached 128³ .pt tensors.

    Expected cached file:
        {
            "x": tensor [4, 128, 128, 128],
            "label": 0/1,
            "NAKO_ID": int
        }

    Cached channels:
        0 = fat
        1 = water
        2 = seg_trunk_binary
        3 = seg_trunk_label
    """

    def __init__(
        self,
        df,
        cache_dir,
        input_mode="fat_water",
        feature_mode="none",
        augment=False,
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.input_mode = input_mode
        self.feature_mode = feature_mode
        self.feature_cols = get_feature_cols(feature_mode)
        self.augment = augment

        if self.input_mode not in CACHED_TRUNK_MODES:
            raise ValueError(
                f"Unknown cached input_mode: {self.input_mode}. "
                f"Allowed: {CACHED_TRUNK_MODES}"
            )

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

    def __len__(self):
        return len(self.df)

    def _select_channels(self, x):
        """
        Select channels from cached tensor.

        x shape: [4, 128, 128, 128]
            0 = fat
            1 = water
            2 = seg_trunk_binary
            3 = seg_trunk_label
        """
        if self.input_mode == "fat":
            return x[0:1]

        if self.input_mode == "fat_water":
            return x[0:2]

        if self.input_mode == "seg_trunk_binary":
            return x[2:3]

        if self.input_mode == "seg_trunk_label":
            return x[3:4]

        if self.input_mode == "fat_water_seg_trunk_binary":
            return x[[0, 1, 2]]

        if self.input_mode == "fat_water_seg_trunk_label":
            return x[[0, 1, 3]]

        if self.input_mode == "fat_water_trunk_masked":
            fat = x[0:1]
            water = x[1:2]
            trunk_mask = x[2:3]

            fat_masked = fat * trunk_mask
            water_masked = water * trunk_mask

            return torch.cat([fat_masked, water_masked], dim=0)

        if self.input_mode == "fat_water_trunk_masked_seg_binary":
            fat = x[0:1]
            water = x[1:2]
            trunk_mask = x[2:3]

            fat_masked = fat * trunk_mask
            water_masked = water * trunk_mask

            return torch.cat([fat_masked, water_masked, trunk_mask], dim=0)

        if self.input_mode == "fat_water_trunk_masked_seg_label":
            fat = x[0:1]
            water = x[1:2]
            trunk_mask = x[2:3]
            trunk_label = x[3:4]

            fat_masked = fat * trunk_mask
            water_masked = water * trunk_mask

            return torch.cat([fat_masked, water_masked, trunk_label], dim=0)

        if self.input_mode == "fat_water_body_masked":
            fat = x[0:1]
            water = x[1:2]

            body_mask = ((fat + water) > 1e-4).float()

            fat_masked = fat * body_mask
            water_masked = water * body_mask

            return torch.cat([fat_masked, water_masked], dim=0)

        if self.input_mode == "fat_water_body_masked_seg_binary":
            fat = x[0:1]
            water = x[1:2]
            trunk_binary = x[2:3]

            body_mask = ((fat + water) > 1e-4).float()

            fat_masked = fat * body_mask
            water_masked = water * body_mask

            return torch.cat([fat_masked, water_masked, trunk_binary], dim=0)

        if self.input_mode == "fat_water_body_masked_seg_label":
            fat = x[0:1]
            water = x[1:2]
            trunk_label = x[3:4]

            body_mask = ((fat + water) > 1e-4).float()

            fat_masked = fat * body_mask
            water_masked = water * body_mask

            return torch.cat([fat_masked, water_masked, trunk_label], dim=0)
        
        raise ValueError(f"Unknown input_mode: {self.input_mode}")
        

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nako_id = int(row["NAKO_ID"])
        label = int(row["label"])

        path = self.cache_dir / f"NAKO751_{nako_id}.pt"

        if not path.exists():
            raise FileNotFoundError(f"Cached file not found: {path}")

        sample = torch.load(path, map_location="cpu")

        x = sample["x"]
        x = self._select_channels(x)

        # Cache is float16 to save disk, but training uses float32.
        x = x.float()

        y = torch.tensor(label, dtype=torch.long)

        if self.augment:
            x = apply_torchio_augmentation(x, input_mode=self.input_mode)

        if len(self.feature_cols) > 0:
            features = torch.tensor(
                row[self.feature_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
            return x, features, y

        return x, y


# ---------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------

def apply_torchio_augmentation(x, input_mode="fat_water"):
    """
    TorchIO augmentation for image tensor [C, D, H, W].

    Important:
    - Spatial transforms are applied to all channels.
    - Intensity transforms are applied only to MRI intensity channels.
    - Segmentation channels are not changed by noise/gamma.
    """

    # -----------------------------
    # Case 1: image-only modes
    # -----------------------------
    if input_mode in ["fat", "fat_water", "fat_water_trunk_masked", "fat_water_body_masked"]:
        transform = tio.Compose([
            tio.RandomFlip(axes=("LR",)),
            tio.RandomFlip(axes=("AP",)),
            tio.RandomAffine(scales=0.1, degrees=10, translation=5),
            tio.RandomNoise(std=0.02),
            tio.RandomGamma(log_gamma=0.2),
        ])

        subject = tio.Subject(image=tio.ScalarImage(tensor=x))
        transformed = transform(subject)
        x = transformed["image"].data
        return torch.clamp(x, 0.0, 1.0)
    
    # -----------------------------
    # Case 1+: trunk masked image-only modes
    # -----------------------------
    if input_mode == "fat_water_trunk_masked":
        spatial = tio.Compose([
            tio.RandomFlip(axes=("LR",)),
            tio.RandomFlip(axes=("AP",)),
            tio.RandomAffine(scales=0.1, degrees=10, translation=5),
        ])

        intensity = tio.Compose([
            tio.RandomNoise(std=0.02),
            tio.RandomGamma(log_gamma=0.2),
        ])

        subject = tio.Subject(image=tio.ScalarImage(tensor=x))
        transformed = spatial(subject)
        x = transformed["image"].data

        img_subject = tio.Subject(image=tio.ScalarImage(tensor=x))
        img_subject = intensity(img_subject)
        x = img_subject["image"].data

        x = torch.clamp(x, 0.0, 1.0)
        return x
    
    # -----------------------------
    # Case 2: segmentation-only modes
    # -----------------------------
    if input_mode in ["seg", "seg_trunk_binary", "seg_trunk_label"]:
        spatial = tio.Compose([
            tio.RandomFlip(axes=("LR",)),
            tio.RandomFlip(axes=("AP",)),
            tio.RandomAffine(scales=0.1, degrees=10, translation=5),
        ])

        subject = tio.Subject(image=tio.ScalarImage(tensor=x))
        transformed = spatial(subject)
        x = transformed["image"].data
        x = torch.clamp(x, 0.0, 1.0)

        if input_mode == "seg_trunk_binary":
            x = (x > 0.5).float()

        if input_mode == "seg_trunk_label":
            x = torch.round(x * 7.0) / 7.0

        return x

    # -----------------------------
    # Case 3: fat/water + segmentation
    # -----------------------------
    if input_mode in [
        "seg_fat_water",
        "fat_water_seg_trunk_binary",
        "fat_water_seg_trunk_label",
        "fat_water_trunk_masked_seg_binary",
        "fat_water_trunk_masked_seg_label",
        "fat_water_body_masked_seg_binary",
        "fat_water_body_masked_seg_label",
    ]:
        spatial = tio.Compose([
            tio.RandomFlip(axes=("LR",)),
            tio.RandomFlip(axes=("AP",)),
            tio.RandomAffine(scales=0.1, degrees=10, translation=5),
        ])

        intensity = tio.Compose([
            tio.RandomNoise(std=0.02),
            tio.RandomGamma(log_gamma=0.2),
        ])

        # Apply spatial augmentation to all channels together
        subject = tio.Subject(image=tio.ScalarImage(tensor=x))
        transformed = spatial(subject)
        x = transformed["image"].data

        # Split image channels and segmentation channel(s)
        img = x[:2]
        seg = x[2:]

        # Apply intensity augmentation only to fat/water
        img_subject = tio.Subject(image=tio.ScalarImage(tensor=img))
        img_subject = intensity(img_subject)
        img = img_subject["image"].data

        img = torch.clamp(img, 0.0, 1.0)
        seg = torch.clamp(seg, 0.0, 1.0)

        # Restore segmentation-like values
        if input_mode == "seg_fat_water":
            seg = torch.round(seg * 19.0) / 19.0

        if input_mode in [
            "fat_water_seg_trunk_binary",
            "fat_water_trunk_masked_seg_binary",
            "fat_water_body_masked_seg_binary",
        ]:
            seg = (seg > 0.5).float()

        if input_mode in [
            "fat_water_seg_trunk_label",
            "fat_water_trunk_masked_seg_label",
            "fat_water_body_masked_seg_label",
        ]:
            seg = torch.round(seg * 7.0) / 7.0
            
        # For trunk-masked modes, keep fat/water zero outside the trunk after augmentation
        if input_mode == "fat_water_trunk_masked_seg_binary":
            img = img * seg

        if input_mode == "fat_water_trunk_masked_seg_label":
            trunk_mask = (seg > 0).float()
            img = img * trunk_mask    

        x = torch.cat([img, seg], dim=0)
        return x

    raise ValueError(f"Unknown input_mode for augmentation: {input_mode}")


# ---------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------

def _check_required_files(df, input_mode, feature_mode):
    """
    Checks required raw NIfTI files.

    For cached trunk modes, this checks the source files that are needed
    to create the cache: fat, water, and segmentation.
    """
    feature_cols = get_feature_cols(feature_mode)

    if input_mode in CACHED_TRUNK_MODES:
        need_fat = True
        need_water = True
        need_seg = True
    else:
        need_fat = input_mode in ["fat", "fat_water", "seg_fat_water"]
        need_water = input_mode in ["fat_water", "seg_fat_water"]
        need_seg = input_mode in ["seg", "seg_fat_water"]

    valid = []

    for _, row in df.iterrows():
        nako_id = int(row["NAKO_ID"])

        fat_ok = (IMG_DIR / f"NAKO751_{nako_id}_0000.nii.gz").exists()
        water_ok = (IMG_DIR / f"NAKO751_{nako_id}_0001.nii.gz").exists()
        seg_ok = (SEG_DIR / f"NAKO751_{nako_id}.nii.gz").exists()

        if need_fat and not fat_ok:
            continue
        if need_water and not water_ok:
            continue
        if need_seg and not seg_ok:
            continue

        item = {
            "NAKO_ID": nako_id,
            "label": int(row["label"]),
        }

        for col in feature_cols:
            item[col] = row[col]

        valid.append(item)

    out = pd.DataFrame(valid)

    if len(feature_cols) > 0:
        before = len(out)
        out = out.dropna(subset=feature_cols).copy()
        removed = before - len(out)

        print(
            f"After dropping missing feature rows: {len(out)} "
            f"(removed {removed})",
            flush=True,
        )

    return out


def load_matched_dataset(input_mode="fat_water", feature_mode="none"):
    """
    Load matched cohort.

    MATCHED_CSV is only used as NAKO_ID + label list.
    Image input can be raw NIfTI or cached .pt tensors.
    """
    if not MATCHED_CSV.exists():
        raise FileNotFoundError(
            f"Matched cohort CSV not found: {MATCHED_CSV}\n"
            "Create it with:\n"
            "mkdir -p /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/cohorts\n"
            "cp /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/"
            "supervised_image_datasets/npz_96_fat_water_t2d_3000/"
            "prepared_npz_matched_cohort.csv "
            "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/cohorts/"
            "nako_t2d_normal_matched_cohort.csv"
        )

    matched = pd.read_csv(MATCHED_CSV)
    matched = matched[["NAKO_ID", "label"]].copy()

    feature_cols = get_feature_cols(feature_mode)

    if len(feature_cols) > 0:
        ana = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
        matched = matched.merge(
            ana[["NAKO_ID"] + feature_cols],
            on="NAKO_ID",
            how="left",
        )

    print("=" * 60, flush=True)
    print("Loading matched cohort", flush=True)
    print(f"Before file check: {len(matched)} subjects", flush=True)
    print(matched["label"].value_counts().to_string(), flush=True)

    out = _check_required_files(
        df=matched,
        input_mode=input_mode,
        feature_mode=feature_mode,
    )

    print(f"\nAfter file check: {len(out)} subjects", flush=True)
    print(out["label"].value_counts().to_string(), flush=True)
    print("=" * 60, flush=True)

    return out


def load_full_dataset(input_mode="fat_water", feature_mode="none"):
    """
    Load all T2D vs normal subjects.
    """
    df = pd.read_csv(ANALYSIS_CSV, sep=";", decimal=",")
    df = df[df["diabetes_group"].isin(["normal", "T2D"])].copy()
    df["label"] = (df["diabetes_group"] == "T2D").astype(int)

    feature_cols = get_feature_cols(feature_mode)
    keep_cols = ["NAKO_ID", "label"] + feature_cols
    df = df[keep_cols].copy()

    print("=" * 60, flush=True)
    print("Loading full cohort", flush=True)
    print(f"Before file check: {len(df)} subjects", flush=True)
    print(df["label"].value_counts().to_string(), flush=True)

    out = _check_required_files(
        df=df,
        input_mode=input_mode,
        feature_mode=feature_mode,
    )

    print(f"\nAfter file check: {len(out)} subjects", flush=True)
    print(out["label"].value_counts().to_string(), flush=True)
    print("=" * 60, flush=True)

    return out


def load_nifti_dataset(
    cohort="matched",
    input_mode="fat_water",
    feature_mode="none",
):
    if cohort == "matched":
        return load_matched_dataset(
            input_mode=input_mode,
            feature_mode=feature_mode,
        )

    if cohort == "full":
        return load_full_dataset(
            input_mode=input_mode,
            feature_mode=feature_mode,
        )

    raise ValueError(f"Unknown cohort: {cohort}")