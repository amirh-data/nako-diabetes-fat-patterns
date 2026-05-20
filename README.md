# AI-Based Identification of Diabetes-Related Fat Distribution Patterns
# in Trunk Using MR Images from a Large Population-Based Study

Master Thesis — University Hospital Tübingen  
Supervisors: Prof. Dr. Thomas Küstner & Prof. Dr. Jürgen Machann  
Expected submission: July 14, 2026

## Overview

Deep learning pipeline for T2D vs normal classification from whole-body
Dixon MRI using the NAKO cohort (~24K subjects). Systematic comparison
of supervised, fusion, proxy-task, and contrastive learning approaches
on a demographically matched cohort.

## Key Scientific Finding

After age/BMI/sex matching (N=3,040: 1,520 T2D vs 1,520 normal),
all methods converged to AUC ~0.55, confirming that the performance
ceiling is biological rather than methodological. Only cardiac adipose
tissue (p=0.018) and back muscle fat fraction (p=0.044) showed
statistically significant differences after matching. Vertebral marrow
fat differences observed in the full cohort disappeared after matching,
confirming demographic confounding.

## Results

### Full Cohort (N=24,112, unmatched)
| Method | Model | AUC |
|--------|-------|-----|
| Clinical only (age/BMI/sex) | LR | 0.822 |
| Global biomarkers only | LR | 0.521 |
| Clinical + biomarkers | LR | 0.822 |

### Matched Cohort (N=3,040, age/BMI/sex controlled)
| Method | Model | AUC |
|--------|-------|-----|
| Clinical only | LR | 0.481 (matching confirmed) |
| Global biomarkers only | LR | 0.555 |
| Distribution only (L1-Th1) | LR | 0.526 |
| Global + distribution | LR | 0.545 |
| Image only (fat+water+seg) | ResNet10 | 0.551 |
| Image + biomarkers fusion | ResNet10+MLP | 0.543 |
| Proxy encoder embeddings | ConvNeXt+MLP | 0.521 |
| CLIP frozen embeddings | ResNet10+MLP | pending |
| CLIP fine-tuned | ResNet10 | pending |

## Pipeline

### Step 1: Preprocessing
Cache raw NIfTI files to 128 cube PyTorch tensors for fast training.
- scripts/41_cache_nifti128_matched.py
- scripts/41b_cache_nifti128_full.py

### Step 2: Tabular Baselines
Logistic regression on global biomarkers and vertebral distribution features.
SHAP feature importance analysis.
- scripts/11b_biomarker_baseline_matched.py
- scripts/42_distribution_features_matched.py
- scripts/run_shap_lr.py

### Step 3: Supervised CNN
ResNet10/ConvNeXt3D on cached 128 cube volumes.
Multiple input modes and fusion variants.
- scripts/39_train_nifti.py

### Step 4: Proxy Task Pretraining
ConvNeXt3D trained on biomarker regression as pretraining task.
Embeddings used for T2D classification.
- scripts/33_train_vatsat_encoder.py
- scripts/36_extract_vatsat_embeddings.py
- scripts/37_train_mlp_on_embeddings.py

### Step 5: CLIP Contrastive Learning
ResNet10 image encoder aligned with tabular MLP encoder via InfoNCE loss.
Full cohort pretraining with matched cohort evaluation to prevent leakage.
- scripts/40_train_clip.py

### Step 6: Interpretability
Grad-CAM spatial attention maps and SHAP feature importance.
- scripts/43_gradcam.py
- scripts/run_shap_lr.py

## Repository Structure

    scripts/
      11b_biomarker_baseline_matched.py   Tabular LR baseline on matched cohort
      33_train_vatsat_encoder.py          Proxy task pretraining (biomarker regression)
      36_extract_vatsat_embeddings.py     Extract proxy encoder embeddings
      37_train_mlp_on_embeddings.py       MLP classifier on proxy embeddings
      39_train_nifti.py                   Main supervised CNN training script
      40_train_clip.py                    CLIP contrastive learning (Phase 1,2,3)
      41_cache_nifti128_matched.py        Cache matched cohort NIfTI to tensors
      41b_cache_nifti128_full.py          Cache full cohort NIfTI to tensors
      42_distribution_features_matched.py Distribution feature analysis (L1-Th1)
      43_gradcam.py                       Grad-CAM visualization
      create_matched_cohort.py            Age/BMI/sex matched cohort creation
      run_shap_lr.py                      SHAP feature importance analysis

    src/
      models/
        supervised_3d_models.py           ResNet10_3D, ConvNeXt3D architectures
        vatsat_encoder.py                 Proxy encoder wrapper
      data/
        nifti_dataset.py                  Main dataset class (cached + direct NIfTI)
        supervised_npz_dataset.py         NPZ dataset for supervised training
        regression_npz_dataset.py         NPZ dataset for regression pretraining
        mask_only_dataset.py              Segmentation mask dataset

    sbatch/                               SLURM job submission scripts

## Technical Stack

- PyTorch, TorchIO, scikit-learn, SHAP, matplotlib
- SLURM/HPC cluster with GPU nodes
- SimpleITK for NIfTI processing
- ResNet10_3D and ConvNeXt3D 3D CNN architectures
- InfoNCE contrastive loss for CLIP-style alignment
- Grad-CAM for 3D spatial interpretability

## Data

Data not included (NAKO cohort, restricted access).
Model checkpoints not included (file size).
Please contact supervisors for data access information.
