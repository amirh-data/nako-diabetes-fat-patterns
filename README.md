# nako-diabetes-fat-patterns

This repository contains code for master thesis experiments on image-based and biomarker-based analysis of type 2 diabetes-related fat distribution patterns in NAKO Dixon MRI data.

## Main idea

The project investigates whether MRI-derived fat distribution patterns can help distinguish subjects with type 2 diabetes (T2D) from normal controls.

The current experiments include:

1. 3D image encoder for normal-vs-T2D classification.
2. VAT/SAT image-encoder sanity task.
3. Image embedding extraction.
4. Biomarker-only and clinical-only baselines.
5. Combined image embedding + tabular biomarker classifiers.
6. Regional pixel/cluster feature extraction.

## Data

Data are not included in this repository.

The scripts expect data and outputs on the cluster under paths such as:

FAT_IMAGE_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_repaired_input"
)

SEGMENTATION_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/dataset301_full_output"
)

OUT_CSV = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/
)
