#!/bin/bash

#SBATCH --job-name=nifti_train_fat_water_seg_trunk_binary_distribution
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=100:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail

# ============================================================
# CHANGE ONLY THESE LINES
# ============================================================

DATA_BACKEND="cached"
CACHE_DIR="/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/cached_nifti128/matched_fat_water_trunkseg"

INPUT_MODE="fat_water_seg_trunk_binary"
MODEL="resnet10_3d"
COHORT="matched"
FEATURE_MODE="distribution" 

RESOLUTION=128
EPOCHS=30
LR=1e-3
BATCH_SIZE=16
ACCUM_STEPS=1
NUM_WORKERS=8
VAL_EVERY=1
PATIENCE=8
AUGMENT="no"
OVERFIT_N=0
RESUME_FROM=""

# ============================================================

echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Data backend: ${DATA_BACKEND}"
echo "Cache dir: ${CACHE_DIR}"
echo "Input mode: ${INPUT_MODE}"
echo "Model: ${MODEL}"
echo "Cohort: ${COHORT}"
echo "Feature mode: ${FEATURE_MODE}"
echo "Resolution: ${RESOLUTION}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Accum steps: ${ACCUM_STEPS}"
echo "Num workers: ${NUM_WORKERS}"
echo "Augment: ${AUGMENT}"
echo "Overfit N: ${OVERFIT_N}"
echo "Resume from: ${RESUME_FROM}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/thesis_image_env2

unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd ~/nako-diabetes-fat-patterns

CMD="python -u -m scripts.39_train_nifti \
    --input_mode ${INPUT_MODE} \
    --model_name ${MODEL} \
    --cohort ${COHORT} \
    --feature_mode ${FEATURE_MODE} \
    --data_backend ${DATA_BACKEND} \
    --cache_dir ${CACHE_DIR} \
    --resolution ${RESOLUTION} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --val_every ${VAL_EVERY} \
    --patience ${PATIENCE} \
    --overfit_n ${OVERFIT_N}"

if [ "${AUGMENT}" = "yes" ]; then
    CMD="${CMD} --augment"
fi

if [ -n "${RESUME_FROM}" ]; then
    CMD="${CMD} --resume_from ${RESUME_FROM}"
fi

echo "Running command:"
echo "${CMD}"

eval ${CMD}


echo "Finished at $(date)"
