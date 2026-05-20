#!/bin/bash

#SBATCH --job-name=cache_trunk128
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail

echo "Job started on $(hostname)"
echo "Time: $(date)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/thesis_image_env2

unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd ~/nako-diabetes-fat-patterns

python -u -m scripts.41_cache_nifti128_matched \
    --resolution 128 \
    --num_workers 8

echo "Finished at $(date)"
