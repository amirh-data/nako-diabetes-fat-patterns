#!/bin/bash
#SBATCH --job-name=cache_full128
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=100:00:00
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

python -u -m scripts.41b_cache_nifti128_full \
    --resolution 128 \
    --num_workers 12

echo "Finished at $(date)"
