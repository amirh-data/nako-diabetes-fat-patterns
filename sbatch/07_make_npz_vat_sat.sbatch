#!/bin/bash
#SBATCH --job-name=npz_vatsat
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail

echo "Job started on $(hostname)"
echo "Time: $(date)"

# Activate conda
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/nnunetv2_env

# Avoid broken ~/.local
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Go to repo
cd ~/nako-diabetes-fat-patterns

# Run preprocessing
python -u -m scripts.16_make_3d_npz_vat_sat

echo "Finished at $(date)"
