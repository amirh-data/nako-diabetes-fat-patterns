#!/bin/bash
#SBATCH --job-name=reg_all_t2d
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail

echo "Job started on $(hostname)"
echo "Time: $(date)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/nnunetv2_env

unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd ~/nako-diabetes-fat-patterns

python -u -m scripts.23_extract_regional_features_all_t2d

echo "Finished at $(date)"
