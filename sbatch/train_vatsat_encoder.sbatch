#!/bin/bash
#SBATCH --job-name=vatsat_encoder
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

python -u -m scripts.33_train_vatsat_encoder \
    --input_mode fat_water \
    --model_name resnet10_3d \
    --epochs 40 \
    --lr 1e-3 \
    --batch_size 16 \
    --num_workers 4 \
    --augment

echo "Finished at $(date)"
