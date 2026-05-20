#!/bin/bash
#SBATCH --job-name=vatsat_eval
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail
echo "Job started on $(hostname)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/thesis_image_env2

unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd ~/nako-diabetes-fat-patterns

echo "Step 1: Extract embeddings"
python -u -m scripts.36_extract_vatsat_embeddings \
    --model_name convnext3d \
    --input_mode fat_water \
    --batch_size 32 \
    --num_workers 8

echo "Step 2: Train MLP classifier"
python -u -m scripts.37_train_mlp_on_embeddings \
    --model_name convnext3d \
    --input_mode fat_water

echo "Finished at $(date)"