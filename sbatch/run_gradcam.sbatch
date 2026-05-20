#!/bin/bash
#SBATCH --job-name=gradcam
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.out
#SBATCH --error=/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/logs/%x_%j.err

set -eo pipefail
echo "Job started on $(hostname)"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/envs/thesis_image_env2

unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd ~/nako-diabetes-fat-patterns

python -u -m scripts.43_gradcam \
    --n_subjects 20 \
    --input_mode fat_water_seg_trunk_binary \
    --target_layer layer3 \
    --num_workers 4

echo "Finished at $(date)"