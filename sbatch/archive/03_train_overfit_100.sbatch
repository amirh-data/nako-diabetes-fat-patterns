#!/bin/bash
#SBATCH --job-name=overfit100
#SBATCH --partition=student
#SBATCH --nodelist=node-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -u -m scripts.09_train_overfit_100

echo "Finished at $(date)"
