#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=62
#SBATCH --gres=gpu:8
#SBATCH --output=training-runs/slurm/slurm-%j.out
#SBATCH --error=training-runs/slurm/slurm-%j.err
#SBATCH --job-name=train_ffhq_disco_masked_gamma_0.5

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

PYTHON_CMD="python"
if [ "$1" == "--debug" ]; then
    export JAX_DISABLE_JIT=1 # disable JIT for debugging
    echo "$JAX_DISABLE_JIT"
    PYTHON_CMD="python -m debugpy --listen 5678"
fi

if [ "$1" == "--dry-run" ]; then
    export WANDB_MODE="disabled"
    echo "WandB disabled for dry run"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('device_count=', torch.cuda.device_count())"

# Set nproc_per_node to the number of available GPUs detected by torch
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using nproc_per_node=$NPROC_PER_NODE"

torchrun \
    --standalone \
    --nproc_per_node=$NPROC_PER_NODE \
    train.py \
    --outdir=training-runs \
    --data=datasets/ffhq-64x64.zip \
    --cond=0 \
    --arch=ddpmpp \
    --cres=1,2,2,2 \
    --lr=2e-4 \
    --dropout=0.05 \
    --augment=0.15 \
    --disco=True \
    --noise-unconditional=True \
    --beta2=0.95 \
    --t0=0.01 \
    --gamma=0.5 \
    --t_independent_cs=True \
    --predict_cs=True \
    --batch=112
