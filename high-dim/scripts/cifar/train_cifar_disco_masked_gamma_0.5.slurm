#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=training-runs/slurm/slurm-%j.out
#SBATCH --error=training-runs/slurm/slurm-%j.err
#SBATCH --job-name=train_cifar_disco_masked_gamma_0.5

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

PYTHON_CMD="python"
if [ "$1" == "--debug" ]; then
    echo "debugging..."
    PYTHON_CMD="python -m debugpy --listen 5678"
fi

if [ "$1" == "--dry-run" ]; then
    export WANDB_MODE="disabled"
    echo "WandB disabled for dry run"
else
    export WANDB_MODE="online"
    echo "WandB enabled"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('device_count=', torch.cuda.device_count())"

# Set nproc_per_node to the number of available GPUs detected by torch
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using nproc_per_node=$NPROC_PER_NODE"

TRAIN_ARGS=( \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --cond=0 \
    --arch=ddpmpp \
    --disco=True \
    --noise-unconditional=True \
    --beta2=0.95 \
    --t0=0.01 \
    --gamma=0.5 \
    --t_independent_cs=True \
    --predict_cs=True \
    --batch=224 `# set to 224 when training` \
)

if [ "$1" == "--debug" ]; then
    $PYTHON_CMD train.py --no-compile "${TRAIN_ARGS[@]}"
else
    torchrun \
        --standalone \
        --nproc_per_node=$NPROC_PER_NODE \
        train.py "${TRAIN_ARGS[@]}"
fi