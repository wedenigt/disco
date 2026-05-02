#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=training-runs/slurm/slurm-%j.out
#SBATCH --error=training-runs/slurm/slurm-%j.err
#SBATCH --job-name=train_cifar_nodisco_edm_masked_gamma_0.5

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

# Set rendezvous (master) address and port for multi-node training
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

# Set nproc_per_node to the number of available GPUs detected by torch (should be 4)
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using nnodes=$SLURM_NNODES nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

# Launch one torchrun per node via srun
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id=$SLURM_JOB_ID \
    train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --cond=0 \
    --arch=ddpmpp \
    --disco=False \
    --noise-unconditional=False \
    --t0=0.01 \
    --gamma=0.5 \
    --edm-masked-conditional-training=True \
    --batch=224