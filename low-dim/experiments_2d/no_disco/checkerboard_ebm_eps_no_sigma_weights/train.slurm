#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=experiments_2d/no_disco/checkerboard_ebm_eps_no_sigma_weights/slurm-%j.out
#SBATCH --error=experiments_2d/no_disco/checkerboard_ebm_eps_no_sigma_weights/slurm-%j.err
#SBATCH --job-name=checkerboard_no_disco_ebm_eps_no_sigma_weights

# Activate conda environment if needed
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jax_cuda
WORKDIR="${HOME}/PhD/Research/disco-model/low-dim"

PYTHON_CMD="python"
if [ "$1" == "--debug" ]; then
    export WANDB_MODE="disabled"
    PYTHON_CMD="python -m debugpy --listen 5678"
fi

if [ "$1" == "--dry-run" ]; then
    export WANDB_MODE="disabled"
    echo "WandB disabled for dry run"
fi

# Run the training script
$PYTHON_CMD ./main.py \
   --mode train \
   --workdir "$WORKDIR" \
   --config=configs/ve/ncsnv2/checkerboard.py \
   --config.model.net_output="eps" \
   --config.training.n_iters=200001 \
   --config.training.ebm=True \
   --config.model.name="ncsnv2_lowdim_ebm" \
   --config.model.nf=128 \
   --experiment_name="checkerboard_no_disco_ebm_eps_no_sigma_weights"