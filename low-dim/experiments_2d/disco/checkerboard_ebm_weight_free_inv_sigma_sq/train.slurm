#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=experiments_2d/disco/checkerboard_ebm_weight_free_inv_sigma_sq/slurm-%j.out
#SBATCH --error=experiments_2d/disco/checkerboard_ebm_weight_free_inv_sigma_sq/slurm-%j.err
#SBATCH --job-name=checkerboard_disco_ebm_weight_free_inv_sigma_sq

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
   --config=configs/ve/ncsnv2/checkerboard_disco.py \
   --config.model.loss_variant="inv_sigma_sq_weighting" \
   --config.training.weight_free_disco=True \
   --config.model.net_output="sigma_times_eps" \
   --config.training.n_iters=200001 \
   --config.training.ebm=True \
   --config.model.name="ncsnv2_lowdim_ebm" \
   --config.model.nf=128 \
   --experiment_name="checkerboard_disco_ebm_weight_free_inv_sigma_sq"
