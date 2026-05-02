#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --output=samples/slurm-%j.out
#SBATCH --error=samples/slurm-%j.err
#SBATCH --job-name=sample_cifar_disco_unconditional

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

outdir="samples_cifar_edm_masked_unconditional"
mkdir -p "$outdir"

# Best EDM Masked Unconditional model snapshot for sampling (Table 3 in the paper)
SNAPSHOT_PATH="training-runs/00061-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch224-fp32-noise_uncond=False-beta1=0.9-beta2=0.999-lr=0.001-disco=False-t0=0.01-t_independent_cs=False-gamma=0.5/network-snapshot-185651.pkl"

# Draw samples
torchrun \
    --standalone \
    --nproc_per_node=4 \
    generate.py \
    --outdir="$outdir" \
    --seeds=0-49999 \
    --batch=2048 \
    --network="$SNAPSHOT_PATH"

# Calculate FID, which should be ~2.59 (Table 3 in the paper)
torchrun \
    --standalone \
    --nproc_per_node=4 \
    fid.py calc \
    --images="$outdir" \
    --ref=fid-refs/cifar10-32x32.npz
