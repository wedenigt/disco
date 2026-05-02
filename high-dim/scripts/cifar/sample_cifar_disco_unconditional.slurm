#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --output=samples/slurm-%j.out
#SBATCH --error=samples/slurm-%j.err
#SBATCH --job-name=sample_cifar_disco_unconditional

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

outdir="samples_cifar_disco_unconditional"
mkdir -p "$outdir"

# Draw samples
torchrun \
    --standalone \
    --nproc_per_node=8 \
    generate.py \
    --outdir="$outdir" \
    --seeds=0-49999 \
    --batch=1024 \
    --rho 2.5 \
    --network=training-runs/00041-cifar10-32x32-uncond-ddpmpp-edm-gpus7-batch224-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01-t_independent_cs=True-gamma=0.5/network-snapshot-135475.pkl # this snapshot gave us the best results

# Calculate FID, which should be ~3.58 (Table 3 in the paper)
torchrun \
    --standalone \
    --nproc_per_node=8 \
    fid.py calc \
    --images="$outdir" \
    --ref=fid-refs/cifar10-32x32.npz