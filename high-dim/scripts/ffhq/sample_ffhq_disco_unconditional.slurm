#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --output=samples/slurm-%j.out
#SBATCH --error=samples/slurm-%j.err
#SBATCH --job-name=sample_ffhq_disco_unconditional

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

outdir="samples_ffhq_disco_unconditional"
mkdir -p "$outdir"

# Snapshot 160204 gave us the best results, so we use that one for sampling.
SNAPSHOT_PATH="training-runs/00068-ffhq-64x64-uncond-ddpmpp-edm-gpus8-batch112-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.0002-disco=True-t0=0.01-t_independent_cs=True-gamma=0.5/network-snapshot-160204.pkl"

# Draw samples
torchrun \
    --standalone \
    --nproc_per_node=8 \
    generate.py \
    --outdir="$outdir" \
    --seeds=0-49999 \
    --batch=512 \
    --steps=40 \
    --network="$SNAPSHOT_PATH"

# Calculate FID
torchrun \
    --standalone \
    --nproc_per_node=8 \
    fid.py calc \
    --images="$outdir" \
    --ref=fid-refs/ffhq-64x64.npz