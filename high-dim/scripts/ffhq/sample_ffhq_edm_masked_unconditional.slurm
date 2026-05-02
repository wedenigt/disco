#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:5
#SBATCH --output=samples/slurm-00069-%j.out
#SBATCH --error=samples/slurm-00069-%j.err
#SBATCH --job-name=sample_00069_fid

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

echo "[FFHQ] Evaluating snapshot ${SNAP}"
TMP_DIR="samples/edm_masked_ffhq/"
# Snapshot 25160 gave us the best results (FID of ~5.71)
SNAPSHOT_PATH="training-runs/00069-ffhq-64x64-uncond-ddpmpp-edm-gpus4-batch112-fp32-noise_uncond=False-beta1=0.9-beta2=0.999-lr=0.0002-disco=False-t0=0.01-t_independent_cs=False-gamma=0.5/network-snapshot-125160.pkl"

# Make sure the tmp dir is clean
mkdir -p "$TMP_DIR"

# Draw samples
torchrun \
    --standalone \
    --nproc_per_node=5 \
    generate.py \
    --outdir="$TMP_DIR" \
    --seeds=0-49999 \
    --batch=512 \
    --steps=40 \
    --network="$SNAPSHOT_PATH"

# Calculate FID
torchrun \
    --standalone \
    --nproc_per_node=5 \
    fid.py calc \
    --images="$TMP_DIR" \
    --ref=fid-refs/ffhq-64x64.npz

echo "--------------------------------"