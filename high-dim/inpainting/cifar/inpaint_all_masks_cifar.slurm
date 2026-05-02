#!/bin/bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=inpainting/cifar/slurm-cifar-%A_%a.out
#SBATCH --error=inpainting/cifar/slurm-cifar-%A_%a.err
#SBATCH --job-name=inpaint_all_masks_cifar

# Activate conda environment if needed
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate edm

OUTDIR="inpainting/cifar"
EDM_PKL="./training-runs/edm-cifar10-32x32-uncond-vp/edm-cifar10-32x32-uncond-vp.pkl"
MASK_TRAINED_EDM_PKL="./training-runs/00061-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch224-fp32-noise_uncond=False-beta1=0.9-beta2=0.999-lr=0.001-disco=False-t0=0.01-t_independent_cs=False-gamma=0.5/network-snapshot-100352.pkl"

# Original DISCO as in UAI paper
DISCO_NETWORK_PKL="./training-runs/00041-cifar10-32x32-uncond-ddpmpp-edm-gpus7-batch224-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01-t_independent_cs=True-gamma=0.5/network-snapshot-100352.pkl"

# Select mask based on array task ID
MASKS=("wide" "narrow" "sr2x" "alt_lines" "expand" "half")
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set; defaulting to index 0"
  SLURM_ARRAY_TASK_ID=0
fi
MASK_INDEX=$SLURM_ARRAY_TASK_ID
MASK=${MASKS[$MASK_INDEX]}

STEPS=18
NON_EDM_STEPS=36 # to have the same NFE for non-EDM samplers
REPAINT_STEPS=9
NUM_IMAGES=1024
case "$MASK" in
    "wide")
        RHO=1.5
        ;;
    "narrow")
        RHO=1.5
        ;;
    "sr2x")
        RHO=1.8
        ;;
    "alt_lines")
        RHO=1.8
        ;;
    "expand")
        RHO=7
        ;;
    "half")
        RHO=7
        ;;
    *)
        RHO=7
        ;;
esac

echo "Running inpainting experiment for mask: $MASK"
echo "----------------------------------------"

echo "Original EDM model, replacement heuristic:"
OUTDIR_IMGS="${OUTDIR}/original_edm_${MASK}_replacement/"
mkdir -p "${OUTDIR_IMGS}"
python ./inpaint.py \
    --steps $NON_EDM_STEPS \
    --num_images $NUM_IMAGES \
    --mask "$MASK" \
    --network_pkl "${EDM_PKL}" \
    --no_plot \
    --solver=euler \
    --save_images \
    --disc=vp \
    --schedule=vp \
    --scaling=vp \
    --ancestral_sampling \
    --outdir "${OUTDIR_IMGS}"

echo "----------------------------------------"
echo "Original EDM model, Repaint-4:"
OUTDIR_IMGS="${OUTDIR}/original_edm_${MASK}_repaint_20/"
mkdir -p "${OUTDIR_IMGS}"
python ./inpaint.py \
    --steps $REPAINT_STEPS \
    --num_images $NUM_IMAGES \
    --mask "$MASK" \
    --network_pkl "${EDM_PKL}" \
    --no_plot \
    --solver=euler \
    --save_images \
    --disc=vp \
    --schedule=vp \
    --scaling=vp \
    --ancestral_sampling \
    --repaint_enable \
    --repaint_jump 1 \
    --repaint_repeats 4 \
    --outdir "${OUTDIR_IMGS}"

echo "----------------------------------------"
echo "Original EDM model, gradient guidance:"
OUTDIR_IMGS="${OUTDIR}/original_edm_${MASK}_gradient_guidance/"
mkdir -p "${OUTDIR_IMGS}"
python ./inpaint.py \
    --steps $NON_EDM_STEPS \
    --num_images $NUM_IMAGES \
    --mask "$MASK" \
    --network_pkl "${EDM_PKL}" \
    --no_plot \
    --solver=euler \
    --save_images \
    --disc=vp \
    --schedule=vp \
    --scaling=vp \
    --ancestral_sampling \
    --gradient_guidance \
    --outdir "${OUTDIR_IMGS}"


echo "----------------------------------------"
echo "Mask-trained EDM model, no replacement heuristic:"
OUTDIR_IMGS="${OUTDIR}/mask_trained_edm_${MASK}/"
mkdir -p "${OUTDIR_IMGS}"
python ./inpaint.py \
    --steps $STEPS \
    --num_images $NUM_IMAGES \
    --mask "$MASK" \
    --network_pkl "${MASK_TRAINED_EDM_PKL}" \
    --no_plot \
    --save_images \
    --outdir "${OUTDIR_IMGS}"


echo "----------------------------------------"
echo "DISCO Model, no replacement heuristic:"
OUTDIR_IMGS="${OUTDIR}/disco_${MASK}/"
mkdir -p "${OUTDIR_IMGS}"
python ./inpaint.py \
    --steps $STEPS \
    --num_images $NUM_IMAGES \
    --mask "$MASK" \
    --network_pkl "${DISCO_NETWORK_PKL}" \
    --no_plot \
    --save_images \
    --rho $RHO \
    --outdir "${OUTDIR_IMGS}"