#!/bin/bash
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --job-name=eval_2d_models
#SBATCH --array=0-2

# Activate conda environment if needed
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jax_cuda

PYTHON_CMD="python"
if [ "$1" == "--debug" ]; then
    export WANDB_MODE="disabled"
    PYTHON_CMD="python -m debugpy --listen 5678"
fi

# Select dataset based on array task ID
DATASETS=("moons" "checkerboard" "rings")
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  DATASET=""
  for arg in "$@"; do
    case "$arg" in
      --dataset=*) DATASET="${arg#--dataset=}" ;;
    esac
  done
  if [ -z "$DATASET" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set and --dataset was not provided." >&2
    exit 1
  fi
else
  DATASET_INDEX=$SLURM_ARRAY_TASK_ID
  DATASET=${DATASETS[$DATASET_INDEX]}
fi

# Ensure output directory exists and run, teeing to a dataset-specific log
mkdir -p ./experiment_results
echo "Running experiment for $DATASET"


# Check model fit: Wasserstein distance between p* and the model (joint)
$PYTHON_CMD ./all_d_cond_sampling_exp.py \
  --dataset="$DATASET" \
  --num-conditionals 30 \
  --dim 2 \
  --num-samples 10000 `# anything larger than this cannot be computed with non-sliced W1` \
  --num-runs-per-cond 1 \
  --step 20 \
  --condition-noise-std=0.0 \
  --num-dim-to-sample=2 \
  --num-mcmc-steps=2 \
  --num-integration-steps=3 \
  --no-wasserstein-slicing=True \
  --model-fit=True \
  --save-dir=./experiment_results/${DATASET}


# Check inference quality: Measure deviation between true model conditional, and the conditional sampled from the model
$PYTHON_CMD ./all_d_cond_sampling_exp.py \
  --dataset="$DATASET" \
  --num-conditionals 30 \
  --dim 2 \
  --num-samples 1000000 `# we need a lot of joint samples to slice out conditionals` \
  --num-runs-per-cond 1 \
  --step 20 \
  --condition-noise-std=0.0 \
  --num-dim-to-sample=1 \
  --num-mcmc-steps=2 \
  --num-integration-steps=3 \
  --no-wasserstein-slicing=True \
  --inference-divergence=True \
  --save-dir=./experiment_results/${DATASET}