# High-dimensional DISCO Experiments

This code is based on the great [EDM GitHub Repository](https://github.com/NVlabs/edm). Check their `README.md` for installation instructions.

## Training DISCO and EDM-Masked models

Run the training scripts in `scripts`, again via `sbatch` or locally with a shell (e.g., `bash`).

## EDM Models

We use the pre-trained models `edm-cifar10-32x32-uncond-vp.pkl` for CIFAR-10 and `edm-ffhq-64x64-uncond-vp.pkl` for FFHQ-64, both of which can be downloaded using the instructions here: [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm).

## Unconditional Sampling

Run the sampling scripts in `scripts` via `sbatch`/`bash`.

## Inpainting Results

Run the scripts in `inpainting` via `sbatch`/`bash`.
