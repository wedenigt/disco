# Low Dimensional DISCO Experiments

This code repository is heavily based on [https://github.com/yang-song/score_sde](https://github.com/yang-song/score_sde).

To reproduce Table 1, train one DISCO model for each of the three datasets (Moons, Checkerboard, Rings) by running

```bash
bash experiments_2d/disco/moons_ebm_weight_free_inv_sigma_sq/train.slurm
bash experiments_2d/disco/checkerboard_ebm_weight_free_inv_sigma_sq/train.slurm
bash experiments_2d/disco/rings_ebm_weight_free_inv_sigma_sq/train.slurm
```
from within this directory (`low-dim`).

Analogously, train the respective diffusion model baselines by running
```bash
bash experiments_2d/no_disco/moons_ebm_eps_no_sigma_weights/train.slurm
bash experiments_2d/no_disco/checkerboard_ebm_eps_no_sigma_weights/train.slurm
bash experiments_2d/no_disco/rings_ebm_eps_no_sigma_weights/train.slurm
```
If you use slurm, replace `bash` by `sbatch` and modify the `#SBATCH` definitions in the `*.slurm` files.

After training all models, run the evaluation via
```bash
bash experiments_2d/eval_2d_models.slurm --dataset moons
bash experiments_2d/eval_2d_models.slurm --dataset checkerboard
bash experiments_2d/eval_2d_models.slurm --dataset rings
```
which will use the most recently trained models in the `checkpoints` folder to run the evaluation.
This script will print the numbers shown in Table 1 (both "Model Fit" and "Inference Quality").

If you use `slurm`, running
```bash
sbatch experiments_2d/eval_2d_models.slurm
```
will parallelize evaluation over all datasets.