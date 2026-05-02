import sys
import os

sys.path.append(os.path.abspath(".."))
# disable jax preallocation of gpu memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import glob
import shutil
import subprocess


def _has_cuda_device():
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "GPU " in result.stdout:
                return True
        except Exception:
            pass

    return len(glob.glob("/dev/nvidia[0-9]*")) > 0


if not _has_cuda_device():
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from cond_sampling import (
    load_model,
    get_latest_model,
    setup_sde,
    sample,
    get_score_function,
)
from gmm_utils_sampling import get_latest_checkpoints
from gmm_utils import compute_wasserstein_distance, sliced_wasserstein
import datasets
from cond_sampling import plot_conditional_distributions, get_smc_sampler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from tqdm import tqdm
import copy
from gmm_utils import eval_gmm_conditional, sample_conditional_gmm
from exact_ebm_sampling import ar_sample as ar_sample_fn
from exact_ebm_sampling import (
    sample_1d_energy,
    plot_1d_density_and_hist,
)

checkpoint_path = Path("./checkpoints").absolute()


def exact_1d_sampling_fn(
    neg_energy_fn,
    rng,
    num_samples,
    exact_samples,
    means,
    covs,
    weights,
    cond_values,
    cond_indices,
    out_path,
    title,
    min_val=-2,
    max_val=2,
    grid_size=100_000,
    no_wasserstein_slicing=False,
):
    rng, key_1d = jax.random.split(rng)
    xs, pdf_1d, _cdf_1d, exact_1d_samples = sample_1d_energy(
        neg_energy_fn,
        key_1d,
        n_draws=num_samples,
        grid_size=grid_size,
        min_val=min_val,
        max_val=max_val,
    )

    if means is not None and covs is not None and weights is not None:
        true_log_gmm = jax.vmap(
            lambda x: eval_gmm_conditional(
                x, means, covs, weights, cond_values, cond_indices
            )
        )(jnp.linspace(min_val, max_val, grid_size))
        true_pdf_1d = jnp.exp(true_log_gmm)
    else:
        true_pdf_1d = None

    plot_1d_density_and_hist(
        xs,
        pdf_1d,
        exact_1d_samples,
        out_path=out_path,
        title=title,
        true_pdf=true_pdf_1d,
    )
    # compute sliced w1
    rng, rng_1 = jax.random.split(rng)
    w1_exact_1d = sliced_wasserstein(
        exact_1d_samples.reshape(-1, 1),
        exact_samples,
        rng_1,
        no_slicing=no_wasserstein_slicing,
    )
    return w1_exact_1d


def nodisco_exps(
    state,
    config_pc,
    config_tds,
    score_model,
    sde,
    sampling_eps,
    cond_values,
    cond_indices,
    exact_samples,
    dim,
    num_dim_to_sample,
    k,
    rng=None,
    save_dir=None,
    means=None,
    covs=None,
    weights=None,
    sigma_min=None,
    save_disco_energy=False,
    ar_sample=False,
    exact_1d_sampling=False,
    no_wasserstein_slicing=False,
    box_min=None,
    box_max=None,
):
    num_samples = exact_samples.shape[0]
    rng = jax.random.PRNGKey(42) if rng is None else rng
    sampling_shape = (num_samples, config_pc.data.num_channels)

    # Replacement Heuristic
    cond_indices_heuristic = jnp.repeat(
        cond_indices[None, ...], sampling_shape[0], axis=0
    )
    cond_values_heuristic = jnp.repeat(
        cond_values[None, ...], sampling_shape[0], axis=0
    )
    cond_samples_heuristic, _, all_samples, _ = sample(
        config_pc,
        sde,
        score_model,
        state,
        sampling_shape,
        rng,
        sampling_eps,
        cond_indices=cond_indices_heuristic,
        cond_values=cond_values_heuristic,
        heuristic_cond_sampling=True,
        store_intermediate_samples=False,
        guidance_alpha=0.0,
    )

    ar_samples = None
    w1_replacement_ar = None
    _neg_energy_fn = get_smc_sampler(
        config_pc,
        sde,
        score_model,
        state,
        sampling_shape,
        rng,
        cond_indices=cond_indices,
        cond_values=cond_values,
        num_integration_steps=None,
        num_mcmc_steps=None,
        step_size=None,
        disco=False,
        get_just_energy=True,
    )
    neg_energy_fn = lambda x: _neg_energy_fn(x[None, ...])  # [0].squeeze()

    if ar_sample:
        ar_samples, acc_rate, logM = ar_sample_fn(
            rng,
            n_samples=num_samples,
            neg_energy_fn=neg_energy_fn,
            dim=dim,
            box_min=box_min,
            box_max=box_max,
        )
        rng, rng_1 = jax.random.split(rng)
        w1_replacement_ar = sliced_wasserstein(
            ar_samples, exact_samples, rng_1, no_slicing=no_wasserstein_slicing
        )
        print(
            f"[NoDISCO AR] W1: {w1_replacement_ar:.4f}, Acceptance rate: {acc_rate:.3f}, LogM: {logM:.3f}"
        )

    if num_dim_to_sample == 1 and exact_1d_sampling:
        # Perform exact 1D sampling along the free dimension and save a plot
        w1_exact_1d = exact_1d_sampling_fn(
            neg_energy_fn,
            rng,
            num_samples,
            exact_samples,
            means,
            covs,
            weights,
            cond_values,
            cond_indices,
            out_path=f"./plots/1d_density_nodisco.png",
            title="1D density and histogram",
            no_wasserstein_slicing=no_wasserstein_slicing,
        )
        print(f"[NoDISCO Exact 1D] W1: {w1_exact_1d:.4f}")

    energy = None
    true_log_gmm = None
    grid_points = None
    if save_disco_energy:
        # Create a grid of points between box_min and box_max in each dimension
        grid_points = jnp.linspace(box_min, box_max, 100)
        grid_points = jnp.meshgrid(*[grid_points] * num_dim_to_sample)
        grid_points = jnp.stack(grid_points, axis=-1)

        # Evaluate the energy of the DISCO model on the grid of points
        energy = jax.vmap(neg_energy_fn)(grid_points.reshape(-1, grid_points.shape[-1]))
        energy = energy.reshape(*grid_points.shape[:-1])

        if means is not None and covs is not None and weights is not None:
            true_log_gmm = jax.vmap(
                lambda x: eval_gmm_conditional(
                    x, means, covs, weights, cond_values, cond_indices
                )
            )(grid_points.reshape(-1, grid_points.shape[-1]))
            true_log_gmm = true_log_gmm.reshape(*grid_points.shape[:-1])
        else:
            true_log_gmm = None

    assert (
        cond_samples_heuristic.shape[0] == 1
    ), "cond_samples_heuristic should have shape (1, num_samples, num_dim_to_sample)"
    # remove first dimension
    cond_samples_heuristic = cond_samples_heuristic[0, :, :]

    cond_samples_heuristic = cond_samples_heuristic[:, ~cond_indices]
    _, rng_1 = jax.random.split(rng)
    w1_replacement = sliced_wasserstein(
        cond_samples_heuristic, exact_samples, rng_1, no_slicing=no_wasserstein_slicing
    )

    if save_dir is not None:
        save_dict = {
            "exact_samples": exact_samples,
            "means": means,
            "covs": covs,
            "weights": weights,
            "sigma_min": sigma_min,
            "cond_values": cond_values,
            "cond_indices": cond_indices,
            "energy": energy,
            "true_log_gmm": true_log_gmm,
            "grid_points": grid_points,
        }

        np.savez(
            f"{save_dir}/replacement_cond_samples_dim_{dim}_num_dim_to_sample_{num_dim_to_sample}_it_{k}.npz",
            cond_samples=cond_samples_heuristic,
            w1=w1_replacement,
            ar_samples=ar_samples,
            ar_w1=w1_replacement_ar,
            **save_dict,
        )

    # Gradient Guidance
    cond_samples_guidance, _, all_samples, _ = sample(
        config_pc,
        sde,
        score_model,
        state,
        sampling_shape,
        rng,
        sampling_eps,
        cond_indices=cond_indices,
        cond_values=cond_values,
        heuristic_cond_sampling=False,
        store_intermediate_samples=False,
        guidance_alpha=0.0,
    )
    cond_samples_guidance = cond_samples_guidance[0, :, :]
    cond_samples_guidance = cond_samples_guidance[:, ~cond_indices]
    rng, rng_1 = jax.random.split(rng)
    w1_guidance = sliced_wasserstein(
        cond_samples_guidance, exact_samples, rng, no_slicing=no_wasserstein_slicing
    )

    if save_dir is not None:
        np.savez(
            f"{save_dir}/guidance_cond_samples_dim_{dim}_num_dim_to_sample_{num_dim_to_sample}_it_{k}.npz",
            cond_samples=cond_samples_guidance,
            w1=w1_guidance,
            **save_dict,
        )

    # TDS
    assert config_tds.sampling.method == "tds"
    cond_samples_tds, _, all_samples, _ = sample(
        config_tds,
        sde,
        score_model,
        state,
        sampling_shape,
        rng,
        sampling_eps,
        cond_indices=cond_indices,
        cond_values=cond_values,
        heuristic_cond_sampling=False,
        store_intermediate_samples=False,
        guidance_alpha=0.0,
    )
    cond_samples_tds = cond_samples_tds[0, :, :]
    cond_samples_tds = cond_samples_tds[:, ~cond_indices]
    rng, rng_1 = jax.random.split(rng)
    w1_tds = sliced_wasserstein(
        cond_samples_tds, exact_samples, rng_1, no_slicing=no_wasserstein_slicing
    )

    if save_dir is not None:
        np.savez(
            f"{save_dir}/tds_cond_samples_dim_{dim}_num_dim_to_sample_{num_dim_to_sample}_it_{k}.npz",
            cond_samples=cond_samples_tds,
            w1=w1_tds,
            **save_dict,
        )

    return (
        w1_replacement,
        w1_guidance,
        w1_tds,
        cond_samples_heuristic,  # replacement
        cond_samples_guidance,
        cond_samples_tds,
    )


def disco_exps(
    state,
    config,
    score_model,
    sde,
    sampling_eps,
    cond_values,
    cond_indices,
    exact_samples,
    dim,
    num_dim_to_sample,
    k,
    rng=None,
    save_dir=None,
    save_disco_energy=False,
    means=None,
    covs=None,
    weights=None,
    sigma_min=None,
    ar_sample=False,
    exact_1d_sampling=False,
    no_wasserstein_slicing=False,
    num_integration_steps=2,
    num_mcmc_steps=10,
    step_size=5e-2,
    target_ess=0.75,
    box_min=None,
    box_max=None,
):
    num_samples = exact_samples.shape[0]
    rng = jax.random.PRNGKey(42) if rng is None else rng

    sampling_shape = (num_samples, config.data.num_channels)
    cond_samples, _, _, grad_norms, neg_energy_fn = sample(
        config,
        sde,
        score_model,
        state,
        sampling_shape,
        rng,
        sampling_eps,
        cond_indices=cond_indices,
        cond_values=cond_values,
        store_intermediate_samples=False,
        hmc=True,
        hmc_disco=True,
        num_integration_steps=num_integration_steps,
        num_mcmc_steps=num_mcmc_steps,
        step_size=step_size,
        target_ess=target_ess,
    )

    rng, rng_1 = jax.random.split(rng)
    if exact_samples is not None:
        w1_disco = sliced_wasserstein(
            cond_samples, exact_samples, rng_1, no_slicing=no_wasserstein_slicing
        )
    else:
        w1_disco = None

    ar_samples = None
    w1_disco_ar = None
    if ar_sample:
        ar_samples, acc_rate, logM = ar_sample_fn(
            rng,
            n_samples=num_samples,
            neg_energy_fn=neg_energy_fn,
            dim=dim,
            box_min=box_min,
            box_max=box_max,
        )
        rng, rng_1 = jax.random.split(rng)
        w1_disco_ar = sliced_wasserstein(
            ar_samples, exact_samples, rng_1, no_slicing=no_wasserstein_slicing
        )
        print(
            f"[DISCO AR] W1: {w1_disco_ar:.4f}, Acceptance rate: {acc_rate:.3f}, LogM: {logM:.3f}"
        )

    if num_dim_to_sample == 1 and exact_1d_sampling:
        # Perform exact 1D sampling along the free dimension and save a plot
        w1_exact_1d = exact_1d_sampling_fn(
            neg_energy_fn,
            rng,
            num_samples,
            exact_samples,
            means,
            covs,
            weights,
            cond_values,
            cond_indices,
            out_path=f"./plots/1d_density_disco.png",
            title="1D density and histogram",
            no_wasserstein_slicing=no_wasserstein_slicing,
        )
        print(f"[DISCO Exact 1D] W1: {w1_exact_1d:.4f}")

    if save_dir is not None:
        energy = None
        true_log_gmm = None
        grid_points = None

        if save_disco_energy:
            # save the energy of the DISCO model (evaluated on a grid of points)
            print(
                "Saving the energy of the DISCO model (evaluated on a grid of points)"
            )

            # Create a grid of points between box_min and box_max in each dimension
            grid_points = jnp.linspace(box_min, box_max, 100)
            grid_points = jnp.meshgrid(*[grid_points] * num_dim_to_sample)
            grid_points = jnp.stack(grid_points, axis=-1)

            # Evaluate the energy of the DISCO model on the grid of points
            energy = jax.vmap(neg_energy_fn)(
                grid_points.reshape(-1, grid_points.shape[-1])
            )
            energy = energy.reshape(*grid_points.shape[:-1])

            if means is not None and covs is not None and weights is not None:
                true_log_gmm = jax.vmap(
                    lambda x: eval_gmm_conditional(
                        x, means, covs, weights, cond_values, cond_indices
                    )
                )(grid_points.reshape(-1, grid_points.shape[-1]))
                true_log_gmm = true_log_gmm.reshape(*grid_points.shape[:-1])
            else:
                true_log_gmm = None

        np.savez(
            f"{save_dir}/disco_cond_samples_dim_{dim}_num_dim_to_sample_{num_dim_to_sample}_it_{k}.npz",
            cond_samples=cond_samples,
            exact_samples=exact_samples,
            w1=w1_disco,
            energy=energy,
            true_log_gmm=true_log_gmm,
            grid_points=grid_points,
            means=means,
            covs=covs,
            weights=weights,
            sigma_min=sigma_min,
            cond_values=cond_values,
            cond_indices=cond_indices,
            ar_samples=ar_samples,
            ar_w1=w1_disco_ar,
        )

    return w1_disco, cond_samples


def get_model_names(
    dataset, dim, minibatch_posterior=False, batch_size=None, checkpoints_folder=None
):
    """Get model names for the specified dimension."""

    disco_matches = [
        p
        for p in checkpoints_folder.iterdir()
        if p.is_dir() and dataset in p.name and "_disco_" in p.name
    ]

    nodisco_matches = [
        p
        for p in checkpoints_folder.iterdir()
        if p.is_dir() and dataset in p.name and "_nodisco_" in p.name
    ]

    if not disco_matches or not nodisco_matches:
        raise FileNotFoundError(
            f"No checkpoint folder found for dataset='{dataset}' in {checkpoints_folder}"
        )

    # overwrite disco_model_name with the most recent matching checkpoint
    disco_model_name = max(disco_matches, key=lambda p: p.stat().st_mtime).name
    nodisco_model_name = max(nodisco_matches, key=lambda p: p.stat().st_mtime).name

    return nodisco_model_name, disco_model_name


def slice_dataset(dataset, cond_values, cond_indices, epsilon=1e-2):
    # return subset of dataset where dataset[:, cond_indices] == cond_values, up to epsilon
    if jnp.all(~cond_indices):
        return dataset  # no slicing needed if we don't condition on anything

    return dataset[
        jnp.all(
            jnp.abs(dataset[:, cond_indices] - cond_values[cond_indices]) < epsilon,
            axis=1,
        ),
    ][:, ~cond_indices]


def main(
    dataset,
    num_samples,
    num_conditionals,
    num_runs_per_cond=5,
    dim=None,
    step=None,
    disco_fixed_sigma_is=None,
    rng=None,
    condition_noise_std=0.0,
    num_dim_to_sample=None,
    save_dir=None,
    save_disco_energy=False,
    ar_sample=False,
    exact_1d_sampling=False,
    no_wasserstein_slicing=False,
    num_integration_steps=2,
    num_mcmc_steps=10,
    step_size=5e-2,
    target_ess=0.75,
    box_min=None,
    box_max=None,
    inference_divergence=False,
    model_fit=False,
    minibatch_posterior=False,
    batch_size=None,
):
    rng = jax.random.PRNGKey(42) if rng is None else rng
    print(f"Condition noise std: {condition_noise_std}")

    nodisco_model_name, disco_model_name = get_model_names(
        dataset,
        dim,
        minibatch_posterior,
        batch_size=batch_size,
        checkpoints_folder=checkpoint_path,
    )

    # NO-DISCO
    nodisco_state, nodisco_config, nodisco_score_model, nodisco_model_dir = load_model(
        nodisco_model_name, checkpoint_path, step=step
    )

    nodisco_config.sampling.method = "pc"
    nodisco_config.sampling.predictor = "ancestral_sampling"
    nodisco_config.sampling.corrector = "none"
    nodisco_config.sampling.n_steps_each = 1  # same number of steps as DISCO
    nodisco_config.sampling.noise_removal = False  # We don't want noise removal here, this will hurt mixing -- we only use it for CIFAR

    # deepcopy the config
    nodisco_config_tds = copy.deepcopy(nodisco_config)
    nodisco_config_tds.sampling.method = "tds"

    nodisco_config.training.batch_size = num_samples
    # Get dataset
    if dataset == "gmm":
        train_ds, eval_ds, _, means, covs, weights = datasets.get_dataset(
            nodisco_config, return_gmm_params=True
        )
    else:
        train_ds, eval_ds, _ = datasets.get_dataset(nodisco_config)
        means, covs, weights = None, None, None

    # nodisco_config.model.num_scales = 100  # DEBUG
    nodisco_sde, nodisco_sampling_eps = setup_sde(nodisco_config)
    print("NO-DISCO SDE.N", nodisco_sde.N)

    # DISCO
    disco_state, disco_config, disco_score_model, disco_model_dir = load_model(
        disco_model_name, checkpoint_path, step=step
    )
    if disco_fixed_sigma_is is not None:
        disco_config.training.fixed_sigma_is = disco_fixed_sigma_is  # overwrite to slightly change the final temperature of the model

    disco_sde, disco_sampling_eps = setup_sde(disco_config)

    # if num_dim_to_sample is None, we sample only from a 1D distribution (the last coordinate)
    split_point = -1 if num_dim_to_sample is None else -num_dim_to_sample
    cond_indices = jnp.ones(dim, dtype=bool)
    cond_indices = cond_indices.at[split_point:].set(False)

    cond_values = jnp.zeros(dim) + jnp.nan

    assert (
        disco_config.model.sigma_min == nodisco_config.model.sigma_min
    ), f"{disco_config.model.sigma_min} != {nodisco_config.model.sigma_min}"
    sigma_min = disco_config.model.sigma_min

    true_data = next(iter(eval_ds))["image"]
    true_data = jnp.array(true_data).reshape(true_data.shape[1], -1)
    rng, key = jax.random.split(rng)
    # perturb the data with sigma_min noise, so true_data ~ p'(x)
    true_data = true_data + jax.random.normal(key, shape=true_data.shape) * sigma_min

    if inference_divergence or model_fit:
        rng_joint = jax.random.PRNGKey(0)
        num_repeat_joint_sampling = 1 if inference_divergence else num_conditionals
        sampling_shape = (num_samples, disco_config.data.num_channels)
        w1_disco_list = []
        w1_nodisco_list = []
        for j in range(num_repeat_joint_sampling):
            rng_joint, key = jax.random.split(rng_joint)
            joint_samples_disco, _, _, _, _ = sample(
                disco_config,
                disco_sde,
                disco_score_model,
                disco_state,
                sampling_shape,
                rng_joint,
                disco_sampling_eps,
                cond_indices=None,
                cond_values=None,
                store_intermediate_samples=False,
                hmc=True,
                hmc_disco=True,
                num_integration_steps=num_integration_steps,
                num_mcmc_steps=num_mcmc_steps,
                step_size=step_size,
                target_ess=target_ess,
            )

            joint_samples_nodisco, _, _, _ = sample(
                nodisco_config,
                nodisco_sde,
                nodisco_score_model,
                nodisco_state,
                sampling_shape,
                rng_joint,
                nodisco_sampling_eps,
                cond_indices=None,
                cond_values=None,
                heuristic_cond_sampling=False,
                store_intermediate_samples=False,
            )
            joint_samples_nodisco = joint_samples_nodisco[
                0
            ]  # squeeze out first dimension

            if model_fit:
                w1_disco = sliced_wasserstein(
                    joint_samples_disco,
                    true_data,
                    rng,
                    no_slicing=no_wasserstein_slicing,
                )
                w1_nodisco = sliced_wasserstein(
                    joint_samples_nodisco,
                    true_data,
                    rng,
                    no_slicing=no_wasserstein_slicing,
                )
                # print(f"W1 DISCO: {w1_disco}")
                # print(f"W1 No-DISCO: {w1_nodisco}")
                w1_disco_list.append(w1_disco)
                w1_nodisco_list.append(w1_nodisco)

        if model_fit:
            w1_disco_list = jnp.array(w1_disco_list)
            w1_nodisco_list = jnp.array(w1_nodisco_list)

            disco_mean = w1_disco_list.mean()
            disco_std = w1_disco_list.std()
            disco_min = w1_disco_list.min()
            disco_max = w1_disco_list.max()
            nodisco_mean = w1_nodisco_list.mean()
            nodisco_std = w1_nodisco_list.std()
            nodisco_min = w1_nodisco_list.min()
            nodisco_max = w1_nodisco_list.max()
            print(f"Model fit:")
            print(
                f"W1 DISCO: {disco_mean:.4f} ± {2*disco_std:.4f} (min: {disco_min:.4f}, max: {disco_max:.4f})"
            )
            print(
                f"W1 No-DISCO: {nodisco_mean:.4f} ± {2*nodisco_std:.4f} (min: {nodisco_min:.4f}, max: {nodisco_max:.4f})"
            )

            # Save inference-divergence W1 lists
            out_dir_tmp = save_dir if save_dir is not None else "."
            print(f"Saving model fit W1 lists to {out_dir_tmp}")
            np.savez(
                os.path.join(
                    out_dir_tmp,
                    (
                        "model_fit_w1_dataset_"
                        f"{dataset}_"
                        f"dim_{dim}_"
                        f"num_samples_{num_samples}_"
                        f"step_{step}_"
                        f"num_mcmc_steps_{num_mcmc_steps}_"
                        f"num_integration_steps_{num_integration_steps}_"
                        f"step_size_{step_size}_"
                        f"target_ess_{target_ess}.npz"
                    ),
                ),
                w1_disco_list=w1_disco_list,
                w1_nodisco_list=w1_nodisco_list,
            )
            return

        # print(f"Joint samples DISCO shape: {joint_samples_disco.shape}")
        # print(f"Joint samples No-DISCO shape: {joint_samples_nodisco.shape}")
    else:
        joint_samples_disco, joint_samples_nodisco = None, None

    w1_disco_list = []
    w1_replacement_list = []
    w1_guidance_list = []
    w1_tds_list = []
    w1_randn_baseline_list = []
    w1_true_vs_true_list = []
    for k in tqdm(range(0, num_conditionals)):
        values = true_data[k, cond_indices]
        rng, key = jax.random.split(rng)
        values = (
            values + jax.random.normal(key, shape=values.shape) * condition_noise_std
        )
        cond_values = cond_values.at[:split_point].set(values)
        print(f"Cond values: {cond_values}")

        if not inference_divergence:
            if dataset == "gmm":
                sample_batch = jax.vmap(
                    lambda key: sample_conditional_gmm(
                        key, means, covs, weights, cond_values, cond_indices
                    )
                )
            else:
                # we'll always return the same subset of the eval dataset
                def sample_batch(key_int):
                    # copy the config
                    cfg = copy.deepcopy(nodisco_config)
                    cfg.training.batch_size = (
                        num_samples if dim == 2 else 10 * num_samples
                    )  # sample enough data before slicing
                    ds, _, _ = datasets.get_dataset(cfg, rng_key=key_int)
                    data = next(iter(ds))["image"]
                    data = jnp.array(data).reshape(data.shape[1], -1)
                    data = slice_dataset(data, cond_values, cond_indices)
                    print(f"Exact cond samples shape: {data.shape}")
                    return data

        else:
            # inference divergence just slices out the joint samples with slice_dataset, we don't resample
            exact_cond_samples_disco = slice_dataset(
                joint_samples_disco, cond_values, cond_indices
            )
            print(f"Exact cond samples DISCO shape: {exact_cond_samples_disco.shape}")
            exact_cond_samples_nodisco = slice_dataset(
                joint_samples_nodisco, cond_values, cond_indices
            )
            print(
                f"Exact cond samples No-DISCO shape: {exact_cond_samples_nodisco.shape}"
            )
            # trim samples to have the same number of elements
            min_length = min(
                exact_cond_samples_disco.shape[0], exact_cond_samples_nodisco.shape[0]
            )
            exact_cond_samples_disco = exact_cond_samples_disco[:min_length]
            exact_cond_samples_nodisco = exact_cond_samples_nodisco[:min_length]
            print(f"Trimmed exact cond samples shape: {exact_cond_samples_disco.shape}")

            if min_length < 1000:
                print(
                    f"Warning: Less than 1000 exact cond samples, skipping this conditional"
                )
                continue

        # Exact Conditional Sampling from Ground Truth GMM
        for i in range(num_runs_per_cond):
            if not inference_divergence:
                rng, key = jax.random.split(rng)
                subkeys = jax.random.split(key, num_samples)
                exact_cond_samples = (
                    sample_batch(subkeys)
                    if dataset == "gmm"
                    else sample_batch(k * num_runs_per_cond + i)
                )
                # print(f"Exact cond samples shape: {exact_cond_samples.shape}")
                # add noise to the exact samples
                rng, key = jax.random.split(rng)
                exact_cond_samples = (
                    exact_cond_samples
                    + jax.random.normal(key, shape=exact_cond_samples.shape) * sigma_min
                )
                exact_cond_samples_disco = exact_cond_samples_nodisco = (
                    exact_cond_samples
                )

            # Run DISCO experiments
            w1_disco, _ = disco_exps(
                disco_state,
                disco_config,
                disco_score_model,
                disco_sde,
                disco_sampling_eps,
                cond_values,
                cond_indices,
                exact_cond_samples_disco,
                dim,
                num_dim_to_sample,
                k,
                rng,
                save_dir,
                save_disco_energy,
                means,
                covs,
                weights,
                sigma_min,
                ar_sample,
                exact_1d_sampling,
                no_wasserstein_slicing,
                num_integration_steps,
                num_mcmc_steps,
                step_size,
                target_ess,
                box_min,
                box_max,
            )

            # Run No-DISCO experiments
            w1_replacement, w1_guidance, w1_tds, _, _, _ = nodisco_exps(
                nodisco_state,
                nodisco_config,
                nodisco_config_tds,
                nodisco_score_model,
                nodisco_sde,
                nodisco_sampling_eps,
                cond_values,
                cond_indices,
                exact_cond_samples_nodisco,
                dim,
                num_dim_to_sample,
                k,
                rng,
                save_dir,
                means,
                covs,
                weights,
                sigma_min,
                save_disco_energy,
                ar_sample,
                exact_1d_sampling,
                no_wasserstein_slicing,
                box_min,
                box_max,
            )

            w1_disco_list.append(w1_disco)
            w1_replacement_list.append(w1_replacement)
            w1_guidance_list.append(w1_guidance)
            w1_tds_list.append(w1_tds)
            rng, rng_1, rng_2, rng_3 = jax.random.split(rng, 4)
            if not inference_divergence:
                w1_randn_baseline_list.append(
                    sliced_wasserstein(
                        jax.random.normal(rng_1, shape=exact_cond_samples.shape),
                        exact_cond_samples,
                        rng_2,
                        no_slicing=no_wasserstein_slicing,
                    )
                )
                w1_true_vs_true_list.append(
                    sliced_wasserstein(
                        exact_cond_samples,
                        exact_cond_samples,
                        rng_3,
                        no_slicing=no_wasserstein_slicing,
                    )
                )

        # Calculate means and standard deviations (no min/max for first print)
        replacement_arr = jnp.array(w1_replacement_list[-num_runs_per_cond:])
        guidance_arr = jnp.array(w1_guidance_list[-num_runs_per_cond:])
        tds_arr = jnp.array(w1_tds_list[-num_runs_per_cond:])
        disco_arr = jnp.array(w1_disco_list[-num_runs_per_cond:])
        randn_baseline_arr = jnp.array(w1_randn_baseline_list[-num_runs_per_cond:])

        replacement_mean = jnp.mean(replacement_arr)
        replacement_std = jnp.std(replacement_arr)

        guidance_mean = jnp.mean(guidance_arr)
        guidance_std = jnp.std(guidance_arr)

        tds_mean = jnp.mean(tds_arr)
        tds_std = jnp.std(tds_arr)

        disco_mean = jnp.mean(disco_arr)
        disco_std = jnp.std(disco_arr)

        if not inference_divergence:
            randn_baseline_mean = jnp.mean(randn_baseline_arr)
            randn_baseline_std = jnp.std(randn_baseline_arr)
            randn_str = f"W1 Randn Baseline: {randn_baseline_mean:.3f} ± {2*randn_baseline_std:.3f}, \n"
        else:
            randn_str = ""

        print(
            f"[{k}] "
            f"W1 Replacement: {replacement_mean:.3f} ± {2*replacement_std:.3f}, \n"
            f"W1 Guidance: {guidance_mean:.3f} ± {2*guidance_std:.3f}, \n"
            f"W1 TDS: {tds_mean:.3f} ± {2*tds_std:.3f}, \n"
            f"W1 DISCO: {disco_mean:.3f} ± {2*disco_std:.3f} \n"
            f"{randn_str}"
        )
        # flush the print buffer
        sys.stdout.flush()

    w1_disco_list = jnp.array(w1_disco_list)
    w1_replacement_list = jnp.array(w1_replacement_list)
    w1_guidance_list = jnp.array(w1_guidance_list)
    w1_tds_list = jnp.array(w1_tds_list)
    w1_randn_baseline_list = jnp.array(w1_randn_baseline_list)
    w1_true_vs_true_list = jnp.array(w1_true_vs_true_list)

    # save them to a file
    out_dir = save_dir if save_dir is not None else "."
    np.savez(
        os.path.join(
            out_dir,
            f"w1_disco_list_dataset_{dataset}_dim_{dim}_num_conditionals_{num_conditionals}_condition_noise_std_{condition_noise_std}_"
            f"step_{step}_num_samples_{num_samples}_num_runs_per_cond_{num_runs_per_cond}_"
            f"disco_fixed_sigma_is_{disco_fixed_sigma_is}.npz",
        ),
        w1_disco_list=w1_disco_list,
        w1_replacement_list=w1_replacement_list,
        w1_guidance_list=w1_guidance_list,
        w1_tds_list=w1_tds_list,
        w1_randn_baseline_list=w1_randn_baseline_list,
        w1_true_vs_true_list=w1_true_vs_true_list,
        disco_model_name=disco_model_name,
        nodisco_model_name=nodisco_model_name,
        disco_step=step,
        nodisco_step=step,
    )

    # Compute min/max for the final print
    replacement_min = w1_replacement_list.min()
    replacement_max = w1_replacement_list.max()
    guidance_min = w1_guidance_list.min()
    guidance_max = w1_guidance_list.max()
    tds_min = w1_tds_list.min()
    tds_max = w1_tds_list.max()
    disco_min = w1_disco_list.min()
    disco_max = w1_disco_list.max()
    if not inference_divergence:
        randn_baseline_min = w1_randn_baseline_list.min()
        randn_baseline_max = w1_randn_baseline_list.max()
        true_vs_true_min = w1_true_vs_true_list.min()
        true_vs_true_max = w1_true_vs_true_list.max()
    else:
        randn_baseline_min = randn_baseline_max = true_vs_true_min = (
            true_vs_true_max
        ) = 0

    print("-" * 30)
    print("Inference Quality:")
    print(
        f"W1 Replacement: {w1_replacement_list.mean():.3f} ± {2*w1_replacement_list.std():.3f} (min: {replacement_min:.3f}, max: {replacement_max:.3f}), \n"
        f"W1 Guidance: {w1_guidance_list.mean():.3f} ± {2*w1_guidance_list.std():.3f} (min: {guidance_min:.3f}, max: {guidance_max:.3f}), \n"
        f"W1 TDS: {w1_tds_list.mean():.3f} ± {2*w1_tds_list.std():.3f} (min: {tds_min:.3f}, max: {tds_max:.3f}), \n"
        f"W1 DISCO: {w1_disco_list.mean():.3f} ± {2*w1_disco_list.std():.3f} (min: {disco_min:.3f}, max: {disco_max:.3f}), \n"
        # f"W1 Randn Baseline: {w1_randn_baseline_list.mean():.3f} ± {2*w1_randn_baseline_list.std():.3f} (min: {randn_baseline_min:.3f}, max: {randn_baseline_max:.3f}), \n"
        # f"W1 True vs True: {w1_true_vs_true_list.mean():.3f} ± {2*w1_true_vs_true_list.std():.3f} (min: {true_vs_true_min:.3f}, max: {true_vs_true_max:.3f})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run conditional sampling experiments")
    parser.add_argument(
        "--num-conditionals",
        type=int,
        default=1,
        help="Number of conditional samples to generate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of samples per conditional",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gmm",
        help="Name of the dataset to use. Options: gmm, moons, checkerboard, rings",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Step number to load the model from"
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimension of the GMM")
    parser.add_argument(
        "--disco-fixed-sigma-is",
        type=float,
        default=None,
        help="Fixed sigma is for DISCO",
    )
    parser.add_argument(
        "--num-runs-per-cond",
        type=int,
        default=1,
        help="Number of runs per conditional",
    )
    parser.add_argument(
        "--condition-noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of the condition noise",
    )
    parser.add_argument(
        "--num-dim-to-sample",
        type=int,
        default=None,
        help="Number of dimensions to sample from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save all .npz outputs. Will be created if missing.",
    )
    parser.add_argument(
        "--save-disco-energy",
        type=bool,
        default=False,
        help="Save the energy of the DISCO model (evaluated on a grid of points)",
    )
    parser.add_argument(
        "--ar-sample",
        type=bool,
        default=False,
        help="Use acceptance-rejection sampling instead of SMC sampling",
    )
    parser.add_argument(
        "--exact-1d-sampling",
        type=bool,
        default=False,
        help="Use exact 1D sampling instead of SMC/ancestral sampling",
    )
    parser.add_argument(
        "--no-wasserstein-slicing",
        type=bool,
        default=False,
        help="Compute 2D Wasserstein distance instead of sliced Wasserstein",
    )
    parser.add_argument(
        "--num-integration-steps",
        type=int,
        default=2,
        help="Number of integration steps for SMC sampling",
    )
    parser.add_argument(
        "--num-mcmc-steps",
        type=int,
        default=10,
        help="Number of MCMC steps for SMC sampling",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=5e-2,
        help="Step size for SMC sampling",
    )
    parser.add_argument(
        "--target-ess",
        type=float,
        default=0.75,
        help="Target ESS for SMC sampling",
    )
    parser.add_argument(
        "--box-min",
        type=float,
        default=None,
        help="Minimum value for the box",
    )
    parser.add_argument(
        "--box-max",
        type=float,
        default=None,
        help="Maximum value for the box",
    )
    parser.add_argument(
        "--inference-divergence",
        type=bool,
        default=False,
        help="Measure divergence between true model conditional and (heuristic) conditional",
    )
    parser.add_argument(
        "--model-fit",
        type=bool,
        default=False,
        help="Measure model fit between true model and model (joint) samples",
    )
    parser.add_argument(
        "--minibatch-posterior",
        type=bool,
        default=False,
        help="Whether the DISCO model was trained with the minibatch posterior objective, which changes the model and thus the results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used for minibatch posterior ablation. Only used if --minibatch-posterior is True",
    )

    args = parser.parse_args()
    print(f"Save DISCO Energy: {args.save_disco_energy}")
    # Ensure output directory exists if provided
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    main(
        dataset=args.dataset,
        num_conditionals=args.num_conditionals,
        num_samples=args.num_samples,
        dim=args.dim,
        step=args.step,
        disco_fixed_sigma_is=args.disco_fixed_sigma_is,
        num_runs_per_cond=args.num_runs_per_cond,
        condition_noise_std=args.condition_noise_std,
        num_dim_to_sample=args.num_dim_to_sample,
        save_dir=args.save_dir,
        save_disco_energy=args.save_disco_energy,
        ar_sample=args.ar_sample,
        exact_1d_sampling=args.exact_1d_sampling,
        no_wasserstein_slicing=args.no_wasserstein_slicing,
        num_integration_steps=args.num_integration_steps,
        num_mcmc_steps=args.num_mcmc_steps,
        step_size=args.step_size,
        target_ess=args.target_ess,
        box_min=args.box_min,
        box_max=args.box_max,
        inference_divergence=args.inference_divergence,
        model_fit=args.model_fit,
        minibatch_posterior=args.minibatch_posterior,
        batch_size=args.batch_size,
    )
