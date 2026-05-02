import sys
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import flax.jax_utils as flax_utils
from tqdm import tqdm
from flax.training import orbax_utils, checkpoints

from models import ddpm, ncsnv2, ncsnpp  # Don't remove, this registers the models
from models import utils as mutils
import losses
import datasets
import sde_lib
from gmm_utils_sampling import conditional_gmm, eval_gmm, eval_cond_gmm
import yaml
from ml_collections import ConfigDict
from gmm_utils import compute_wasserstein_distance
import blackjax

# Configure JAX and TF
jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


def get_latest_model(
    checkpoint_path: Path,
    use_disco: bool = True,
    loss_variant: str = "loss_variant=None",
    use_baseline: bool = False,
) -> str:
    """Get the name of the latest model based on configuration."""
    if use_disco:
        model_paths = (checkpoint_path).glob(f"*{loss_variant}*")
    else:
        if use_baseline:
            model_paths = (checkpoint_path).glob("*fixed_sigma_baseline=0*")
        else:
            model_paths = (checkpoint_path).glob("*loss_variant=None*")

    return max(model_paths, key=os.path.getctime).name


def load_model(
    model_name: str,
    checkpoint_path: Path,
    use_disco: bool = True,
    use_baseline: bool = False,
    step: Optional[int] = None,
    use_old_state=False,
    config=None,
    return_just_config=False,
) -> Tuple[Any, Any, Any, Any]:
    """Load model and configuration based on model name."""
    model_dir = checkpoint_path / model_name
    model_dir = model_dir.absolute()  # Ensure absolute path

    if config is None:
        # read yml file in model_dir
        with open(model_dir / "config.yml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = ConfigDict(config)

    if return_just_config:
        return config

    rng = jax.random.PRNGKey(config.seed)
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
    optimizer = losses.get_optimizer(config)
    opt_state = optimizer.init(initial_params)
    if use_old_state:
        state = mutils.OldState(
            step=0,
            optimizer=None,
            lr=None,
            model_state=init_model_state,
            ema_rate=config.model.ema_rate,
            params_ema=initial_params,
            rng=rng,
        )
    else:
        state = mutils.State(
            step=0,
            opt_state=opt_state,
            params=initial_params,
            model_state=init_model_state,
            ema_rate=config.model.ema_rate,
            params_ema=initial_params,
            rng=rng,
        )

    if config.model.name not in ["gmm_score_wrapper", "gmm_score_wrapper_ebm"]:
        # Restore checkpoint
        print(f"Loading checkpoint from: {model_dir}")
        state = checkpoints.restore_checkpoint(model_dir, state, step=step)
        initial_step = int(state.step)
        print(f"Loaded model at step {initial_step}")

    return state, config, score_model, model_dir


def setup_sde(config: Any) -> Tuple[Any, float]:
    """Setup SDE based on configuration."""
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    return sde, sampling_eps


def get_score_function(sde: Any, score_model: Any, state: Any, config: Any) -> Callable:
    """Get score function from model and configuration."""
    fixed_sigma_is = (
        config.training.fixed_sigma_is
        if hasattr(config.training, "fixed_sigma_is")
        else None
    )
    fixed_sigma_baseline = (
        config.training.fixed_sigma_baseline
        if hasattr(config.training, "fixed_sigma_baseline")
        else None
    )

    return mutils.get_score_fn(
        sde,
        score_model,
        state.params_ema,
        state.model_state,
        train=False,
        continuous=config.training.continuous,
        fixed_sigma=fixed_sigma_is,
        fixed_sigma_baseline=fixed_sigma_baseline,
    )


def sample(
    config: Any,
    sde: Any,
    score_model: Any,
    state: Any,
    shape: Tuple[int, int],
    rng: Any,
    sampling_eps: float,
    cond_indices=None,
    cond_values=None,
    heuristic_cond_sampling=False,
    store_intermediate_samples=False,
    hmc=False,
    num_integration_steps=10,
    num_mcmc_steps=1,
    step_size=1e-2,
    target_ess=0.75,
    hmc_disco=True,
    guidance_alpha=0.0,
) -> Tuple[Any, Any, Any]:
    """Generate samples using the model."""
    from sampling import get_sampling_fn

    inverse_scaler = datasets.get_data_inverse_scaler(config)

    if hmc:
        # print("Using Tempered SMC sampler")
        sampling_fn, neg_energy_fn = get_smc_sampler(
            config,
            sde,
            score_model,
            state,
            shape,
            rng,
            cond_indices,
            cond_values,
            num_integration_steps,
            num_mcmc_steps,
            step_size,
            disco=hmc_disco,
            target_ess=target_ess,
        )
    else:
        sampling_fn = get_sampling_fn(
            config,
            sde,
            score_model,
            shape,
            inverse_scaler,
            sampling_eps,
            cond_indices=cond_indices,
            cond_values=cond_values,
            heuristic_cond_sampling=heuristic_cond_sampling,
            store_intermediate_samples=store_intermediate_samples,
            guidance_alpha=guidance_alpha,
        )

        neg_energy_fn = None

        # neg_energy_fn = get_smc_sampler(config, sde, score_model,
        #                               state, shape, rng, cond_indices, cond_values,
        #                               num_integration_steps, num_mcmc_steps,
        #                               step_size, disco=hmc_disco, get_just_energy=True)

    pstate = flax_utils.replicate(state)
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    sample, n, images, grad_norms = sampling_fn(sample_rng if not hmc else rng, pstate)

    if hmc:
        return (
            sample,
            n,
            images[0] if images is not None else None,
            grad_norms,
            neg_energy_fn,
        )
    else:
        # jax.debug.print('sample: {sample}', sample=sample)
        return sample, n, images[0] if images is not None else None, grad_norms


def get_hmc_sampler(
    config: Any,
    sde: Any,
    score_model: Any,
    state: Any,
    shape: Tuple[int, int],
    rng: Any,
    cond_indices=None,
    cond_values=None,
    num_warmup_steps=1000,
    num_steps=1000,
) -> Tuple[Any, Any, Any]:
    """Generate conditional samples using the model."""
    import blackjax
    from models.utils import get_model_fn

    if cond_indices is not None:
        if cond_indices.shape[0] == 1:
            cond_indices = cond_indices[0]
        if cond_values.shape[0] == 1:
            cond_values = cond_values[0]

        assert (
            cond_indices.shape[0] == shape[1] and len(cond_indices.shape) == 1
        ), f"we only support single-dimensional conditioning for now, got cond_indices.shape={cond_indices.shape} and shape={shape}"
        shape = (shape[0], jnp.sum(cond_indices))
        x_cond = jnp.where(cond_indices, cond_values, jnp.zeros_like(cond_values))
        # Get the indices where we need to place x_subset values
        subset_indices = jnp.where(~cond_indices)[
            0
        ]  # Get the indices where cond_indices is False

    model_fn = get_model_fn(
        score_model,
        state.params_ema,
        state.model_state,
        train=False,
        return_only_neg_energy=True,
    )
    init_gen = (
        lambda key: jax.random.normal(key, shape=(1, *shape[1:]))
        * config.model.sigma_max
    )

    if cond_values is None:
        # unconditional sampling
        neg_energy_fn = (
            lambda x: model_fn(x, labels=None, rng=None)[0].squeeze()
            / config.training.fixed_sigma_is**2
        )
    else:
        # conditional sampling
        def neg_energy_fn(x_subset):
            # Place x_subset values at the correct positions
            x = x_cond.at[subset_indices].set(x_subset[0])[None, ...]
            # jax.debug.print('x: {x}, x_subset: {x_subset}', x=x, x_subset=x_subset)

            return (
                model_fn(x, labels=None, rng=None)[0].squeeze()
                / config.training.fixed_sigma_is**2
            )

    rng, rng_key = jax.random.split(rng)
    # warmup_init = init_gen(rng_key)
    # warmup = blackjax.window_adaptation(blackjax.nuts, neg_energy_fn)
    # (state, params), _ = warmup.run(rng_key, warmup_init, num_steps=num_warmup_steps)
    # nuts = blackjax.nuts(neg_energy_fn, **params)
    # print(params)
    nuts = blackjax.nuts(
        neg_energy_fn, step_size=1.0, inverse_mass_matrix=jnp.ones(shape[1])
    )
    num_samples = shape[0]

    def hmc_sampler(rng_key, pstate):
        # Initialize N chains with random starting points
        rng_keys = jax.random.split(rng_key, num_samples)
        initial_positions = jax.vmap(init_gen)(rng_keys)
        states = jax.vmap(nuts.init)(initial_positions)

        # Iterate all chains in parallel
        step = jax.jit(jax.vmap(nuts.step))
        for i in range(num_steps):
            nuts_keys = jax.vmap(lambda key: jax.random.fold_in(key, i))(rng_keys)
            states, _ = step(nuts_keys, states)

        samples = states.position.reshape(shape)
        return samples, num_samples, None, None

    return hmc_sampler, neg_energy_fn


def get_smc_sampler(
    config: Any,
    sde: Any,
    score_model: Any,
    state: Any,
    shape: Tuple[int, int],
    rng: Any,
    cond_indices=None,
    cond_values=None,
    num_integration_steps=10,
    num_mcmc_steps=1,
    step_size=1e-2,
    disco=True,
    get_just_energy=False,
    target_ess=0.75,
) -> Tuple[Any, Any, Any]:
    """Generate conditional samples using the model."""
    import blackjax
    from models.utils import get_model_fn
    import blackjax.smc.resampling as resampling
    from blackjax.smc import extend_params

    if cond_indices is not None:
        if cond_indices.shape[0] == 1:
            cond_indices = cond_indices[0]
        if cond_values.shape[0] == 1:
            cond_values = cond_values[0]

        assert (
            cond_indices.shape[0] == shape[1] and len(cond_indices.shape) == 1
        ), f"we only support single-dimensional conditioning for now, got cond_indices.shape={cond_indices.shape} and shape={shape}"
        shape = (shape[0], jnp.sum(~cond_indices))
        x_cond = jnp.where(cond_indices, cond_values, jnp.zeros_like(cond_values))
        # Get the indices where we need to place x_subset values
        subset_indices = jnp.where(~cond_indices)[
            0
        ]  # Get the indices where cond_indices is False

    model_fn = get_model_fn(
        score_model,
        state.params_ema,
        state.model_state,
        train=False,
        return_only_neg_energy=True,
    )
    init_gen = (
        lambda key: jax.random.normal(key, shape=(1, *shape[1:]))
        * config.model.sigma_max
    )

    if disco:
        # DISCO (EBM-DISCO)
        if cond_values is None:
            # unconditional sampling
            neg_energy_fn = (
                lambda x: (model_fn(x[None, ...], labels=None, rng=None)[0].squeeze())
                / config.training.fixed_sigma_is**2
            )
            if config.model.name == "gmm_score_wrapper_ebm":
                # no scaling by sigma_is^2, we already have logp here
                neg_energy_fn = lambda x: model_fn(x[None, ...], labels=None, rng=None)[
                    0
                ].squeeze()  # / config.training.fixed_sigma_is**2
        else:
            # conditional sampling
            def neg_energy_fn(x_subset):
                # Place x_subset values at the correct positions
                x = x_cond.at[subset_indices].set(x_subset)[None, ...]
                # jax.debug.print('x: {x}, x_subset: {x_subset}', x=x, x_subset=x_subset)

                return (
                    model_fn(x, labels=None, rng=None)[0].squeeze()
                    / config.training.fixed_sigma_is**2
                )

    else:
        smld_sigma_array = sde.discrete_sigmas[::-1]
        t0_label = jnp.array(len(smld_sigma_array) - 1)  # label for smallest sigma
        # t0_label = jnp.array(99) # test
        t0_sigma = smld_sigma_array[t0_label]
        # print(f't0_sigma: {t0_sigma}')

        # No-DISCO (EBM-DM)
        if cond_values is None:
            # unconditional sampling
            neg_energy_fn = (
                lambda x: model_fn(x[None, ...], labels=t0_label[None, ...], rng=None)[
                    0
                ].squeeze()
                / t0_sigma
            )
        else:
            # conditional sampling
            def neg_energy_fn(x_subset):
                # Place x_subset values at the correct positions
                x = x_cond.at[subset_indices].set(x_subset[0])[None, ...]
                # jax.debug.print('x: {x}, x_subset: {x_subset}', x=x, x_subset=x_subset)

                return (
                    model_fn(x, labels=t0_label[None, ...], rng=None)[0].squeeze()
                    / t0_sigma
                )

        if get_just_energy:
            return neg_energy_fn

    num_samples = shape[0]
    # rng, rng_key = jax.random.split(rng)
    hmc_parameters = dict(
        step_size=step_size,
        inverse_mass_matrix=jnp.eye(shape[1]),
        num_integration_steps=num_integration_steps,
    )

    tempered = blackjax.adaptive_tempered_smc(
        lambda x: 0,  # prior is (unnormalized) uniform
        neg_energy_fn,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        extend_params(hmc_parameters),
        resampling.systematic,  # lambda rng_key, w, n: jnp.arange(n)
        target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    # tempered = blackjax.tempered_smc(
    #     lambda x: 0,  # prior is (unnormalized) uniform
    #     neg_energy_fn,
    #     blackjax.hmc.build_kernel(),
    #     blackjax.hmc.init,
    #     extend_params(hmc_parameters),
    #     resampling.systematic,  # lambda rng_key, w, n: jnp.arange(n)
    #     num_mcmc_steps=num_mcmc_steps,
    # )

    def smc_sampler(rng_key, pstate):
        rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
        # init points are sampled from N(0, sigma_max^2)
        # initial_smc_state = jax.random.multivariate_normal(
        #     init_key,
        #     jnp.zeros([shape[1]]),
        #     jnp.eye(shape[1]) * config.model.sigma_max**2,
        #     (num_samples,),
        # )

        # Set initial_smc_state to be uniform in the hypercube around 0, with length 4*sigma_max
        sigma_max = config.model.sigma_max
        # low = -2.5  # * sigma_max
        # high = 2.5  # * sigma_max
        low = -2.5  # * sigma_max
        high = 2.5  # * sigma_max
        initial_smc_state = jax.random.uniform(
            init_key, shape=(num_samples, shape[1]), minval=low, maxval=high
        )
        initial_smc_state = tempered.init(initial_smc_state)
        n_iter, smc_samples = smc_inference_loop(
            sample_key, tempered.step, initial_smc_state
        )
        return smc_samples.particles, num_samples, None, None

    return smc_sampler, neg_energy_fn


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1  # 0.05

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        # increase lambda
        # r = 1.1
        # num_steps = 100
        # alpha = (r - 1.0) / (r**num_steps - 1.0)
        # state, _ = smc_kernel(subk, state, state.lmbda * r + alpha)
        # state, _ = smc_kernel(subk, state, state.lmbda + 0.01)

        # compute ess
        ess = blackjax.smc.ess.ess(jnp.log(state.weights))

        # jax.debug.print(
        #     "i: {i}, lmbda: {lmbda}, ess: {ess}",
        #     i=i,
        #     lmbda=state.lmbda,
        #     ess=ess,
        # )
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )
    # jax.debug.print("n_iter: {n_iter}", n_iter=n_iter)

    return n_iter, final_state


def conditional_sample(
    config: Any,
    sde: Any,
    score_model: Any,
    state: Any,
    shape: Tuple[int, int],
    rng: Any,
    cond_x1: float,
    train_scaler: Any,
    sampling_eps: float,
) -> Tuple[Any, Any, Any]:
    """Generate conditional samples using the model."""
    from sampling import get_sampling_fn

    cond_indices = jnp.array([[True, False]])  # condition on the first RV
    cond_values = np.array([[cond_x1, np.nan]])  # condition the first RV to cond_x1
    cond_values = jnp.array(train_scaler.transform(cond_values))

    cond_indices = jnp.repeat(cond_indices, shape[0], axis=0)
    cond_values = jnp.repeat(cond_values, shape[0], axis=0)

    inverse_scaler = datasets.get_data_inverse_scaler(config)

    sampling_fn = get_sampling_fn(
        config,
        sde,
        score_model,
        shape,
        inverse_scaler,
        sampling_eps,
        cond_indices=cond_indices,
        cond_values=cond_values,
        heuristic_cond_sampling=False,
        store_intermediate_samples=True,
    )

    pstate = flax_utils.replicate(state)
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    sample, n, images = sampling_fn(sample_rng, pstate)

    return sample, n, images[0]


def plot_2d_samples(
    samples: np.ndarray,
    true_data: np.ndarray = None,
    title: str = "2D Sample Distribution",
) -> None:
    """Plot 2D samples with optional true data overlay."""
    plt.figure(figsize=(10, 10))
    if true_data is not None:
        plt.scatter(
            true_data[:, 0], true_data[:, 1], alpha=0.6, color="blue", label="True Data"
        )
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.4, color="red", label="Generated")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# def conditional_sampling_experiment(config, sde, score_model, state, sampling_eps, model_dir: Path,
#                                    means, covs, weights,
#                                    sampling_shape=(1024*4, 2), heuristic_cond_sampling=False,
#                                    show_sliced_joint_samples=False, joint_sample_shape=(1024*20, 2),
#                                    x1_values=jnp.linspace(-4, 4, 9), sliced_joint_eps=0.1,
#                                    hmc=False, num_integration_steps=10, num_mcmc_steps=1, step_size=1e-2,
#                                    hmc_disco=True, guidance_alpha=0.0, compute_w1_dist=False, noplot=False,
#                                    joint_samples=None, store_intermediate_samples=False) -> None:


def plot_conditional_distributions(
    config,
    sde,
    score_model,
    state,
    sampling_eps,
    model_dir: Path,
    means,
    covs,
    weights,
    sampling_shape=(1024 * 4, 2),
    heuristic_cond_sampling=False,
    show_sliced_joint_samples=False,
    joint_sample_shape=(1024 * 20, 2),
    x1_values=jnp.linspace(-4, 4, 9),
    sliced_joint_eps=0.1,
    hmc=False,
    num_integration_steps=10,
    num_mcmc_steps=1,
    step_size=1e-2,
    hmc_disco=True,
    guidance_alpha=0.0,
    compute_w1_dist=False,
    noplot=False,
    joint_samples=None,
    store_intermediate_samples=False,
) -> None:
    """Plot conditional distributions for different x1 values."""

    if config.sampling.method == "tds":
        assert not hmc, "TDS is not compatible with HMC/DISCO"

    rng = jax.random.PRNGKey(110)
    if show_sliced_joint_samples:
        rng, rng_key = jax.random.split(rng)
        if joint_samples is None:  # if joint_samples is not provided, sample it
            if hmc:
                joint_samples, _, _, _, _ = sample(
                    config,
                    sde,
                    score_model,
                    state,
                    joint_sample_shape,
                    rng_key,
                    sampling_eps,
                    store_intermediate_samples=True,
                    hmc=True,
                    num_integration_steps=num_integration_steps,
                    num_mcmc_steps=num_mcmc_steps,
                    step_size=step_size,
                    hmc_disco=hmc_disco,
                )
            else:
                joint_samples, _, _, _ = sample(
                    config,
                    sde,
                    score_model,
                    state,
                    joint_sample_shape,
                    rng_key,
                    sampling_eps,
                    store_intermediate_samples=True,
                    hmc=False,
                )

            if joint_samples.shape[0] == 1:
                joint_samples = joint_samples[0]  # remove batch dimension

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    w1_dists = []
    for i, cond_x1 in enumerate(tqdm(x1_values)):
        cond_indices = jnp.array([[True, False]])  # condition on the first RV
        cond_values = jnp.array(
            [[cond_x1, np.nan]]
        )  # condition the first RV to cond_x1, the rest is arbitrary

        if not hmc and heuristic_cond_sampling:
            cond_indices = jnp.repeat(cond_indices, sampling_shape[0], axis=0)
            cond_values = jnp.repeat(cond_values, sampling_shape[0], axis=0)
        elif guidance_alpha >= 0 or config.sampling.method == "tds":
            cond_indices = cond_indices[0]
            cond_values = cond_values[0]

        if hmc:
            samples, _, all_samples, _, neg_energy_fn = sample(
                config,
                sde,
                score_model,
                state,
                sampling_shape,
                rng,
                sampling_eps,
                cond_indices=cond_indices,
                cond_values=cond_values,
                heuristic_cond_sampling=heuristic_cond_sampling,
                store_intermediate_samples=store_intermediate_samples,
                hmc=hmc,
                num_integration_steps=num_integration_steps,
                num_mcmc_steps=num_mcmc_steps,
                step_size=step_size,
                hmc_disco=hmc_disco,
            )
        else:
            samples, _, all_samples, _ = sample(
                config,
                sde,
                score_model,
                state,
                sampling_shape,
                rng,
                sampling_eps,
                cond_indices=cond_indices,
                cond_values=cond_values,
                heuristic_cond_sampling=heuristic_cond_sampling,
                store_intermediate_samples=store_intermediate_samples,
                guidance_alpha=guidance_alpha,
            )

        if samples.shape[0] == 1:
            samples = samples[0]  # remove batch dimension

        smin = config.model.sigma_min * jnp.ones(2)

        log_cond_gmm = lambda x: eval_cond_gmm(
            x, means, covs, weights, cond_x1=cond_x1, perturb_sigma=smin
        )  # we perturb the GMM with smin (=fixed_sigma in DISCO)
        cond_gmm = lambda x: jnp.exp(log_cond_gmm(x))
        xx = jnp.linspace(-4, 4, 500)
        cond_gmm_vals = jax.vmap(cond_gmm)(xx)

        x1 = cond_values[0, 0] if cond_values.shape[0] == 1 else cond_values[0]
        if not noplot:
            ax = axes[i]
            ax.plot(xx, cond_gmm_vals, label="Conditional GMM")
            ax.hist(samples[:, 1], bins=80, alpha=0.6, label="Generated", density=True)

        if show_sliced_joint_samples or joint_samples is not None:
            # slice out samples where samples[:, 0] \approx cond_x1
            samples_cond = joint_samples[
                jnp.abs(joint_samples[:, 0] - cond_x1) < sliced_joint_eps
            ]
            if not noplot:
                ax.hist(
                    samples_cond[:, 1],
                    bins=80,
                    alpha=0.3,
                    label="Sliced Joint Samples",
                    density=True,
                )

        if hmc and not noplot:
            neg_energy_vals = jax.vmap(lambda x: neg_energy_fn(x[None, ...]))(xx)
            unnormalized_p_vals = jnp.exp(neg_energy_vals - jnp.max(neg_energy_vals))
            Z = jnp.trapezoid(unnormalized_p_vals, xx)
            p_vals = unnormalized_p_vals / Z
            ax.plot(xx, p_vals, label="EBM")

        if not noplot:
            ax.legend()

        if compute_w1_dist and (show_sliced_joint_samples or joint_samples is not None):
            n = min(samples.shape[0], samples_cond.shape[0])
            w1_dist = compute_wasserstein_distance(
                samples[:n, 1:], samples_cond[:n, 1:]
            )
            w1_dists.append(w1_dist)
            if not noplot:
                print(x1)
                print(w1_dist)
                # if x1 is not a scalar, take the first element
                x1 = x1[0] if not x1.shape == () else x1
                ax.set_title(f"p(x2 | x1), W1 Distance: {w1_dist:.3f}, n={n}")
                # ax.set_title(f'p(x2 | x1={x1:.3f}), W1 Distance: {w1_dist:.3f}, n={n}')
            else:
                print(x1)
                print(f"p(x2 | x1), W1 Distance: {w1_dist:.3f}, n={n}")
        else:
            if not noplot:
                ax.set_title(f"p(x2 | x1={x1:.3f})")

        samples.block_until_ready()

    if not noplot:
        plt.suptitle(
            f"Directly sampling from conditionals | N={sde.N}, n_steps_each={config.sampling.n_steps_each}, snr={config.sampling.snr}"
        )
        plt.tight_layout()
        plt.savefig(model_dir / "gmm_sampling_conditional.png")
        plt.show()

    return (
        joint_samples if show_sliced_joint_samples else None,
        samples,
        jax.vmap(log_cond_gmm),
        all_samples if store_intermediate_samples else None,
        w1_dists,
    )


def animate_2d_samples(
    samples: np.ndarray,
    sde: Any,
    score_model: Any,
    state: Any,
    config: Any,
    true_data: np.ndarray = None,
    title: str = "2D Sample Distribution",
    k=None,
) -> None:
    """
    Animate 2D samples with optional true data overlay.
    samples: shape (N, num_samples, 2) where N is the number of timesteps.
    true_data: shape (num_samples, 2)
    """
    score_fn = get_score_function(sde, score_model, state, config)
    k = samples.shape[0] // 10 if k is None else k

    fig, ax = plt.subplots(figsize=(7, 7))
    x = jnp.linspace(-4, 4, 25)
    y = jnp.linspace(-4, 4, 25)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.stack([X, Y], axis=-1).reshape(-1, 2)

    def animate(i):
        i = i * k  # Only animate every kth frame

        t = 1.0 - i / samples.shape[0]
        t = jnp.repeat(t, grid.shape[0])
        scores = score_fn(grid, t=t)
        # print(jnp.mean(jnp.linalg.norm(scores, axis=-1)))

        ax.clear()
        generated_points = samples[i, :, :]
        if true_data is not None:
            ax.scatter(true_data[:, 0], true_data[:, 1], alpha=0.6, color="blue")
        ax.scatter(
            generated_points[:, 0], generated_points[:, 1], alpha=0.4, color="red"
        )
        ax.quiver(X, Y, scores[:, 0], scores[:, 1], color="green")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend(["True Data", "Generated Data"])
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    anim = animation.FuncAnimation(
        fig, animate, frames=samples.shape[0] // k + 1, interval=100, repeat=False
    )

    # Save animation to prevent garbage collection
    plt.suptitle(title)
    plt.tight_layout()
    plt.close()
    return anim
