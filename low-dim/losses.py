# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization."""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul
import optax
from gmm_utils import eval_log_gmm


def get_optimizer(config):
    """Returns an optax GradientTransformation based on `config`."""
    if config.optim.optimizer == "Adam":
        # Create the learning rate schedule first
        if config.optim.warmup > 0:
            schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=config.optim.lr,
                transition_steps=config.optim.warmup,
            )
        else:
            schedule = (
                lambda _: config.optim.lr
            )  # Convert constant to schedule function

        # Create a list of transformations
        transforms = []

        # Add gradient clipping if enabled
        if config.optim.grad_clip >= 0:
            transforms.append(optax.clip_by_global_norm(config.optim.grad_clip))

        beta2 = config.optim.beta2 if hasattr(config.optim, "beta2") else 0.999
        # Add Adam scaling
        transforms.append(
            optax.scale_by_adam(b1=config.optim.beta1, b2=beta2, eps=config.optim.eps)
        )

        # Add learning rate scaling using schedule
        transforms.append(optax.scale_by_schedule(schedule))

        # Add negative scaling for gradient descent
        transforms.append(optax.scale(-1.0))

        # Chain all transformations together
        optimizer = optax.chain(*transforms)
        return optimizer
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""
    optimizer = get_optimizer(config)

    def optimize_fn(state, grad):
        """Optimizes using the pre-configured optax optimizer."""
        updates, new_opt_state = optimizer.update(grad, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        return state.replace(params=new_params, opt_state=new_opt_state)

    return optimize_fn


def get_sde_loss_fn(
    sde,
    model,
    train,
    reduce_mean=True,
    continuous=True,
    likelihood_weighting=True,
    eps=1e-5,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = mutils.get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            continuous=continuous,
            return_state=True,
        )
        data = batch["image"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

        if not likelihood_weighting:
            losses = jnp.square(batch_mul(score, std) + z)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        else:
            g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
            losses = jnp.square(score + batch_mul(z, 1.0 / std))
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_true_gmm_weights(xs, xts, std, std_prime, means, covs, gmm_weights):
    """Computes the true importance sampling weights for a GMM (given by means, covs, weights)."""
    log_gmm_std_prime = jax.vmap(
        lambda xt, std_pr: eval_log_gmm(
            xt[None], means, covs, gmm_weights, perturb_std=std_pr
        )
    )(xts, std_prime)[
        :, 0
    ]  # q_{std_prime}(xt)
    log_gmm_std = eval_log_gmm(
        xts, means, covs, gmm_weights, perturb_std=std
    )  # q_{std}(xt)

    log_gauss_std_xts_given_xs = -0.5 * jnp.sum((xts - xs) ** 2, axis=-1) / (std**2)
    log_gauss_std_prime_xts_given_xs = (
        -0.5 * jnp.sum((xts - xs) ** 2, axis=-1) / (std_prime**2)
    )

    return jnp.exp(
        log_gmm_std_prime
        + log_gauss_std_xts_given_xs
        - log_gmm_std
        - log_gauss_std_prime_xts_given_xs
    )


def get_is_weights(xs, xts, std=0.001, std_prime=3.0):
    """Computes importance sampling weights based on posterior ratios.

    Args:
        xs: Array of shape (N, D) containing N data points of dimension D.
            These are the original, unperturbed data points.
        xts: Array of shape (N, D) containing N perturbed data points.
            These should be noisy versions of xs.
        std: Float, standard deviation for the small noise distribution.
        std_prime: Float, standard deviation for the large noise distribution.

    Returns:
        weights: Array of shape (N,) containing importance sampling weights
            computed as exp(log p(x|xt,std) - log p(x|xt,std_prime)) for each
            data point, where p(x|xt,std) is the posterior probability under
            a Gaussian noise model with standard deviation std.
    """

    def compute_posterior_ratio(xs, x_idx, xt, std):
        """Computes the log ratio of posterior probabilities for a given data point.

        Args:
            xs: Array of shape (N, D) containing N data points of dimension D
            x_idx: Integer index of the reference point in xs that produced xt
            xt: Array of shape (D,) containing the perturbed point \tilde{x}
            std: Float standard deviation of the distribution

        Returns:
            ratio: Float containing log(p(xs[x_idx]|xt)/sum_i p(xs[i]|xt)) where p(x|xt)
                is the Gaussian probability of x given xt with variance std^2
        """
        diff = xs - xt
        log_pdf = -0.5 * jnp.sum(diff**2, axis=1) / (std**2)

        # Compute logsumexp of log probabilities
        log_sum_gaussians = jax.scipy.special.logsumexp(log_pdf)
        x_log_pdf = log_pdf[x_idx]

        ratio = x_log_pdf - log_sum_gaussians
        return ratio

    post_ratio_fn = lambda x_idx, xt, sigma: compute_posterior_ratio(
        xs, x_idx, xt, std=sigma
    )

    # # Vectorize post_ratio_std over both x_idx and xts
    post_ratios_std = jax.vmap(post_ratio_fn, in_axes=(0, 0, None))(
        jnp.arange(xts.shape[0]), xts, std
    )
    std_prime = (
        jnp.repeat(std_prime, xts.shape[0])
        if isinstance(std_prime, float)
        else std_prime
    )
    post_ratios_std_prime = jax.vmap(post_ratio_fn, in_axes=(0, 0, 0))(
        jnp.arange(xts.shape[0]), xts, std_prime
    )

    weights = jnp.exp(post_ratios_std - post_ratios_std_prime)
    return weights


def get_is_weights_separate_batches(xs0, xs1, xs2, xts0, std_fixed=0.001, std=3.0):
    """
    Computes importance sampling weights.
    xs0: Array of shape (N, D) containing N data points. Used to estimate the outer expectation.
    xs1: Array of shape (N, D) containing N data points. Used to estimate the numerator of the ratio.
    xs2: Array of shape (N, D) containing N data points. Used to estimate the denominator of the ratio.
    xts0: Array of shape (N, D) containing N perturbed data points (xs0 + sigma * eps). Used to estimate the outer expectation.
    std_fixed: Float, standard deviation for the small noise distribution.
    std: Float, standard deviation for the large noise distribution.
    """

    def log_expected_likelihood_ratio(x, xt, xps, std):
        """
        Computes log(E_{xp}[p(xt|xp,std) / p(xt|x,std)]) for a single (x, xt, std) and a batch of xps.
        """
        log_pdf_xt_given_xps = -0.5 * jnp.sum((xps - xt) ** 2, axis=1) / (std**2)
        log_pdf_xt_given_xs = -0.5 * jnp.sum((xt - x) ** 2) / (std**2)
        log_ratios = log_pdf_xt_given_xps - log_pdf_xt_given_xs
        return jax.scipy.special.logsumexp(log_ratios) - jnp.log(xps.shape[0])

    log_numerator_fn = lambda x, xt, sigma: log_expected_likelihood_ratio(
        x, xt, xs1, sigma
    )
    log_denominator_fn = lambda x, xt: log_expected_likelihood_ratio(
        x, xt, xs2, std_fixed
    )

    log_numerator = jax.vmap(log_numerator_fn, in_axes=(0, 0, 0))(xs0, xts0, std)
    log_denominator = jax.vmap(log_denominator_fn, in_axes=(0, 0))(xs0, xts0)
    weights = jnp.exp(log_numerator - log_denominator)
    # jax.debug.print("weights={weights}", weights=weights)
    # jax.debug.print("avg_weight={avg_weight}, max_weight={max_weight}, min_weight={min_weight}", avg_weight=jnp.mean(weights), max_weight=jnp.max(weights), min_weight=jnp.min(weights))
    return weights


def get_fixed_sigma_no_disco_loss_fn(
    vesde, model, train, fixed_sigma, reduce_mean=False
):
    """Loss function for training a (baseline) model with a fixed sigma, without DISCO (no importance sampling)."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."
    assert fixed_sigma is not None, "fixed_sigma must be provided"

    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        eps = random.normal(step_rng, data.shape)
        sigma_times_eps = eps * fixed_sigma
        perturbed_data = sigma_times_eps + data

        # We pass None as the label, since we will not use it.
        score, new_model_state = model_fn(perturbed_data, None, rng=step_rng)

        # We regress -eps, since we are using a fixed sigma.
        losses = jnp.square(-eps - score)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, (loss, new_model_state)

    return loss_fn


def sample_gmm_posterior(xt, means, covs, gmm_weights, std, rng):
    """
    Sample from the GMM posterior p(x|xt,std) where p(x) is a GMM with means, covs, and gmm_weights,
    and p(xt|x,std) is a Gaussian with mean x and covariance std^2.
    """

    rng, _ = random.split(rng)
    bmm = jax.vmap(lambda x, y: x @ y)
    inv_covs = jnp.linalg.inv(covs)
    posterior_covs = jnp.linalg.inv(
        inv_covs + 1 / (std**2) * jnp.eye(covs.shape[-1])[None, :, :]
    )
    transformed_means = bmm(inv_covs, means)
    v = transformed_means + 1 / (std**2) * xt
    posterior_means = bmm(posterior_covs, v)

    fluffy_covs = covs + std**2 * jnp.eye(covs.shape[-1])[None, :, :]
    log_prob_fn = lambda m, c: jax.scipy.stats.multivariate_normal.logpdf(xt, m, c)
    log_probs = jax.vmap(log_prob_fn)(means, fluffy_covs)  # Shape: (k, B)
    log_probs = log_probs + jnp.log(gmm_weights)  # Shape: (k, B)
    Z = jax.scipy.special.logsumexp(log_probs)
    posterior_weights = jnp.exp(log_probs - Z)

    # ancestral sampling from posterior GMM
    component_idx = jax.random.choice(rng, gmm_weights.shape[0], p=posterior_weights)
    return jax.random.multivariate_normal(
        rng, posterior_means[component_idx], posterior_covs[component_idx]
    )


def get_nearest_neighbor(x, batch):
    """
    Get the nearest neighbor of x in the batch.
    """
    distances = jnp.linalg.norm(batch - x, axis=-1)
    return batch[jnp.argmin(distances)]


def posterior_expectation(x, batch, sigma, rng):
    """
    Compute the expectation of x under the posterior p_0(x | x_t) where sigma is sigma(0), the small std of the noise.
    We don't use rng here, but keep it for consistency with the other functions.
    """
    probs = get_p0_posterior(x, batch, sigma)
    probs = probs.reshape(
        [-1] + [1] * (len(batch.shape) - 1)
    )  # add singleton dimensions to match batch shape
    return jnp.sum(batch * probs, axis=0)


def get_p0_posterior(x, batch, sigma):
    """
    Returns p_0(x | x_t) where sigma is sigma(0), the small std of the noise.
    """
    x = x.reshape(-1)
    batch = batch.reshape(batch.shape[0], -1)
    assert (
        x.shape[-1] == batch.shape[-1]
    ), f"x shape {x.shape[-1]} does not match batch shape {batch.shape[-1]}"
    # jax.debug.print("x={x}, batch={batch}", x=x.shape, batch=batch.shape)

    distances = -0.5 * jnp.sum((batch - x) ** 2, axis=-1) / (sigma**2)
    # softmax over the distances
    probs = jax.nn.softmax(distances)
    return probs


def sample_p0_posterior(x, batch, sigma, rng, return_probs=False, top_k_approx=None):
    """
    Get the nearest neighbor of x in the batch.
    """
    rng, _ = random.split(rng)
    og_shape = batch.shape
    probs = get_p0_posterior(x, batch, sigma)

    # take the top k probabilities and re-normalize them
    if top_k_approx is not None:
        k = min(top_k_approx, probs.shape[0])
        k = max(k, 1)
        top_idx = jnp.argsort(probs)[-k:]
        mask = jnp.zeros_like(probs).at[top_idx].set(1.0)
        probs = probs * mask
        probs = probs / jnp.sum(probs)

    choice_int = random.choice(rng, batch.shape[0], p=probs)
    data_choice = batch[choice_int].reshape(og_shape[1:])

    # # compute entropy of probs
    # entropy = -jnp.sum(probs * jnp.log(probs))
    # sorted_probs = jnp.sort(probs)[::-1]
    # jax.debug.print("entropy={entropy}", entropy=entropy)
    # jax.debug.print("sorted_probs={sorted_probs}", sorted_probs=sorted_probs)

    if return_probs:
        return data_choice, probs  # , entropy
    else:
        return data_choice


def get_weight_free_disco_loss_fn(
    vesde,
    model,
    train,
    reduce_mean=False,
    fixed_sigma_is=None,
    loss_variant=None,
    net_output="sigma_times_eps",
    means=None,
    covs=None,
    gmm_weights=None,
    true_gmm_weights=False,
    full_dataset=None,
    sample_posterior=True,  # if False, we compute the posterior expectation instead of sampling from it (should be lower variance)
    top_k_approx=None,  # if an integer, we approximate the posterior by taking the top k probabilities and re-normalizing them
):
    assert isinstance(
        vesde, VESDE
    ), "Weight-free DISCO training is only implemented for VESDEs right now."
    assert (
        fixed_sigma_is is not None
    ), "Weight-free DISCO training should only be used with DISCO (fixed sigma)."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        # jax.debug.print("sigmas={sigmas}", sigmas=sigmas)

        rng, step_rng = random.split(rng)
        eps = random.normal(step_rng, data.shape)
        sigma_times_eps = batch_mul(eps, sigmas)
        perturbed_data = sigma_times_eps + data
        rng, step_rng = random.split(rng)
        model_out, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)

        # nearest_clean_x = jax.vmap(lambda x: get_nearest_neighbor(x, data))(perturbed_data)
        rng, step_rng = random.split(rng)
        if true_gmm_weights:
            # sample from the posterior GMM
            nearest_clean_x = jax.vmap(
                lambda xt: sample_gmm_posterior(
                    xt, means, covs, gmm_weights, fixed_sigma_is, step_rng
                )
            )(perturbed_data)
        else:
            x_batch = full_dataset if full_dataset is not None else data
            if top_k_approx is not None and sample_posterior:
                get_target_fn = lambda x, x_batch, sigma, rng: sample_p0_posterior(
                    x, x_batch, sigma, rng, top_k_approx=top_k_approx
                )
            else:
                get_target_fn = (
                    sample_p0_posterior if sample_posterior else posterior_expectation
                )

            nearest_clean_x = jax.vmap(
                lambda x: get_target_fn(x, x_batch, fixed_sigma_is, step_rng)
            )(perturbed_data)

        assert (
            nearest_clean_x.shape == data.shape
        ), f"Nearest neighbor shape {nearest_clean_x.shape} does not match data shape {data.shape}"

        # check how many nearest neighbors are the same as the data
        # same_as_perturbed = jnp.all(nearest_clean_x == data, axis=-1).sum() / data.shape[0]
        # jax.debug.print("same_as_perturbed={same_as_perturbed}", same_as_perturbed=same_as_perturbed)

        if net_output == "sigma_times_eps":
            target = (
                nearest_clean_x - perturbed_data
            )  # this is the equivalent to -sigma_times_eps in the original disco loss
        elif net_output == "x_pred":
            target = nearest_clean_x
        elif net_output == "eps":
            # TODO: eps is meaningless in weight-free DISCO, since it's only used to generate a pair (x, x_perturbed).
            #       We don't have x_perturbed = x + sigma * eps here.
            assert False, "eps is meaningless in weight-free DISCO"
            target = -eps
            model_out = batch_mul(model_out, 1.0 / sigmas)
        else:
            raise ValueError(f"Net output {net_output} not supported")

        losses = jnp.square(target - model_out)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        if loss_variant == "sigma_sq_weighting":
            loss = jnp.mean(sigmas**2 * losses)  # we scale by sigmas^2
        elif loss_variant == "inv_sigma_sq_weighting":
            loss = jnp.mean(losses / sigmas**2)  # we scale by 1/sigmas^2
        elif loss_variant is not None and loss_variant != "None":
            raise ValueError(f"Loss variant {loss_variant} not supported")
        else:
            loss = jnp.mean(losses)

        return loss, (loss, new_model_state)

    return loss_fn


def get_uedm_loss_fn(vesde, model, train):
    assert isinstance(
        vesde, VESDE
    ), "uEDM training is only implemented for VESDEs right now."
    assert (
        not model.config.model.conditional
    ), "uEDM training requires unconditional training"

    sigma_data = getattr(model.config.model, "sigma_data", 0.5)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        per_gpu_batch_size = data.shape[0] // jax.local_device_count()
        # jax.debug.print("data.min={data_min}, data.max={data_max}", data_min=jnp.min(data), data_max=jnp.max(data))

        # draw log(sigma) ~ N(-1.2, 1.2^2)
        rng, step_rng = random.split(rng)
        log_sigma = random.normal(step_rng, data.shape[0]) * 1.2 - 1.2
        sigma = jnp.exp(log_sigma)

        # draw eps ~ N(0, 1)
        rng, step_rng = random.split(rng)
        eps = random.normal(step_rng, data.shape)

        a = 1 / jnp.sqrt(sigma**2 + 1)
        b = sigma / jnp.sqrt(sigma**2 + 1)
        c = sigma**2 / (sigma**2 + sigma_data**2)
        d = -sigma * sigma_data**2 / (sigma**2 + sigma_data**2)
        weights = (sigma_data**2 + sigma**2) / (sigma_data * sigma)

        z = batch_mul(a, data) + batch_mul(b, eps)
        r_x_eps_t = batch_mul(c, data) + batch_mul(d, eps)

        rng, step_rng = random.split(rng)
        model_out, new_model_state = model_fn(z, None, rng=step_rng)

        losses = batch_mul(weights, (model_out - r_x_eps_t) ** 2)
        loss = jnp.sum(losses) / per_gpu_batch_size

        return loss, (loss, new_model_state)

    return loss_fn


def get_eps_pred_loss_fn(
    vesde,
    model,
    train,
    reduce_mean=False,
    fixed_sigma_is=None,
    loss_variant=None,
    net_output="sigma_times_eps",
    means=None,
    covs=None,
    gmm_weights=None,
    true_gmm_weights=False,
):
    assert isinstance(
        vesde, VESDE
    ), "Eps prediction training is only implemented for VESDEs right now."
    assert (
        not model.config.model.scale_by_sigma
    ), "Eps prediction training requires scale_by_sigma=False"

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        if fixed_sigma_is is not None:  # DISCO
            assert (
                data.shape[0] % 3 == 0
            ), "Data must be divisible by 3 for DISCO (weight estimation needs 3 independent batches)"
            batch_size = data.shape[0] // 3
            data, xs1, xs2 = (
                data[:batch_size],
                data[batch_size : 2 * batch_size],
                data[2 * batch_size :],
            )

        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        # jax.debug.print("sigmas={sigmas}", sigmas=sigmas)

        rng, step_rng = random.split(rng)
        eps = random.normal(step_rng, data.shape)
        sigma_times_eps = batch_mul(eps, sigmas)
        perturbed_data = sigma_times_eps + data
        rng, step_rng = random.split(rng)
        model_out, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)

        if net_output == "sigma_times_eps":
            target = -sigma_times_eps
        elif net_output == "eps":
            target = -eps
        elif net_output == "x_pred":
            target = data
        elif net_output == "score":
            target = -batch_mul(eps, 1.0 / sigmas)  # this is the conditional score
        else:
            raise ValueError(f"Net output {net_output} not supported")

        # if fixed_sigma_is is None and not model.config.model.conditional:
        #     # we also try No-DISCO, but unconditional, i.e., || eps - eps(x)/sigma ||^2
        #     assert (
        #         net_output == "eps"
        #     ), "Only eps prediction is supported for unconditional no-DISCO"
        #     model_out = batch_mul(model_out, 1.0 / sigmas)

        # if fixed_sigma_is is not None:
        #     if not model.config.model.conditional:
        #         target = -sigma_times_eps
        #     else:
        #         target = -eps

        #     model_out_over_sigma = model_out # batch_mul(model_out, 1. / sigmas)
        # else:
        #     # we should not scale by sigma in the no-disco case (also not within the model)
        #     target = -eps
        #     model_out_over_sigma = model_out

        losses = jnp.square(model_out - target)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        unweighted_loss = jnp.mean(
            losses
        )  # no need to explicitly scale by sigmas^2, this is implicitly done by the loss function

        if fixed_sigma_is is None:
            # No DISCO
            if loss_variant == "sigma_sq_weighting":
                loss = jnp.mean(sigmas**2 * losses)
            elif loss_variant == "inv_sigma_sq_weighting":
                loss = jnp.mean(losses / sigmas**2)
            elif loss_variant is not None and loss_variant != "None":
                raise ValueError(f"Loss variant {loss_variant} not supported")
            else:
                # default behavior
                loss = unweighted_loss
        else:
            # DISCO
            if not true_gmm_weights:
                if loss_variant == "disco_set_weights_1":
                    # we neglect the IS weights, like in the 2019 paper (not sound)
                    weights = jnp.ones((data.shape[0],))
                else:
                    weights = get_is_weights_separate_batches(
                        xs0=data.reshape(data.shape[0], -1),
                        xs1=xs1.reshape(xs1.shape[0], -1),
                        xs2=xs2.reshape(xs2.shape[0], -1),
                        xts0=perturbed_data.reshape(perturbed_data.shape[0], -1),
                        std_fixed=fixed_sigma_is,
                        std=sigmas,
                    )

                assert weights.shape == (
                    data.shape[0],
                ), f"Weights shape {weights.shape} does not match data shape {data.shape}"

                # weights = get_is_weights(xs=data.reshape(data.shape[0], -1),
                #                         xts=perturbed_data.reshape(perturbed_data.shape[0], -1),
                #                         std=fixed_sigma_is, std_prime=sigmas)
            else:
                weights = get_true_gmm_weights(
                    xs=data.reshape(data.shape[0], -1),
                    xts=perturbed_data.reshape(perturbed_data.shape[0], -1),
                    std=fixed_sigma_is,
                    std_prime=sigmas,
                    means=means,
                    covs=covs,
                    gmm_weights=gmm_weights,
                )

            if loss_variant == "disco_clip_weights_1e-6":
                # clamp weights that are smaller than 1e-6 to 1e-6
                weights = jnp.where(weights < 1e-6, 1e-6, weights)
            elif loss_variant == "disco_normalize_weights":
                weights = weights / jnp.sum(weights)
            elif loss_variant == "sigma_sq_weighting":
                weights = weights * sigmas**2
            elif loss_variant == "inv_sigma_sq_weighting":
                weights = weights / sigmas**2
            elif loss_variant == "disco_set_weights_1":
                # we neglect the IS weights, like in the 2019 paper (not sound)
                weights = jnp.ones_like(weights)
            elif loss_variant == "inv_sigma_sq_set_weights_1":
                # we neglect the IS weights, like in the 2019 paper (not sound)
                weights = jnp.ones_like(weights) / sigmas**2
            else:
                if loss_variant is not None and loss_variant != "None":
                    raise ValueError(f"Loss variant {loss_variant} not supported")

            # DISCO scales by sigmas^2. If we wouldn't, the implicit weights would be w(\sigma_i) = 1/\sigma_i^2.
            # This would give more weight to the smaller sigmas, which is not what we want.
            # loss = jnp.mean(sigmas**2 * weights * losses)
            loss = jnp.mean(weights * losses)

        return loss, (unweighted_loss, new_model_state)

    return loss_fn


def get_smld_loss_fn(
    vesde, model, train, reduce_mean=False, fixed_sigma_is=None, loss_variant=None
):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        # jax.debug.print("sigmas={sigmas}", sigmas=sigmas)

        rng, step_rng = random.split(rng)
        eps = random.normal(step_rng, data.shape)
        sigma_times_eps = batch_mul(eps, sigmas)
        perturbed_data = sigma_times_eps + data
        rng, step_rng = random.split(rng)
        model_out, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        # jax.debug.print("{data}", data=data)

        if fixed_sigma_is is None:
            score = model_out
            target = -batch_mul(sigma_times_eps, 1.0 / (sigmas**2))
            losses = jnp.square(score - target)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            losses = sigmas**2 * losses  # this sets w(\sigma_i) = \sigma_i^2.
            unweighted_loss = jnp.mean(losses)
        else:
            # target = -sigma_times_eps # in importance sampling, we always regress the score of p_{fixed_sigma_is}
            target = (
                -eps
            )  # in importance sampling, we always regress the score of p_{fixed_sigma_is}
            weights = get_is_weights(
                xs=data.reshape(data.shape[0], -1),
                xts=perturbed_data.reshape(perturbed_data.shape[0], -1),
                std=fixed_sigma_is,
                std_prime=sigmas,
            )
            model_out_over_sigma = batch_mul(model_out, 1.0 / sigmas)
            losses = jnp.square(model_out_over_sigma - target)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

            unweighted_loss = jnp.mean(
                sigmas**2 * losses
            )  # just for logging and comparing to non-DISCO

            if loss_variant == "disco_clip_weights_1e-6":
                # clamp weights that are smaller than 1e-6 to 1e-6
                weights = jnp.where(weights < 1e-6, 1e-6, weights)
            elif loss_variant == "disco_set_weights_1":
                # what if we neglect the IS weights? This is technically not quite sound, but the weights are 1 in many cases.
                weights = jnp.ones_like(weights)
            elif loss_variant == "disco_normalize_weights":
                weights = weights / jnp.sum(weights)
            else:
                if loss_variant is not None and loss_variant != "None":
                    raise ValueError(f"Loss variant {loss_variant} not supported")

            losses = weights * sigmas**2 * losses
            # losses = weights * losses

        loss = unweighted_loss if fixed_sigma_is is None else jnp.mean(losses)
        return loss, (unweighted_loss, new_model_state)

    return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + batch_mul(
            sqrt_1m_alphas_cumprod[labels], noise
        )
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        # we return loss == unweighted_loss
        return loss, (loss, new_model_state)

    return loss_fn


def get_step_fn(
    sde,
    model,
    train,
    net_output,
    optimize_fn=None,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
    fixed_sigma_is=None,
    fixed_sigma_baseline=None,
    loss_variant=None,
    weight_free_disco=False,
    uEDM=False,
    means=None,
    covs=None,
    gmm_weights=None,
    true_gmm_weights=False,
    full_dataset=None,
    sample_posterior=True,
    top_k_approx=None,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            model,
            train,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=likelihood_weighting,
        )
    else:
        assert (
            not likelihood_weighting
        ), "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            if fixed_sigma_baseline is not None:
                # this trains a baseline model with a fixed sigma, without DISCO.
                loss_fn = get_fixed_sigma_no_disco_loss_fn(
                    sde,
                    model,
                    train,
                    fixed_sigma=fixed_sigma_baseline,
                    reduce_mean=reduce_mean,
                )
            elif weight_free_disco:
                loss_fn = get_weight_free_disco_loss_fn(
                    sde,
                    model,
                    train,
                    reduce_mean=reduce_mean,
                    fixed_sigma_is=fixed_sigma_is,
                    loss_variant=loss_variant,
                    net_output=net_output,
                    means=means,
                    covs=covs,
                    gmm_weights=gmm_weights,
                    true_gmm_weights=true_gmm_weights,
                    full_dataset=full_dataset,
                    sample_posterior=sample_posterior,
                    top_k_approx=top_k_approx,
                )
            elif uEDM:
                loss_fn = get_uedm_loss_fn(sde, model, train)
            else:
                loss_fn = get_eps_pred_loss_fn(
                    sde,
                    model,
                    train,
                    reduce_mean=reduce_mean,
                    fixed_sigma_is=fixed_sigma_is,
                    loss_variant=loss_variant,
                    net_output=net_output,
                    means=means,
                    covs=covs,
                    gmm_weights=gmm_weights,
                    true_gmm_weights=true_gmm_weights,
                )
                # loss_fn = get_smld_loss_fn(
                #     sde, model, train, reduce_mean=reduce_mean, fixed_sigma_is=fixed_sigma_is, loss_variant=loss_variant)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = state.params
            states = state.model_state
            (loss, (unweighted_loss, new_model_state)), grad = grad_fn(
                step_rng, params, states, batch
            )
            grad = jax.lax.pmean(grad, axis_name="batch")
            new_state = optimize_fn(state, grad)
            new_params_ema = jax.tree_map(
                lambda p_ema, p: p_ema * new_state.ema_rate
                + p * (1.0 - new_state.ema_rate),
                new_state.params_ema,
                new_state.params,
            )
            step = new_state.step + 1
            new_state = new_state.replace(
                step=step, model_state=new_model_state, params_ema=new_params_ema
            )
        else:
            loss, (unweighted_loss, _) = loss_fn(
                step_rng, state.params_ema, state.model_state, batch
            )
            new_state = state

        loss = jax.lax.pmean(loss, axis_name="batch")
        unweighted_loss = jax.lax.pmean(unweighted_loss, axis_name="batch")
        new_carry_state = (rng, new_state)
        return new_carry_state, (loss, unweighted_loss)

    return step_fn
