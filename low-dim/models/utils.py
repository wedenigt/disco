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

"""All functions and modules related to model definition."""
from typing import Any

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import checkpoints
from utils import batch_mul
import optax
from flax import core as flax_core
import flax.linen as nn


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
    step: int
    opt_state: Any
    params: Any
    model_state: Any
    ema_rate: float
    params_ema: Any
    rng: Any


@flax.struct.dataclass
class OldState:
    step: int
    optimizer: Any
    lr: float
    model_state: Any
    ema_rate: float
    params_ema: Any
    rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        # if local_name in _MODELS:
        #   raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = jnp.exp(
        jnp.linspace(
            jnp.log(config.model.sigma_max),
            jnp.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def init_model(rng, config):
    """Initialize a `flax.linen.Module` model."""
    model_name = config.model.name
    model_def = functools.partial(get_model(model_name), config=config)
    if config.data.image_size == 1:
        # squeeze the input, this is for low-dimensional data
        input_shape = (jax.local_device_count(), config.data.num_channels)
    else:
        input_shape = (
            jax.local_device_count(),
            config.data.image_size,
            config.data.image_size,
            config.data.num_channels,
        )

    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    model = model_def()
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng}, fake_input, fake_label
    )
    init_model_state, initial_params = flax_core.pop(variables, "params")
    return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False, return_only_neg_energy=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: A `flax.linen.Module` object the represent the architecture of score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all mutable states.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels, rng=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          rng: If present, it is the random state for dropout

        Returns:
          A tuple of (model output, new mutable states)
        """
        variables = {"params": params, **states}
        if not train:
            if "ebm" in model.config.model.name:
                return (
                    model.apply(
                        variables,
                        x,
                        labels,
                        train=False,
                        mutable=False,
                        return_only_neg_energy=return_only_neg_energy,
                    ),
                    states,
                )
            else:
                return (
                    model.apply(variables, x, labels, train=False, mutable=False),
                    states,
                )
        else:
            rngs = {"dropout": rng}
            return model.apply(
                variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs
            )
            # if states:
            #   return outputs
            # else:
            #   return outputs, states

    return model_fn


def get_score_fn(
    sde,
    model,
    params,
    states,
    train=False,
    continuous=False,
    return_state=False,
    fixed_sigma=None,
    fixed_sigma_baseline=None,
    guidance_alpha=0.0,
    guidance_cond_values=None,
    guidance_cond_indices=None,
    is_uedm=False,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
      return_state: If `True`, return the new mutable states alongside the model output.
      fixed_sigma: If not None, we use DISCO and the score is scaled by 1/sigma^2.
      fixed_sigma_baseline: If not None, we train a baseline model with a fixed sigma, without DISCO.
      guidance_alpha: If not 0, we use gradient guidance with this coefficient.
      guidance_cond_values: If not None, we use gradient guidance with these values.
      guidance_cond_indices: If not None, we use gradient guidance on the values at the specified indices.
    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, params, states, train=train)

    if not hasattr(model.config.model, "net_output"):
        model.config.model.net_output = (
            "eps" if not fixed_sigma else "sigma_times_eps"
        )  # set default

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t, rng=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                model, state = model_fn(x, labels, rng)
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                model, state = model_fn(x, labels, rng)
                std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

            score = batch_mul(-model, 1.0 / std)
            if return_state:
                return score, state
            else:
                return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t, rng=None, label_input=False):
            if continuous:
                labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                # I don't think that's true -- sigmas is an array from large to small; if t=T, then labels should be 0, and the highest noise level
                # So, t=0 should be the lowest noise level
                if label_input:
                    labels = t
                else:
                    labels = sde.T - t
                    labels *= sde.N - 1
                    labels = jnp.round(labels).astype(jnp.int32)

            if is_uedm:
                sigma_data = 0.5
                c_skip = sigma_data**2 / (sigma_data**2 + t**2)
                c_out = 1.0  # constant in uEDM paper
                c_in = 1 / jnp.sqrt(t**2 + 1.0)
                model_input = batch_mul(c_in, x)
                net_out, state = model_fn(model_input, None, rng)  # no labels for uEDM
                denoiser = batch_mul(c_skip, x) + c_out * net_out
                score = batch_mul((denoiser - x), 1.0 / t**2)
                return (score, state) if return_state else score

            score, state = model_fn(x, labels, rng)

            if model.config.model.name == "gmm_score_wrapper":
                return score  # this is already the score, do not post-process

            if fixed_sigma is not None:
                # DISCO
                assert (
                    guidance_alpha == 0.0
                ), "Gradient guidance is not supported/needed for DISCO."

                if model.config.model.conditional:
                    assert (
                        not model.config.training.weight_free_disco
                    ), "Conditional is not supported for weight-free DISCO."
                    used_sigmas = sde.discrete_sigmas[::-1][labels].reshape(
                        (x.shape[0], *([1] * len(x.shape[1:])))
                    )
                    return used_sigmas * score / (fixed_sigma**2)
                else:
                    if model.config.model.net_output in ["sigma_times_eps", "eps"]:
                        score = score / (
                            fixed_sigma**2
                        )  # regular DISCO and weight-free DISCO use this
                    elif model.config.model.net_output == "x_pred":
                        score = (score - x) / (fixed_sigma**2)  # x_pred uses this
                    else:
                        raise ValueError(
                            f"Invalid net_output: {model.config.model.net_output}"
                        )

            elif fixed_sigma_baseline is not None:
                # jax.debug.print('{}', fixed_sigma_baseline)
                score = (
                    score / fixed_sigma_baseline
                )  # in the baseline, scale by 1/sigma, not 1/sigma^2
            else:
                # No DISCO
                assert (
                    not model.config.model.scale_by_sigma
                ), "We use eps prediction, so we don't need to scale by sigma within the model."
                # we have to reverse sigmas because we want to start with large sigmas
                used_sigmas = sde.discrete_sigmas[::-1][labels].reshape(
                    (x.shape[0], *([1] * len(x.shape[1:])))
                )

                # jax.debug.print('score_fn: {}, {}, {}', labels[0], t[0], used_sigmas[0])

                if model.config.model.net_output == "eps":
                    if model.config.model.conditional:
                        # this is the default
                        score = score / used_sigmas
                    else:
                        # this is just a test to see if uncondtional models behave similarly to DISCO
                        score = score / used_sigmas**2
                elif model.config.model.net_output == "x_pred":
                    score = (score - x) / (used_sigmas**2)  # x_pred uses this
                elif model.config.model.net_output == "score":
                    score = score  # we directly regress the score
                elif model.config.model.net_output == "sigma_times_eps":
                    score = score / (used_sigmas**2)  # sigma_times_eps uses this
                else:
                    raise ValueError(
                        f"Invalid net_output: {model.config.model.net_output}"
                    )

                if (
                    guidance_cond_values is not None
                    and guidance_cond_indices is not None
                ):
                    # apply gradient guidance
                    if guidance_alpha == 0.0:
                        # jax.debug.print('variable guidance alpha')
                        vars = sde.discrete_sigmas[::-1][labels].reshape(-1) ** 2
                    else:
                        vars = guidance_alpha * jnp.ones(x.shape[0])

                    assert (
                        model.config.model.net_output == "eps"
                    ), "Gradient guidance is only supported for eps prediction right now."
                    cond_values = guidance_cond_values[guidance_cond_indices]

                    def log_gauss_y_given_x_noisy(x_noisy, label, variance):
                        # x_noisy and sigma are a single object, not a batch
                        # jax.debug.print('grad guidance')
                        sigma = sde.discrete_sigmas[::-1][label]
                        score_x = (
                            model_fn(x_noisy[None, ...], label[None, ...], rng)[
                                0
                            ].squeeze()
                            / sigma
                        )
                        denoised_x = x_noisy + sigma**2 * score_x  # tweedie
                        denoised_cond_values = denoised_x[guidance_cond_indices]
                        diff_l2 = jnp.sum((cond_values - denoised_cond_values) ** 2)
                        log_gauss_unnorm = -diff_l2 / (2 * variance)
                        return log_gauss_unnorm

                    grad_log_gauss_y_given_x_noisy = jax.grad(
                        log_gauss_y_given_x_noisy, argnums=0
                    )  # grad wrt x_noisy
                    grads = jax.vmap(grad_log_gauss_y_given_x_noisy)(x, labels, vars)

                    # NOTE: if guidance_cond_indices is False everywhere, grads will be constant 0
                    # This is because diff_l2 is a sum over a jnp array of shape (0,), which is constant 0 by definition
                    # and hence, the gradient of log_gauss_y_given_x_noisy w.r.t. x_noisy is 0 everywhere.
                    score = (
                        score + grads
                    )  # we push into the direction of the conditioning information

            if return_state:
                return score, state
            else:
                return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


def create_classifier(prng_key, batch_size, ckpt_path):
    """Create a noise-conditional image classifier.

    Args:
      prng_key: A JAX random state.
      batch_size: The batch size of input data.
      ckpt_path: The path to stored checkpoints for this classifier.

    Returns:
      classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
      classifier_params: A dictionary that contains trainable parameters of the classifier.
    """
    input_shape = (batch_size, 32, 32, 3)
    classifier = wideresnet_noise_conditional.WideResnet(
        blocks_per_group=4, channel_multiplier=10, num_outputs=10
    )
    initial_variables = classifier.init(
        {"params": prng_key, "dropout": jax.random.PRNGKey(0)},
        jnp.ones(input_shape, dtype=jnp.float32),
        jnp.ones((batch_size,), dtype=jnp.float32),
        train=False,
    )
    model_state, init_params = initial_variables.pop("params")
    classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
    return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
    """Create a logit function for the classifier."""

    def preprocess(data):
        image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
        image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
        return (data - image_mean[None, ...]) / image_std[None, ...]

    def logit_fn(data, ve_noise_scale):
        """Give the logits of the classifier.

        Args:
          data: A JAX array of the input.
          ve_noise_scale: time conditioning variables in the form of VE SDEs.

        Returns:
          logits: The logits given by the noise-conditional classifier.
        """
        data = preprocess(data)
        logits = classifier.apply(
            {"params": classifier_params},
            data,
            ve_noise_scale,
            train=False,
            mutable=False,
        )
        return logits

    return logit_fn


def get_classifier_grad_fn(logit_fn):
    """Create the gradient function for the classifier in use of class-conditional sampling."""

    def grad_fn(data, ve_noise_scale, labels):
        def prob_fn(data):
            logits = logit_fn(data, ve_noise_scale)
            prob = jax.nn.log_softmax(logits, axis=-1)[
                jnp.arange(labels.shape[0]), labels
            ].sum()
            return prob

        return jax.grad(prob_fn)(data)

    return grad_fn
