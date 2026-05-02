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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import abc
import flax

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from utils import batch_mul, batch_add
from jax_tqdm import loop_tqdm
from jax import lax
import blackjax

from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_temp_schedule(schedule_type, temperature_coeff=1.0):
    """Create a temperature scheduling function.
    We will always have T(0) = temperature_coeff and T(1) = 1.
    If schedule_type is 'constant', then the temperature is always temperature_coeff.

    Args:
      schedule_type: A string specifying the type of temperature schedule.
        Options are 'linear', 'cosine', 'exp', 'std', or 'constant'.
      temperature_coeff: A coefficient to scale the temperature.

    Returns:
      A function that takes time t, standard deviation std, sde, and rsde and returns the temperature.
    """
    if schedule_type.lower() == "linear":
        # Linear schedule based on time
        def temp_fn(t, std, sde, rsde, step_size):
            t_from_0 = (sde.T - t) / sde.T
            return (1 - temperature_coeff) * t_from_0 + temperature_coeff

    elif schedule_type.lower() == "cosine":
        # Cosine schedule based on time
        def temp_fn(t, std, sde, rsde, step_size):
            return temperature_coeff * (0.5 - 0.5 * jnp.cos(jnp.pi * t)) + 1.0

    elif schedule_type.lower() == "exp":
        # Exponential schedule based on time
        def temp_fn(t, std, sde, rsde, step_size):
            n = sde.N - (t * sde.N)
            # return (temperature_coeff - 1.) * 0.995**n + 1.0
            return temperature_coeff * 0.999**n

    elif schedule_type.lower() == "std_normalized":
        # Schedule based on standard deviation
        def temp_fn(t, std, sde, rsde, step_size):
            sigma_min = sde.discrete_sigmas[0]
            sigma_max = sde.discrete_sigmas[-1]
            return (temperature_coeff - 1.0) * (std - sigma_min) / (
                sigma_max - sigma_min
            ) + 1.0

            # return temperature_coeff * (std - 0.01) + 1.0

    elif schedule_type.lower() == "std":
        # Schedule based on standard deviation
        def temp_fn(t, std, sde, rsde, step_size):
            labels = sde.T - t
            labels *= sde.N - 1
            labels = jnp.round(labels).astype(jnp.int32)
            used_sigmas = sde.discrete_sigmas[::-1][labels]
            return temperature_coeff * used_sigmas

    elif schedule_type.lower() == "constant":

        def temp_fn(t, std, sde, rsde, step_size):
            return jnp.ones_like(t) * temperature_coeff

    elif schedule_type.lower() == "polynomial":
        # Polynomial step size
        def temp_fn(t, std, sde, rsde, step_size, beta=1, gamma=1.0):
            timestep = ((sde.T - t) * (sde.N) / sde.T).astype(jnp.int32)
            return temperature_coeff * (beta + timestep) ** (-gamma)

    elif schedule_type.lower() == "step_size_dependent":

        def temp_fn(t, std, sde, rsde, step_size):
            sigma_min = sde.discrete_sigmas[0]
            return temperature_coeff * step_size / sigma_min**2

    else:
        raise ValueError(
            f"Temperature schedule type {schedule_type} unknown. Options are 'linear', 'cosine', 'std', or 'constant'."
        )

    return temp_fn


def get_step_size_schedule(schedule_type, step_size_coeff=1.0):
    """Create a step size scheduling function.

    Args:
      schedule_type: A string specifying the type of step size schedule.
        Options are 'linear', 'reverse_diffusion', 'std', or 'constant'.
      step_size_coeff: A coefficient to scale the step size.

    Returns:
      A function that takes time t, standard deviation std, sde, and rsde and returns the step size.
    """
    if schedule_type.lower() == "linear":
        # Linear schedule based on time
        def step_size_fn(t, std, sde, rsde):
            return step_size_coeff * t

    elif schedule_type.lower() == "exp":
        # Exponential schedule based on time
        def step_size_fn(t, std, sde, rsde):
            n = sde.N - (t * sde.N)
            return step_size_coeff * 0.999**n

    elif schedule_type.lower() == "reverse_diffusion":
        # Reverse diffusion schedule based on time
        def step_size_fn(t, std, sde, rsde):
            # We pass zeros here because we don't need the drift term
            _, G = sde.discretize(jnp.zeros(1), t)
            return step_size_coeff * G**2

    elif schedule_type.lower() == "std_normalized":
        # Schedule based on standard deviation
        def step_size_fn(t, std, sde, rsde):
            sigma_min = sde.discrete_sigmas[0]
            sigma_max = sde.discrete_sigmas[-1]
            normalized_std = (std - sigma_min) / (sigma_max - sigma_min)
            return step_size_coeff * normalized_std

    elif schedule_type.lower() == "std":
        # Schedule based on standard deviation
        def step_size_fn(t, std, sde, rsde):
            labels = sde.T - t
            labels *= sde.N - 1
            labels = jnp.round(labels).astype(jnp.int32)
            used_sigmas = sde.discrete_sigmas[::-1][labels]
            return step_size_coeff * used_sigmas

    elif schedule_type.lower() == "constant":
        # Constant step size
        def step_size_fn(t, std, sde, rsde):
            return step_size_coeff * jnp.ones_like(t)

    elif schedule_type.lower() == "manifold_steps":

        def step_size_fn(t, std, sde, rsde, beta=1, gamma=1.0):
            sigma_min = sde.discrete_sigmas[0]
            timestep = ((sde.T - t) * (sde.N) / sde.T).astype(jnp.int32)
            # jax.debug.print("{}, {}", timestep, sigma_min**2 / (step_size_coeff * timestep))
            return jnp.ones_like(t) * sigma_min**2 * step_size_coeff
            # return step_size_coeff * sigma_min**2 * (beta + timestep)**(-gamma)

    elif schedule_type.lower() == "polynomial":
        # Polynomial step size
        def step_size_fn(t, std, sde, rsde, beta=1, gamma=1.0):
            timestep = ((sde.T - t) * (sde.N) / sde.T).astype(jnp.int32)
            # timestep = ((sde.T - t) / sde.T) * sde.N
            # jax.debug.print("{}, {}", t, timestep)
            return step_size_coeff * (beta + timestep) ** (-gamma)
            # return step_size_coeff * (step)**(-1.)

    else:
        raise ValueError(
            f"Step size schedule type {schedule_type} unknown. Options are 'linear', 'reverse_diffusion', 'std', or 'constant'."
        )

    return step_size_fn


def get_sampling_fn(
    config,
    sde,
    model,
    shape,
    inverse_scaler,
    eps,
    cond_indices=None,
    cond_values=None,
    heuristic_cond_sampling=False,
    store_intermediate_samples=False,
    guidance_alpha=0.0,
):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
      fixed_sigma: If not None, we use DISCO and the score is scaled by 1/sigma^2.
      cond_indices: If not None, it's a boolean jnp.array of the same shape as x. We condition on the RVs where cond_indices is True.
      cond_values: If not None, it's a jnp.array of the same shape as x. We condition on the values in cond_values where cond_indices is True.
    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    temperature_coeff = (
        config.sampling.temperature_coeff
        if hasattr(config.sampling, "temperature_coeff")
        else 1.0
    )
    temp_schedule = (
        config.sampling.temp_schedule
        if hasattr(config.sampling, "temp_schedule")
        else "std"
    )
    temp_scheduler = get_temp_schedule(temp_schedule, temperature_coeff)

    # Get step size schedule
    step_size_coeff = (
        config.sampling.step_size_coeff
        if hasattr(config.sampling, "step_size_coeff")
        else 1.0
    )
    step_size_schedule = (
        config.sampling.step_size_schedule
        if hasattr(config.sampling, "step_size_schedule")
        else "constant"
    )
    step_size_scheduler = get_step_size_schedule(step_size_schedule, step_size_coeff)

    fixed_sigma = (
        config.training.fixed_sigma_is
        if hasattr(config.training, "fixed_sigma_is")
        else None
    )
    fixed_sigma_baseline = (
        config.training.fixed_sigma_baseline
        if hasattr(config.training, "fixed_sigma_baseline")
        else None
    )
    sampler_name = config.sampling.method

    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            model=model,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            model=model,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            fixed_sigma=fixed_sigma,
            cond_indices=cond_indices,
            cond_values=cond_values,
            heuristic_cond_sampling=heuristic_cond_sampling,
            fixed_sigma_baseline=fixed_sigma_baseline,
            store_intermediate_samples=store_intermediate_samples,
            temperature_coeff=temperature_coeff,
            temp_scheduler=temp_scheduler,
            step_size_scheduler=step_size_scheduler,
            guidance_alpha=guidance_alpha,
        )
    elif sampler_name.lower() == "tds":
        sampling_fn = get_tds_sampler(
            sde=sde,
            model=model,
            shape=shape,
            continuous=config.training.continuous,
            cond_indices=cond_indices,
            cond_values=cond_values,
            sampling_eps=eps,
            store_intermediate_samples=store_intermediate_samples,
            n_steps_each=config.sampling.n_steps_each,
        )
    elif sampler_name.lower() == "edm":
        assert guidance_alpha == 0.0, "EDM does not support guidance"
        sampling_fn = get_edm_sampler(
            sde=sde,
            model=model,
            shape=shape,
            continuous=config.training.continuous,
            inverse_scaler=inverse_scaler,
            fixed_sigma=fixed_sigma,
            fixed_sigma_baseline=fixed_sigma_baseline,
            S_churn=config.sampling.s_churn,
            S_min=config.sampling.s_min,
            S_max=config.sampling.s_max,
            S_noise=config.sampling.s_noise,
            num_steps=config.sampling.num_steps,
        )
    elif sampler_name.lower() == "posterior_sampling":
        assert guidance_alpha == 0.0, "Posterior sampling does not support guidance"
        sampling_fn = get_posterior_sampler(
            sde=sde,
            model=model,
            shape=shape,
            continuous=config.training.continuous,
            inverse_scaler=inverse_scaler,
            fixed_sigma=fixed_sigma,
            fixed_sigma_baseline=fixed_sigma_baseline,
            S_noise=config.sampling.s_noise,
            rho=config.sampling.rho,
            inv_temp_scaler=config.sampling.inv_temp_scaler,
            num_steps=config.sampling.num_steps,
            store_intermediate_samples=store_intermediate_samples,
            cond_indices=cond_indices,
            cond_values=cond_values,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
    ):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn
        self.cond_indices = cond_indices
        self.temperature_coeff = temperature_coeff
        self.temp_scheduler = temp_scheduler
        self.step_size_scheduler = step_size_scheduler
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, rng, x, t):
        """One update of the predictor.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state
          t: A JAX array representing the current time step.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps
        self.cond_indices = cond_indices
        self.temperature_coeff = temperature_coeff
        self.temp_scheduler = temp_scheduler

    @abc.abstractmethod
    def update_fn(self, rng, x, t):
        """One update of the corrector.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state
          t: A JAX array representing the current time step.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
        )

    def update_fn(self, rng, x, t):
        dt = -1.0 / self.rsde.N
        z = random.normal(rng, x.shape)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )

    def update_fn(self, rng, x, t, denoise_step=False):
        f, G = self.rsde.discretize(x, t)
        z = random.normal(rng, x.shape)
        # x_mean = x - 0.5 * f
        x_mean = x - f  # original
        x = x_mean + batch_mul(G, z)  # original
        # x = x_mean + batch_mul(jnp.sqrt(2) * G, z) # to match langevin
        return x, x_mean


@register_predictor(name="reverse_diffusion_temp")
class ReverseDiffusionTemperaturePredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
        heuristic_cond_sampling=False,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )
        self.heuristic_cond_sampling = heuristic_cond_sampling

    def update_fn(self, rng, x, t, denoise_step=False):
        _, G = self.sde.discretize(x, t)

        # Default step size calculation
        default_step_size = G**2

        std = self.sde.marginal_prob(x, t)[1]

        # Use the step size scheduler if provided, otherwise use the default calculation
        if self.step_size_scheduler is not None:
            step_size = self.step_size_scheduler(t, std, self.sde, self.rsde)
        else:
            # Default step size calculation (for backward compatibility)
            step_size = default_step_size

        # Use the temperature scheduler if provided, otherwise use the default calculation
        if self.temp_scheduler is not None:
            # jax.debug.print('temp scheduler')
            temp = self.temp_scheduler(t, std, self.sde, self.rsde, step_size=step_size)
        else:
            # Default temperature calculation (for backward compatibility)
            temp = (self.temperature_coeff * (std - 0.1)) + 1.0

        # jax.debug.print("{}, {}", t[0], step_size[0]/temp[0])

        # Broadcast temperature to match input shape, preserving batch dimension
        batch_size = temp.shape[0]
        broadcast_shape = (batch_size,) + (1,) * (x.ndim - 1)
        temp = jnp.reshape(temp, broadcast_shape)

        sigma_min = self.sde.marginal_prob(x, t=jnp.zeros_like(t))[1]

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = self.score_fn(x, t=t)
            grad_norm = jnp.linalg.norm(
                grad.reshape((grad.shape[0], -1)), axis=-1
            ).mean()
            # jax.debug.print("{}, {}", grad_norm, t[0])
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)

            if self.heuristic_cond_sampling and self.cond_indices is not None:
                # clamp noise and score to zero for the conditioned indices
                grad = jnp.where(self.cond_indices, 0.0, grad)
                noise = jnp.where(self.cond_indices, 0.0, noise)

            if denoise_step:
                x_mean = x + batch_mul(sigma_min**2, grad)  # denoising step
            else:
                # jax.debug.print("{}, {}, {}", step_size, temp, grad_norm)
                x_mean = x + batch_mul(step_size, grad / temp)
                # x_mean = x + batch_mul(step_size, grad)

            # x_mean = x + batch_mul(sigma_min**2, grad)
            # jax.debug.print("{}, {}, {}, {}", grad_norm / temp[0], step_size[0], temp[0], grad_norm / temp[0] * step_size[0])
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            # jax.debug.print("{}", x)
            # x = x_mean + batch_mul(noise, jnp.sqrt(step_size * temp * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_predictor(name="disco_ddim")
class DiscoDDIMPredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )

    def update_fn(self, rng, x, t):
        sigma_fixed = self.sde.discrete_sigmas[0]

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = self.score_fn(x, t=jnp.ones_like(t))

            d = sigma_fixed**2 * grad
            x_mean = x + d
            x = x_mean - 0.5 * d
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_predictor(name="disco_temp")
class DiscoTempPredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )

    def update_fn(self, rng, x, t):
        sigma_fixed = self.sde.discrete_sigmas[0]

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = self.score_fn(x, t=jnp.ones_like(t))

            d = sigma_fixed**2 * grad
            x_mean = x + d
            x = x_mean - 0.5 * d
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_predictor(name="ancestral_sampling")
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
        heuristic_cond_sampling=False,
    ):
        super().__init__(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, rng, x, t, denoise_step=False):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = jnp.where(
            timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1]
        )

        def loop_body(step, val):
            rng, x, x_mean = val
            score = self.score_fn(x, t)
            x_mean = x + batch_mul(score, sigma**2 - adjacent_sigma**2)
            std = jnp.sqrt(
                (adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2)
            )
            rng, step_rng = random.split(rng)
            noise = random.normal(step_rng, x.shape)
            x = x_mean + batch_mul(std, noise)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, loop_body, (rng, x, x))
        return x, x_mean

    def vpsde_update_fn(self, rng, x, t, denoise_step=False):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
        beta = sde.discrete_betas[timestep]

        def loop_body(step, val):
            rng, x, x_mean = val
            score = self.score_fn(x, t)
            x_mean = batch_mul((x + batch_mul(beta, score)), 1.0 / jnp.sqrt(1.0 - beta))
            rng, step_rng = random.split(rng)
            noise = random.normal(step_rng, x.shape)
            x = x_mean + batch_mul(jnp.sqrt(beta), noise)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, self.n_steps, loop_body, (rng, x, x))
        return x, x_mean

    def update_fn(self, rng, x, t, denoise_step=False):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(rng, x, t, denoise_step)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(rng, x, t, denoise_step)


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
        step_size_scheduler=None,
        n_steps=1,
    ):
        # Initialize minimal required attributes
        self.sde = sde
        self.score_fn = score_fn
        self.rsde = (
            sde.reverse(score_fn, probability_flow)
            if sde is not None and score_fn is not None
            else None
        )
        self.cond_indices = cond_indices
        self.temperature_coeff = temperature_coeff
        self.temp_scheduler = temp_scheduler
        self.step_size_scheduler = step_size_scheduler
        self.n_steps = n_steps

    def update_fn(self, rng, x, t):
        return x, x


@register_corrector(name="langevin_temp")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, rng, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        def loop_body(step, val):
            rng, x, x_mean = val

            std = self.sde.marginal_prob(x, t)[1]

            # Use the temperature scheduler if provided, otherwise use the default calculation
            if self.temp_scheduler is not None:
                temp = self.temp_scheduler(t, std, self.sde, self.rsde)
            else:
                # Default temperature calculation (for backward compatibility)
                temp = (self.temperature_coeff * (std - 0.01)) + 1.0

            # jax.debug.print("{}, {}", t, temp)

            grad = self.score_fn(x, t=t)
            grad = grad / temp.reshape((temp.shape[0], 1, 1, 1))

            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            grad_norm = jnp.linalg.norm(
                grad.reshape((grad.shape[0], -1)), axis=-1
            ).mean()
            grad_norm = jax.lax.pmean(grad_norm, axis_name="batch")
            noise_norm = jnp.linalg.norm(
                noise.reshape((noise.shape[0], -1)), axis=-1
            ).mean()
            noise_norm = jax.lax.pmean(noise_norm, axis_name="batch")
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            if self.cond_indices is not None:
                # clamp noise and score to zero for the conditioned indices
                grad = jnp.where(self.cond_indices, 0.0, grad)
                noise = jnp.where(self.cond_indices, 0.0, noise)

            # jax.debug.print("{}, {}", t, jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean())

            x_mean = x + batch_mul(step_size, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, rng, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        def loop_body(step, val):
            rng, x, x_mean = val
            # grad = score_fn(x, t)
            grad = score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            grad_norm = jnp.linalg.norm(
                grad.reshape((grad.shape[0], -1)), axis=-1
            ).mean()
            grad_norm = jax.lax.pmean(grad_norm, axis_name="batch")
            noise_norm = jnp.linalg.norm(
                noise.reshape((noise.shape[0], -1)), axis=-1
            ).mean()
            noise_norm = jax.lax.pmean(noise_norm, axis_name="batch")
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + batch_mul(step_size, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector(name="uld_temp_anneal")
class UnadjustedLangevinDynamicsWithTemparatureAnnealing(Corrector):
    """Unadjusted Langevin dynamics predictor with temperature annealing. Used for DISCO."""

    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, rng, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]
        # Use the temperature scheduler if provided, otherwise use the default calculation
        if self.temp_scheduler is not None:
            temparature = self.temp_scheduler(t, std, self.sde, self.rsde)
        else:
            # Default temperature calculation (for backward compatibility)
            temparature = 1.0 / std

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            _, G = self.sde.discretize(x, t)
            step_size = G**2 * target_snr

            if self.cond_indices is not None:
                # clamp noise and score to zero for the conditioned indices
                grad = jnp.where(self.cond_indices, 0.0, grad)
                noise = jnp.where(self.cond_indices, 0.0, noise)

            # we only scale the score by 1/temparature, not the noise
            x_mean = x + batch_mul(step_size / temparature, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector(name="uld")
class UnadjustedLangevinDynamics(Corrector):
    """Unadjusted Langevin dynamics predictor. Used for DISCO."""

    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, rng, x, t):
        score_fn = self.score_fn
        n_steps = self.n_steps
        # assert n_steps == 1, "n_steps must be 1 for ULD"
        target_snr = self.snr

        def loop_body(step, val):
            rng, x, x_mean = val
            # grad = score_fn(x, t=jnp.ones_like(t)) # We don't even pass time here to make sure we really don't use it in the DISCO case.
            grad = score_fn(x, t=t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            step_size = t * target_snr
            # _, G = self.sde.discretize(x, t)
            # step_size = G**2 * target_snr

            # grad_norm = jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1)

            # std = self.sde.marginal_prob(x, t)[1]
            # grad = grad / grad_norm.reshape([-1] + [1] * (len(grad.shape) - 1)) # normalize the gradient
            # grad = grad * (std.reshape([-1] + [1] * (len(grad.shape) - 1)) / 0.01**2) # * 18.
            # # grad = grad / std.reshape([-1] + [1] * (len(grad.shape) - 1))
            # jax.debug.print("{}, {}", t, jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean())

            # step_size = t * target_snr

            # step_size = (1-t + 1e-5) * target_snr # debug, going from small step_size to large step_size!
            # step_size = (1-t + 1e-5) * target_snr / std # debug, going from small step_size to large step_size!
            # jax.debug.print("{}, {}", t, step_size)

            if self.cond_indices is not None:
                # clamp noise and score to zero for the conditioned indices
                grad = jnp.where(self.cond_indices, 0.0, grad)
                noise = jnp.where(self.cond_indices, 0.0, noise)

            std = self.sde.marginal_prob(x, t)[1]

            # Use the temperature scheduler if provided, otherwise use the default calculation
            if self.temp_scheduler is not None:
                temp = self.temp_scheduler(t, std, self.sde, self.rsde)
            else:
                # Default temperature calculation (for backward compatibility)
                temp = self.temperature_coeff * std

            # jax.debug.print("{}, {}", t, temp)
            temp = jnp.reshape(temp, (temp.shape[0], 1, 1, 1))
            x_mean = x + batch_mul(step_size, grad / temp)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        super().__init__(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, rng, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]
        jax.debug.print("{}, {}", t[0], std[0])

        def loop_body(step, val):
            rng, x, x_mean, grad_norm = val
            grad = score_fn(x, t)
            grad_norm = jnp.linalg.norm(
                grad.reshape((grad.shape[0], -1)), axis=-1
            ).mean()
            # jax.debug.print("{}, {}", t, grad_norm)

            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            # step_size = t * target_snr

            if self.cond_indices is not None:
                # clamp noise and score to zero for the conditioned indices
                grad = jnp.where(self.cond_indices, 0.0, grad)
                noise = jnp.where(self.cond_indices, 0.0, noise)

            x_mean = x + batch_mul(step_size, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean, grad_norm

        grad_norm = 0.0
        _, x, x_mean, grad_norm = jax.lax.fori_loop(
            0, n_steps, loop_body, (rng, x, x, grad_norm)
        )
        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        sde,
        score_fn,
        snr,
        n_steps,
        cond_indices=None,
        temperature_coeff=1.0,
        temp_scheduler=None,
    ):
        # Initialize minimal required attributes
        self.sde = sde
        self.score_fn = score_fn
        self.rsde = (
            sde.reverse(score_fn, False)
            if sde is not None and score_fn is not None
            else None
        )
        self.snr = snr
        self.n_steps = n_steps
        self.cond_indices = cond_indices
        self.temperature_coeff = temperature_coeff
        self.temp_scheduler = temp_scheduler

    def update_fn(self, rng, x, t):
        return x, x


def shared_predictor_update_fn(
    rng,
    state,
    x,
    t,
    sde,
    model,
    predictor,
    probability_flow,
    continuous,
    fixed_sigma=None,
    cond_indices=None,
    cond_values=None,
    fixed_sigma_baseline=None,
    temperature_coeff=1.0,
    temp_scheduler=None,
    step_size_scheduler=None,
    n_steps=1,
    denoise_step=False,
    guidance_alpha=0.0,
    heuristic_cond_sampling=False,
):
    """A wrapper that configures and returns the update function of predictors."""
    guidance_cond_values = cond_values if not heuristic_cond_sampling else None
    guidance_cond_indices = cond_indices if not heuristic_cond_sampling else None

    score_fn = mutils.get_score_fn(
        sde,
        model,
        state.params_ema,
        state.model_state,
        train=False,
        continuous=continuous,
        fixed_sigma=fixed_sigma,
        fixed_sigma_baseline=fixed_sigma_baseline,
        guidance_alpha=guidance_alpha,
        guidance_cond_values=guidance_cond_values,
        guidance_cond_indices=guidance_cond_indices,
    )

    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
        )
    else:
        predictor_obj = predictor(
            sde,
            score_fn,
            probability_flow,
            cond_indices,
            temperature_coeff,
            temp_scheduler,
            step_size_scheduler,
            n_steps,
            heuristic_cond_sampling,
        )
    return predictor_obj.update_fn(rng, x, t, denoise_step)


def shared_corrector_update_fn(
    rng,
    state,
    x,
    t,
    sde,
    model,
    corrector,
    continuous,
    snr,
    n_steps,
    fixed_sigma=None,
    cond_indices=None,
    cond_values=None,
    fixed_sigma_baseline=None,
    store_intermediate_samples=False,
    temperature_coeff=1.0,
    temp_scheduler=None,
    store_grad_norm=False,
    guidance_alpha=0.0,
    heuristic_cond_sampling=False,
):
    """A wrapper tha configures and returns the update function of correctors."""
    guidance_cond_values = cond_values if not heuristic_cond_sampling else None
    guidance_cond_indices = cond_indices if not heuristic_cond_sampling else None

    score_fn = mutils.get_score_fn(
        sde,
        model,
        state.params_ema,
        state.model_state,
        train=False,
        continuous=continuous,
        fixed_sigma=fixed_sigma,
        fixed_sigma_baseline=fixed_sigma_baseline,
        guidance_alpha=guidance_alpha,
        guidance_cond_values=guidance_cond_values,
        guidance_cond_indices=guidance_cond_indices,
    )

    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )
    else:
        corrector_obj = corrector(
            sde, score_fn, snr, n_steps, cond_indices, temperature_coeff, temp_scheduler
        )

    if store_grad_norm:
        grad = score_fn(x, t)
        grad_norm = jnp.sum(grad.reshape((grad.shape[0], -1)) ** 2, axis=-1).mean()
    else:
        grad_norm = 0.0

    return corrector_obj.update_fn(rng, x, t), grad_norm


def get_pc_sampler(
    sde,
    model,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    fixed_sigma=None,
    cond_indices=None,
    cond_values=None,
    heuristic_cond_sampling=False,
    fixed_sigma_baseline=None,
    store_intermediate_samples=False,
    temperature_coeff=1.0,
    temp_scheduler=None,
    step_size_scheduler=None,
    store_grad_norm=False,
    guidance_alpha=0.0,
    inpaint=False,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      fixed_sigma: If not None, we use DISCO and the score is scaled by 1/sigma^2.
      cond_indices: If not None, it's a boolean jnp.array of the same shape as x. We condition on the RVs where cond_indices is True.
      cond_values: If not None, it's a jnp.array of the same shape as x. We condition on the values in cond_values where cond_indices is True.

    Returns:
      A sampling function that takes random states, and a replcated training state and returns samples as well as
      the number of function evaluations during sampling.
    """
    if heuristic_cond_sampling:
        assert (
            cond_values is not None and cond_indices is not None
        ), "cond_values and cond_indices must be provided if heuristic_cond_sampling is True"
        assert isinstance(
            sde, sde_lib.VESDE
        ), "Heuristic conditional sampling is only supported for VESDE"

        if cond_indices is not None:
            assert (
                cond_indices.shape == shape
            ), f"cond_indices must be None or have the same shape as x: cond_indices.shape={cond_indices.shape}, shape={shape}"
            assert (
                cond_values.shape == shape
            ), f"cond_values must be None or have the same shape as x. We'll slice out the conditioned RVs from cond_values: cond_values.shape={cond_values.shape}, shape={shape}"

    assert (
        guidance_alpha == 0.0 or not heuristic_cond_sampling
    ), "heuristic_cond_sampling and guidance_alpha > 0 are not compatible"

    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        model=model,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
        fixed_sigma=fixed_sigma,
        cond_indices=cond_indices,
        cond_values=cond_values,
        fixed_sigma_baseline=fixed_sigma_baseline,
        temperature_coeff=temperature_coeff,
        temp_scheduler=temp_scheduler,
        step_size_scheduler=step_size_scheduler,
        n_steps=n_steps,
        guidance_alpha=guidance_alpha,
        heuristic_cond_sampling=heuristic_cond_sampling,
    )

    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        model=model,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
        fixed_sigma=fixed_sigma,
        cond_indices=cond_indices,
        cond_values=cond_values,
        fixed_sigma_baseline=fixed_sigma_baseline,
        store_intermediate_samples=store_intermediate_samples,
        temperature_coeff=temperature_coeff,
        temp_scheduler=temp_scheduler,
        store_grad_norm=store_grad_norm,
        guidance_alpha=guidance_alpha,
        heuristic_cond_sampling=heuristic_cond_sampling,
    )

    def pc_sampler(rng, state):
        """The PC sampler funciton.

        Args:
          rng: A JAX random state
          state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
        Returns:
          Samples, number of function evaluations
        """
        # Initial sample
        is_disco = fixed_sigma is not None

        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)
        # jax.debug.print("{}", x)
        timesteps = jnp.linspace(sde.T, eps, sde.N)

        if inpaint:
            assert (
                cond_indices is not None and cond_values is not None
            ), "cond_indices and cond_values must be provided if inpaint is True"
            x = jnp.where(cond_indices, cond_values, x)
        else:
            if (
                cond_indices is not None
                and (fixed_sigma is not None)
                and not heuristic_cond_sampling
            ):
                x = jnp.where(cond_indices, cond_values, x)

        # @loop_tqdm(sde.N)
        def loop_body(i, val):
            rng, x, x_mean, xs, grad_norms = val
            # jax.debug.print("{}, {}", i, x)
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)

            if inpaint:
                x = jnp.where(
                    cond_indices, cond_values, x
                )  # we use this only for inpainting
            elif heuristic_cond_sampling:
                std = sde.marginal_prob(x, t)[1]
                z = jax.random.normal(step_rng, shape) * std
                # replace entries in x with conditioned values + noise (single sample MC)
                x = jnp.where(cond_indices, cond_values + z, x)

            # jax.debug.print("before corr: {}, {}", i, x)
            (x, x_mean), grad_norm = corrector_update_fn(step_rng, state, x, vec_t)
            # jax.debug.print("after corr: {}, {}", i, x)
            rng, step_rng = random.split(rng)
            x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
            # jax.debug.print("after pred: {}, {}", i, x)
            # jax.debug.print("{}, {}", i, x.shape)
            if store_intermediate_samples:
                # jax.debug.print('storing sample {}', i)
                xs = xs.at[i + 1].set(x)
                grad_norms = grad_norms.at[i + 1].set(grad_norm)
            return rng, x, x_mean, xs, grad_norms

        if store_intermediate_samples:
            xs = jnp.zeros((sde.N + 1,) + shape)
            # add the initial sample to the list of samples
            xs = xs.at[0].set(x)
            grad_norms = jnp.zeros((sde.N + 1,))
            grad_norms = grad_norms.at[0].set(
                jnp.linalg.norm(x.reshape((x.shape[0], -1)), axis=-1).mean()
            )
        else:
            xs = None
            grad_norms = None

        _, x, x_mean, xs, grad_norms = jax.lax.fori_loop(
            0,
            sde.N if not is_disco else sde.N - 1,
            loop_body,
            (rng, x, x, xs, grad_norms),
        )
        # Denoising is equivalent to running one predictor step without adding noise.
        if denoise:
            if is_disco:
                vec_t = jnp.ones(shape[0]) * sde.T
                x, x_mean = predictor_update_fn(
                    step_rng, state, x, vec_t, denoise_step=True
                )

            if store_intermediate_samples:
                xs = xs.at[-1].set(
                    x_mean
                )  # overwerite last noisy sample with the denoised sample

        return (
            inverse_scaler(x_mean if denoise else x),
            sde.N * (n_steps + 1),
            inverse_scaler(xs),
            grad_norms,
        )

    return jax.pmap(pc_sampler, axis_name="batch")


def get_ode_sampler(
    sde,
    model,
    shape,
    inverse_scaler,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-3,
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

    Returns:
      A sampling function that takes random states, and a replicated training state and returns samples
      as well as the number of function evaluations during sampling.
    """

    @jax.pmap
    def denoise_update_fn(rng, state, x):
        score_fn = get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=True,
        )
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = jnp.ones((x.shape[0],)) * eps
        _, x = predictor_obj.update_fn(rng, x, vec_eps)
        return x

    @jax.pmap
    def drift_fn(state, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=True,
        )
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(prng, pstate, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          prng: An array of random state. The leading dimension equals the number of devices.
          pstate: Replicated training state for running on multiple devices.
          z: If present, generate samples from latent code `z`.
        Returns:
          Samples, and the number of function evaluations.
        """
        # Initial sample
        rng = flax.jax_utils.unreplicate(prng)
        rng, step_rng = random.split(rng)
        if z is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
        else:
            x = z

        def ode_func(t, x):
            x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
            vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
            drift = drift_fn(pstate, x, vec_t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(
            ode_func,
            (sde.T, eps),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method=method,
        )
        nfe = solution.nfev
        x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

        # Denoising is equivalent to running one predictor step without adding noise
        if denoise:
            rng, *step_rng = random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)
            x = denoise_update_fn(step_rng, pstate, x)

        x = inverse_scaler(x)
        return x, nfe

    return ode_sampler


def get_tds_sampler(
    sde,
    model,
    shape,
    continuous,
    cond_indices=None,
    cond_values=None,
    inverse_scaler=None,
    sampling_eps=1e-3,
    store_intermediate_samples=False,
    n_steps_each=1,
):
    if inverse_scaler is None:
        inverse_scaler = lambda x: x

    num_particles = shape[0]

    def tds_sampler(rng, state):
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)
        timesteps = jnp.linspace(sde.T, sampling_eps, sde.N)
        sigmas = sde.discrete_sigmas[::-1]
        sigmas_with_0 = jnp.concatenate([sigmas, jnp.array([0])])
        var_diffs = sigmas_with_0[:-1] ** 2 - sigmas_with_0[1:] ** 2

        score_fn = mutils.get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=continuous,
        )

        # we'll use graident guidance with the std of the SDE
        guided_score_fn = mutils.get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=continuous,
            guidance_alpha=0.0,
            guidance_cond_indices=cond_indices,
            guidance_cond_values=cond_values,
        )

        def log_twisting_fn(x_noisy, t):
            # x_noisy, t are a single objects, not a batch
            # Computes Eq. 13 in (Wu et al., 2023):
            # \tilde{p}_{\theta}(y | x^t) = N(y; denoise(x, t)_M, std_t^2)
            # where y is `cond_values` and M is the mask over observed coordinates

            sigma = sde.marginal_prob(x_noisy, t)[1]
            score_x = score_fn(x_noisy[None, ...], t[None, ...]).squeeze()
            denoised_x = x_noisy + sigma**2 * score_x  # tweedie
            denoised_cond_values = jnp.where(cond_indices, denoised_x, 0.0)
            cond_values_array = jnp.where(cond_indices, cond_values, 0.0)
            diff = cond_values_array - denoised_cond_values
            # diff_l2 = jnp.sum((cond_values_array - denoised_cond_values)**2)
            # log_gauss_unnorm = -diff_l2 / (2 * sigma**2)
            dim = jnp.sum(cond_indices)
            # jax.debug.print('dim: {}', dim)
            log_prob = -0.5 * jnp.sum(diff**2) / sigma**2 - 0.5 * dim * jnp.log(
                2 * jnp.pi * sigma**2
            )
            return log_prob

        twisting_weight_fn = jax.vmap(log_twisting_fn)

        def log_transition_density_fn(x_new, x_old, t, sigma, cond_y=False):
            sc_fn = score_fn if not cond_y else guided_score_fn
            mean = x_old + sigma**2 * sc_fn(x_old[None, ...], t[None, ...]).squeeze()
            # Evaluate multivariate Gaussian PDF with covariance sigma^2 * I
            diff = x_new - mean
            log_prob = -0.5 * jnp.sum(
                diff**2
            ) / sigma**2 - 0.5 * x_new.size * jnp.log(2 * jnp.pi * sigma**2)
            return log_prob

        log_cond_transition_density_fn = jax.vmap(
            lambda x_new, x_old, t, sigma: log_transition_density_fn(
                x_new, x_old, t, sigma, cond_y=True
            )
        )
        log_uncond_transition_density_fn = jax.vmap(
            lambda x_new, x_old, t, sigma: log_transition_density_fn(
                x_new, x_old, t, sigma, cond_y=False
            )
        )

        def loop_body(i, val):
            rng, x, xs, weights, old_p = val
            # jax.debug.print("{}, {}", i, x)
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t

            def inner_loop_body(j, val):
                # Resample
                rng, x, weights, old_p = val
                rng, step_rng = random.split(rng)
                # jax.debug.print('weights: {}', weights.max())
                x_idxs = blackjax.smc.resampling.systematic(
                    step_rng, weights, num_particles
                )
                # x_idxs = blackjax.smc.resampling.multinomial(step_rng, weights, num_particles)
                x = x[x_idxs]

                # Conditional Score Approximation
                guided_score = guided_score_fn(x, vec_t)
                # Proposal: single langevin step
                sigma_tilde_sq = var_diffs[i]
                # sigma_tilde_sq = sigmas_with_0[i]**2
                sigma_i, sigma_ip1 = sigmas_with_0[i], sigmas_with_0[i + 1]
                # var = (sigma_ip1**2 / sigma_i**2) * sigma_tilde_sq
                var = sigma_tilde_sq
                # jax.debug.print('i, sigma_tilde_sq, var: {}, {}, {}', i, sigma_tilde_sq, var)
                rng, step_rng = random.split(rng)
                new_x = (
                    x
                    + sigma_tilde_sq * guided_score
                    + jnp.sqrt(var) * jax.random.normal(step_rng, x.shape)
                )

                # Update weights
                new_p = twisting_weight_fn(new_x, vec_t)
                sigmas = jnp.repeat(
                    jnp.sqrt(sigma_tilde_sq), num_particles
                )  # same sigma for all particles
                uncond_transition_densities = log_uncond_transition_density_fn(
                    new_x, x, vec_t, sigmas
                )
                cond_transition_densities = log_cond_transition_density_fn(
                    new_x, x, vec_t, sigmas
                )
                # jax.debug.print('shapes: {}, {}, {}, {}', uncond_transition_densities.shape, cond_transition_densities.shape, new_p.shape, old_p.shape)
                # jax.debug.print('{}', uncond_transition_densities.shape)
                numerator = uncond_transition_densities + new_p
                denom = cond_transition_densities + old_p
                weights = numerator - denom
                # jax.debug.print("weights: {}", weights)
                weights = jax.nn.softmax(weights)

                x = new_x
                return rng, x, weights, new_p

            rng, x, weights, new_p = jax.lax.fori_loop(
                0, n_steps_each, inner_loop_body, (rng, x, weights, old_p)
            )

            if store_intermediate_samples:
                xs = xs.at[i + 1].set(x)
            return rng, x, xs, weights, new_p

        # init weights
        logp = twisting_weight_fn(x, sde.T * jnp.ones(shape[0]))
        weights = jax.nn.softmax(logp)
        assert weights.shape == (
            num_particles,
        ), f"weights.shape: {weights.shape}, num_particles: {num_particles}"

        if store_intermediate_samples:
            xs = jnp.zeros((sde.N + 1,) + shape)
            # add the initial sample to the list of samples
            xs = xs.at[0].set(x)
        else:
            xs = None

        _, x, xs, _, _ = jax.lax.fori_loop(
            0, sde.N, loop_body, (rng, x, xs, weights, logp)
        )
        return inverse_scaler(x), sde.N, inverse_scaler(xs), None

    return jax.pmap(tds_sampler, axis_name="batch")


def get_edm_sampler(
    sde,
    model,
    shape,
    continuous,
    inverse_scaler=None,
    fixed_sigma=None,
    fixed_sigma_baseline=None,
    rho=7,
    num_steps=18,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    store_intermediate_samples=False,
    sigma_min_global=None,
    sigma_max_global=None,
    is_uedm=False,
):
    if inverse_scaler is None:
        inverse_scaler = lambda x: x

    if is_uedm:
        inverse_scaler = lambda x: (x + 1) / 2.0

    def round_sigmas(sigmas, is_disco=False):
        """Round noise levels to the nearest value in the discrete schedule.

        Args:
            sigmas: An array of noise levels (standard deviations).

        Returns:
            Indices of the closest discrete sigma values.
        """
        if is_disco or is_uedm:
            return sigmas, jnp.arange(len(sigmas.reshape(-1)))

        discrete_sigmas = sde.discrete_sigmas[::-1]
        # Reshape sigmas to ensure proper broadcasting
        sigmas = jnp.asarray(sigmas).reshape(-1, 1)
        # Calculate distances between each sigma and all discrete_sigmas
        distances = jnp.abs(sigmas - discrete_sigmas.reshape((1, -1)))
        # Find indices of minimum distances
        indices = jnp.argmin(distances, axis=1)
        rounded_sigmas = discrete_sigmas[indices]
        return rounded_sigmas, indices.astype(jnp.int32)

    def edm_sampler(rng, state):
        score_fn = mutils.get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=continuous,
            fixed_sigma=fixed_sigma,
            fixed_sigma_baseline=fixed_sigma_baseline,
            is_uedm=is_uedm,
        )

        if is_uedm:
            sigma_min = 0.002
            sigma_max = 80
            rng, step_rng = random.split(rng)
            x_next = random.normal(step_rng, shape) * sigma_max
        else:
            sigma_min = sde.sigma_min if sigma_min_global is None else sigma_min_global
            sigma_max = sde.sigma_max if sigma_max_global is None else sigma_max_global
            rng, step_rng = random.split(rng)
            x_next = sde.prior_sampling(
                step_rng, shape, sigma_max_global
            )  # .astype(jnp.float64)

        is_disco = fixed_sigma is not None
        # Time step discretization.
        step_indices = jnp.arange(num_steps)  # , dtype=jnp.float64)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = jnp.concatenate(
            [round_sigmas(t_steps)[0], jnp.zeros_like(t_steps[:1])]
        )  # t_N = 0
        # jax.debug.print("t_steps: {}", t_steps)
        t_indices = jnp.concatenate(
            [round_sigmas(t_steps)[1], jnp.zeros_like(t_steps[:1]) + sde.N - 1]
        ).astype(jnp.int32)

        def loop_body(i, val):
            rng, x_next, xs = val

            t_cur, t_next = t_steps[i], t_steps[i + 1]
            t_cur_index, t_next_index = t_indices[i], t_indices[i + 1]
            x_cur = x_next

            # Increase noise temporarily.
            gamma = jnp.where(
                jnp.logical_and(S_min <= t_cur, t_cur <= S_max),
                jnp.minimum(S_churn / num_steps, jnp.sqrt(2) - 1),
                0.0,
            )
            t_hat, t_hat_index = round_sigmas(t_cur + gamma * t_cur)
            rng, step_rng = random.split(rng)
            x_hat = x_cur + jnp.sqrt(
                t_hat**2 - t_cur**2
            ) * S_noise * jax.random.normal(step_rng, shape)

            # Euler step.
            if is_uedm:
                score = score_fn(
                    x_hat, t_hat.repeat(x_hat.shape[0], axis=0), label_input=True
                )  # pass in t directly for uEDM
            else:
                score = score_fn(
                    x_hat, t_hat_index.repeat(x_hat.shape[0], axis=0), label_input=True
                )  # .astype(jnp.float64)

            if not is_disco:
                denoised = (
                    x_hat + t_hat**2 * score
                )  # we want the posterior mean here, not the score
                d_cur = (x_hat - denoised) / t_hat
            else:
                temp = 1.0 * t_hat  # jnp.clip(t_hat, 1.0, None)
                d_cur = (fixed_sigma**2 / temp) * -score  # scaled negative score
                # d_cur = jnp.where(i < num_steps - 5, d_cur, 1.2 * d_cur)
                # denoised = x_hat + fixed_sigma**2 * score
                # d_cur = (x_hat - denoised) / t_hat # fixed_sigma

            # jax.debug.print("norm of d_cur: {}", jnp.linalg.norm(d_cur))
            x_next_euler = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if is_uedm:
                score_prime = score_fn(
                    x_next_euler,
                    t_next.repeat(x_next_euler.shape[0], axis=0),
                    label_input=True,
                )  # .astype(jnp.float64)
            else:
                score_prime = score_fn(
                    x_next_euler,
                    t_next_index.repeat(x_next_euler.shape[0], axis=0),
                    label_input=True,
                )  # .astype(jnp.float64)

            if not is_disco:
                denoised_prime = x_next_euler + t_next**2 * score_prime
                d_prime = (x_next_euler - denoised_prime) / t_next
            else:
                temp = 1.0 * t_next  # jnp.clip(2 * t_next, 1.0, None)
                d_prime = (
                    fixed_sigma**2 / temp
                ) * -score_prime  # scaled negative score
                # d_prime = jnp.where(i < num_steps - 5, d_prime, 1.2 * d_prime)
                # denoised_prime = x_next_euler + fixed_sigma**2 * score_prime
                # d_prime = (x_next_euler - denoised_prime) / t_next # fixed_sigma

            x_next_corrected = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Only use correction for steps before the last one
            x_next = jnp.where(i < num_steps - 1, x_next_corrected, x_next_euler)

            # x_next = x_next_euler # debug
            if store_intermediate_samples:
                xs = xs.at[i + 1].set(inverse_scaler(x_next))

            return rng, x_next, xs

        if store_intermediate_samples:
            xs = jnp.zeros((num_steps + 1, *shape))
            xs = xs.at[0].set(x_next)
        else:
            xs = None

        rng, x_next, xs = jax.lax.fori_loop(0, num_steps, loop_body, (rng, x_next, xs))
        return inverse_scaler(x_next), 2 * num_steps - 1, xs, None

    return jax.pmap(edm_sampler, axis_name="batch")


def get_posterior_sampler(
    sde,
    model,
    shape,
    continuous,
    inverse_scaler=None,
    fixed_sigma=None,
    fixed_sigma_baseline=None,
    rho=7,
    num_steps=18,
    S_noise=1,
    denoise=True,
    inv_temp_scaler=1.0,
    store_intermediate_samples=False,
    cond_indices=None,
    cond_values=None,
):
    if inverse_scaler is None:
        inverse_scaler = lambda x: x

    def round_sigmas(sigmas, is_disco=True):
        """Round noise levels to the nearest value in the discrete schedule.

        Args:
            sigmas: An array of noise levels (standard deviations).

        Returns:
            Indices of the closest discrete sigma values.
        """
        if is_disco:
            return sigmas, jnp.arange(len(sigmas))

        discrete_sigmas = sde.discrete_sigmas[::-1]
        # Reshape sigmas to ensure proper broadcasting
        sigmas = jnp.asarray(sigmas).reshape(-1, 1)
        # Calculate distances between each sigma and all discrete_sigmas
        distances = jnp.abs(sigmas - discrete_sigmas.reshape((1, -1)))
        # Find indices of minimum distances
        indices = jnp.argmin(distances, axis=1)
        rounded_sigmas = discrete_sigmas[indices]
        return rounded_sigmas, indices.astype(jnp.int32)

    def posterior_sampler(rng, state):
        score_fn = mutils.get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=continuous,
            fixed_sigma=fixed_sigma,
            fixed_sigma_baseline=fixed_sigma_baseline,
        )

        sigma_min = sde.sigma_min
        sigma_max = sde.sigma_max

        rng, step_rng = random.split(rng)
        x_next = sde.prior_sampling(step_rng, shape)  # .astype(jnp.float64)

        if cond_indices is not None:
            # x_next = jnp.where(cond_indices, cond_values, x_next)
            x_next = (
                jnp.where(cond_indices, cond_values, 0) + x_next
            )  # add noise also on condition
            # jax.debug.print("x_next: {}", x_next[0, 0, 0, 0])

        # Time step discretization.
        step_indices = jnp.arange(num_steps)  # , dtype=jnp.float64)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = jnp.concatenate(
            [round_sigmas(t_steps)[0], jnp.zeros_like(t_steps[:1])]
        )  # t_N = 0
        # jax.debug.print("t_steps: {}", t_steps)
        t_indices = jnp.concatenate(
            [round_sigmas(t_steps)[1], jnp.zeros_like(t_steps[:1]) + sde.N - 1]
        ).astype(jnp.int32)

        def loop_body(i, val):
            rng, x_next, xs = val

            t_cur, t_next = t_steps[i], t_steps[i + 1]
            t_cur_index, t_next_index = t_indices[i], t_indices[i + 1]
            x_cur = x_next
            # jax.debug.print("x_next: {}, {}", i, x_next[0, 0, 0, 0])

            score = score_fn(
                x_cur, t_cur_index.repeat(x_cur.shape[0], axis=0), label_input=True
            )
            if cond_indices is not None:
                score = jnp.where(
                    cond_indices, 0.0, score
                )  # clamp score to 0 for conditioning variables

            rng, step_rng = random.split(rng)
            inv_temp = inv_temp_scaler * (fixed_sigma / t_cur)
            alpha_t = jnp.where(
                i < num_steps - 1, (t_cur - t_next), fixed_sigma
            )  # alpha_t = fixed_sigma for last step, to just denoise
            # alpha_t = fixed_sigma
            step_size = alpha_t * fixed_sigma

            x_without_noise = x_cur + inv_temp * step_size * score

            noise = jax.random.normal(step_rng, shape)
            if cond_indices is not None:
                noise = jnp.where(cond_indices, 0.0, noise)
            x_next = x_without_noise + jnp.sqrt(2 * step_size) * S_noise * noise

            # Only use correction for steps before the last one
            x_next = jnp.where(
                i < num_steps - 1, x_next, x_without_noise
            )  # denoise last step

            # x_next = jnp.where(cond_indices, cond_values, 0) + x_next # add noise also on condition

            if store_intermediate_samples:
                xs = xs.at[i + 1].set(x_next)

            return rng, x_next, xs

        if store_intermediate_samples:
            xs = jnp.zeros((num_steps + 1, *shape))
            xs = xs.at[0].set(x_next)
        else:
            xs = None

        rng, x_next, xs = jax.lax.fori_loop(0, num_steps, loop_body, (rng, x_next, xs))
        return inverse_scaler(x_next), num_steps, xs, None

    return jax.pmap(posterior_sampler, axis_name="batch")

    # def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like,
    #     num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    #     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    # ):
    #     # Adjust noise levels based on what's supported by the network.
    #     sigma_min = max(sigma_min, net.sigma_min)
    #     sigma_max = min(sigma_max, net.sigma_max)

    #     # Time step discretization.
    #     step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    #     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    #     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    #     # Main sampling loop.
    #     x_next = latents.to(torch.float64) * t_steps[0]
    #     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
    #         x_cur = x_next

    #         # Increase noise temporarily.
    #         gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    #         t_hat = net.round_sigma(t_cur + gamma * t_cur)
    #         x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    #         # Euler step.
    #         denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    #         d_cur = (x_hat - denoised) / t_hat
    #         x_next = x_hat + (t_next - t_hat) * d_cur

    #         # Apply 2nd order correction.
    #         if i < num_steps - 1:
    #             denoised = net(x_next, t_next, class_labels).to(torch.float64)
    #             d_prime = (x_next - denoised) / t_next
    #             x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    #     return x_next
