# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

# ----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).


def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    conditioning_images=None,
    replacement_heuristic=False,
    return_intermediate=False,
    conditioning_mask=None,
    **sampler_kwargs,
):
    if replacement_heuristic:
        assert (
            conditioning_images is not None
        ), "Replacement heuristic requires conditioning images"

    # Check if conditioning_mask is provided, otherwise create a default mask
    # that masks the left half of the image
    if conditioning_images is not None and conditioning_mask is None:
        # Get the shape of the latents
        batch_size, channels, height, width = latents.shape
        # Create a mask tensor with the same shape as latents
        conditioning_mask = torch.zeros_like(latents, device=latents.device)
        # Set the left half of the mask to 1
        conditioning_mask[:, :, :, : width // 2] = 1
        # conditioning_mask = 1.0 - conditioning_mask  # debug

        # Ensure the mask has the right shape and device
        assert (
            conditioning_mask.shape == latents.shape
        ), f"Mask shape {conditioning_mask.shape} doesn't match latents shape {latents.shape}"
    elif conditioning_mask is not None:
        assert (
            conditioning_images is not None
        ), "Conditioning mask requires conditioning images"
    else:
        conditioning_mask = None

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    # print(f"rho: {rho}")
    # print(f"sigma_min: {sigma_min}, sigma_max: {sigma_max}")

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0
    # print(t_steps)
    # print([f"{v.item():.3f}" for v in t_steps])

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    # if conditioning_images is not None and not replacement_heuristic:
    #     # if not replacement heuristic, use the conditioning images directly
    #     x_next = torch.where(conditioning_mask == 0, x_next, conditioning_images)

    if conditioning_images is not None:
        # replace the left half of the image with the conditioning images
        if replacement_heuristic:
            x_next = x_next + torch.where(
                conditioning_mask == 0, 0.0, conditioning_images
            )
        else:
            x_next = torch.where(conditioning_mask == 0, x_next, conditioning_images)

    if return_intermediate:
        x_intermediates = []
        denoised_intermediates = []

    denoised = None
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if return_intermediate:
            x_intermediates.append(x_cur)
            denoised_intermediates.append(denoised if denoised is not None else x_cur)

        if conditioning_images is not None and replacement_heuristic:
            # replacement heuristic
            noise = randn_like(conditioning_images) * t_cur
            x_cur = torch.where(
                conditioning_mask == 0, x_cur, conditioning_images + noise
            )

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        if conditioning_images is not None and not replacement_heuristic:
            x_hat = torch.where(conditioning_mask == 0, x_hat, conditioning_images)

        # Euler step.
        if hasattr(net, "edm_conditional_training") and net.edm_conditional_training:
            if conditioning_mask is not None and not replacement_heuristic:
                # assert that mask is the same for all channels
                assert torch.equal(
                    conditioning_mask[:, 0, :, :], conditioning_mask[:, 1, :, :]
                ), "mask should be the same for all channels"
                assert torch.equal(
                    conditioning_mask[:, 1, :, :], conditioning_mask[:, 2, :, :]
                ), "mask should be the same for all channels"

                m = conditioning_mask[:, 0:1, :, :]
                # repeat m for all images in the batch
                # m = m.repeat(x_hat.shape[0], 1, 1, 1)
            else:
                # we zero m in the replacement heuristic, otherwise it doesn't work
                m = torch.zeros(
                    (x_hat.shape[0], 1, x_hat.shape[2], x_hat.shape[3]),
                    device=x_hat.device,
                    dtype=x_hat.dtype,
                )

            denoised = net(x_hat, t_hat, class_labels, conditional_mask=m).to(
                torch.float64
            )
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

        # print(f"max of denoised: {denoised[0].max()}")
        # print(f"min of denoised: {denoised[0].min()}")

        # d_cur = (x_hat - denoised) / t_hat
        d_cur = (x_hat - denoised) / t_hat
        # debug = denoised - x_hat
        # print(t_hat)

        # print(f"norm of denoised: {denoised[0].norm()}")
        # print("--------------------------------")
        # print(f"max of denoised: {denoised[0].max()}")
        # print(f"max of x_hat: {x_hat[0].max()}")

        if conditioning_images is not None:  # and not replacement_heuristic:
            # we wont update the conditioned pixels
            d_cur = torch.where(conditioning_mask == 0, d_cur, 0.0)

        x_next = x_hat + (t_next - t_hat) * d_cur
        # print((t_next - t_hat) / t_hat)
        # x_next = x_hat + 0.0001 * (denoised - x_hat) / (0.01**2)

        # Apply 2nd order correction.
        if i < num_steps - 1:
            if (
                hasattr(net, "edm_conditional_training")
                and net.edm_conditional_training
            ):
                # m must exist here
                denoised = net(x_hat, t_hat, class_labels, conditional_mask=m).to(
                    torch.float64
                )
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)

            d_prime = (x_next - denoised) / t_next
            if conditioning_images is not None:  # and not replacement_heuristic:
                # we wont update the conditioned pixels
                d_prime = torch.where(conditioning_mask == 0, d_prime, 0.0)

            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    if conditioning_images is not None:
        # the replacement heuristic doesn't quite end up at exactly the conditional pixels
        # so we clamp them here again
        x_next = torch.where(conditioning_mask == 0, x_next, conditioning_images)

        # assert that conditional pixels match
        assert torch.allclose(
            x_next[conditioning_mask == 1].float(),
            conditioning_images[conditioning_mask == 1].float(),
        ), "Conditional pixels do not match"

    if return_intermediate:
        x_intermediates.append(x_next)
        denoised_intermediates.append(denoised)
        return x_next, x_intermediates, denoised_intermediates

    return x_next


# ----------------------------------------------------------------------------
# Classic Langevin Sampler (Langevin Dynamics with decaying step size)
def langevin_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    step_size_init=1.0,
    step_size_final=0.01,
    temperature_init=1.0,
    temperature_final=0.01,
    conditioning_images=None,
    conditioning_mask=None,
    return_intermediate=False,
    replacement_heuristic=False,
    **kwargs,
):
    """
    Langevin dynamics sampler with annealing step size and temperature.

    The Langevin SDE is: dx = -∇U(x)dt + √(2T) dW
    where U(x) is the energy function (learned by the network), T is temperature,
    and dW is Brownian motion.

    In the context of diffusion models, the energy function is related to the
    score function: ∇U(x) = -score(x) = (x - denoised) / sigma^2
    """
    print("[WARNING] Using Langevin sampler")

    if conditioning_images is not None and conditioning_mask is None:
        # Create default mask for left half of image
        batch_size, channels, height, width = latents.shape
        conditioning_mask = torch.zeros_like(latents, device=latents.device)
        conditioning_mask[:, :, :, : width // 2] = 1
    elif conditioning_mask is not None:
        assert (
            conditioning_images is not None
        ), "Conditioning mask requires conditioning images"
    else:
        conditioning_mask = None

    # Adjust noise levels based on what's supported by the network
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    # print(f"rho: {rho}")

    # Time step discretization (same as EDM)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Initialize with noise
    x = latents.to(torch.float64) * t_steps[0]
    if conditioning_images is not None:
        x = torch.where(conditioning_mask == 0, x, conditioning_images)

    if return_intermediate:
        x_intermediates = []

    # Annealing schedules for step size and temperature
    step_sizes = torch.linspace(
        step_size_init, step_size_final, num_steps, device=latents.device
    )
    temperatures = torch.linspace(
        temperature_init, temperature_final, num_steps, device=latents.device
    )

    # Main Langevin dynamics loop
    denoised = None
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        if return_intermediate:
            # x_intermediates.append(x.clone())
            x_intermediates.append(
                denoised.clone() if denoised is not None else x.clone()
            )

        # Get current step size and temperature
        step_size = step_sizes[i]
        temperature = temperatures[i]

        # Compute the score function (energy gradient)
        if hasattr(net, "edm_conditional_training") and net.edm_conditional_training:
            if conditioning_mask is not None:
                m = conditioning_mask[:, 0:1, :, :]
            else:
                m = torch.zeros(
                    (x.shape[0], 1, x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                )
            denoised = net(x, t_cur, class_labels, conditional_mask=m).to(torch.float64)
        else:
            denoised = net(x, t_cur, class_labels).to(torch.float64)

        # Score function: ∇log p(x) = (x - denoised) / sigma^2
        # In EDM, the network predicts the denoised version, so score = (x - denoised) / t_cur^2
        score = denoised - x  # / (t_cur**2)
        # print(f"score: {score[0].max()}")
        # print(f"x max: {x[0].max()}")

        # Apply conditioning mask (don't update conditioned pixels)
        if conditioning_images is not None:
            score = torch.where(conditioning_mask == 0, score, 0.0)

        # Langevin dynamics step: x_{t+1} = x_t + step_size * score + sqrt(2 * step_size * temperature) * noise
        noise = randn_like(x)
        step_size = 1 / (i + 3)  # t_cur - t_next
        # print(step_size)
        x = x + step_size * score + torch.sqrt(2 * step_size * temperature) * noise

        # Ensure conditioned pixels remain fixed
        if conditioning_images is not None:
            x = torch.where(conditioning_mask == 0, x, conditioning_images)

    if return_intermediate:
        x_intermediates.append(x)
        return x, x_intermediates

    return x


# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.


def ablation_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="heun",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    ancestral_sampling=False,
    conditioning_images=None,
    conditioning_mask=None,
    repaint_enable=False,
    repaint_jump=10,  # J: number of steps to "jump back" (increase noise)
    repaint_repeats=1,  # R: how many resamples per step
    return_intermediate=False,
    gradient_guidance=False,
    tds=False,
    tds_n_steps_each=1,
    **sampler_kwargs,
):
    assert solver in ["euler", "heun"]
    assert discretization in ["vp", "ve", "iddpm", "edm"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]

    if return_intermediate:
        x_intermediates = []
        denoised_intermediates = []

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = (
        lambda beta_d, beta_min: lambda t: (
            np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
        )
        ** 0.5
    )
    vp_sigma_deriv = (
        lambda beta_d, beta_min: lambda t: 0.5
        * (beta_min + beta_d * t)
        * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = (
        lambda beta_d, beta_min: lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        assert discretization == "edm"
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == "linear"
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == "none"
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    # ------------------------------------------------------------------------
    # Branch: DDPM ancestral sampling + RePaint mask-conditioning (optional)
    # Uses the same t_steps/sigma/s/etc. computed above.
    # ------------------------------------------------------------------------
    if ancestral_sampling:
        # print("Using ancestral sampling...")

        # Map EDM sigma(t) to DDPM cumulative: \bar{alpha}(t) = 1 / (1 + sigma(t)^2)
        alpha_bar_of_t = lambda t: 1.0 / (1.0 + sigma(t) ** 2)

        # === RePaint params ===
        # Set repaint_enable=True to activate (mask fusion happens regardless if mask is not None).

        # NOTE: You can wire these as function args later if you like.

        # Expected external tensors:
        # - conditioning_images: [B, C, H, W]
        # - conditioning_mask:   [B, C, H, W], binary {0,1}, where 1 => use conditioning_images
        # If you want to make them optional, guard with `if conditioning_images is not None:`
        if conditioning_mask is not None:
            m = conditioning_mask.to(torch.float64)  # [B,C,H,W], 1=known/conditioned
            inv_m = 1.0 - m
            x0_cond = conditioning_images.to(torch.float64)
            D = latents[0].numel()  # per-sample dimension

        # Helper: Forward diffuse between two cumulative noise levels (VP).
        # Given x at a_src, produce x' at a_dst <= a_src (i.e., more noisy).
        def forward_diffuse_between_levels(x, a_src, a_dst, rng_like=randn_like):
            # factor r = sqrt(a_dst / a_src), variance add = 1 - a_dst/a_src
            r = (a_dst / a_src).clamp(min=1e-12, max=1.0).sqrt()
            var_add = 1.0 - (a_dst / a_src).clamp(min=0.0, max=1.0)
            return r * x + var_add.sqrt() * rng_like(x)

        # Helper: at the current level a_cur, replace known region with properly noised conditioning image.
        def fuse_known_region(x_cur, t_cur):
            a_cur = alpha_bar_of_t(t_cur)  # scalar tensor (float64)
            # Noisify the ground-truth conditioned pixels to exactly match current noise:
            # x_t_known = sqrt(a_cur) * x0_known + sqrt(1-a_cur) * eps
            eps_k = randn_like(x_cur)
            x_known_noisy = a_cur.sqrt() * x0_cond + (1.0 - a_cur).sqrt() * eps_k
            # Fuse:
            return m * x_known_noisy + inv_m * x_cur

        # TDS Helpers ---------------------------------------------------------------
        def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
            """Systematic resampling: weights are assumed normalized (sum==1), shape [B]."""
            B = weights.shape[0]
            # u0 ~ U[0, 1/B)
            u0 = torch.rand((), device=weights.device, dtype=weights.dtype) / B
            positions = (
                torch.arange(B, device=weights.device, dtype=weights.dtype) + u0
            ) / B
            cumsum = torch.cumsum(weights, dim=0)
            idxs = torch.searchsorted(cumsum, positions, right=True)
            return idxs.clamp_max(B - 1).to(torch.long)

        def flatten2(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1)

        # Twisting log p(y | x^t) (Eq. 13 in Wu et al. 2023) -------------------
        def log_twisting_fn(
            x_noisy: torch.Tensor, t_scalar: torch.Tensor
        ) -> torch.Tensor:
            """Per-particle log N(y; denoise(x,t)_M, sigma(t)^2 I_M). Returns shape [B]."""
            sig = sigma(t_scalar)  # scalar (float64)
            with torch.no_grad():
                s_t = s(t_scalar)
                x0_hat = net(x_noisy / s_t, sig, class_labels).to(
                    torch.float64
                )  # EDM denoiser predicts x0
                diff = (x0_hat - x0_cond) * m
                diff2 = flatten2(diff).pow(2).sum(dim=1)  # [B]
                dim_m = flatten2(m).sum(
                    dim=1
                )  # [B] (can be fractional if mask not {0,1})
                logp = -0.5 * (diff2 / (sig**2)) - 0.5 * dim_m * torch.log(
                    2 * torch.pi * (sig**2)
                )
                return logp

        # One-step Gaussian log-density with diagonal var -----------------------
        def gaussian_log_prob(x_new, mean, var_scalar: torch.Tensor) -> torch.Tensor:
            """Returns per-particle log N(x_new; mean, var I). Shapes [B,C,H,W] and scalar var."""
            diff = flatten2(x_new - mean)
            qf = diff.pow(2).sum(dim=1) / var_scalar  # quadratic form
            return -0.5 * (qf + D * torch.log(2 * torch.pi * var_scalar))

        # Initialize **exactly like EDM path** (same scale):
        t0 = t_steps[0]
        x_next = latents.to(torch.float64) * (sigma(t0) * s(t0))

        # If TDS is enabled, initialize importance weights at t0
        if tds:
            logp = log_twisting_fn(x_next, t0)  # [B]
            weights = torch.softmax(logp, dim=0)  # normalized importance weights
            old_p = logp.clone()  # store per-particle twisting logp

        # Iterate reverse ladder: indices i = 0..N-1, (t_cur -> t_next)
        N = num_steps
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0..N-1
            x_cur = x_next

            if conditioning_mask is not None and not tds:
                x_cur = fuse_known_region(x_cur, t_cur)

            # Current schedule values
            s_cur = s(t_cur)
            sig_cur = sigma(t_cur)
            a_cur = alpha_bar_of_t(t_cur)
            a_next = alpha_bar_of_t(
                t_next
            )  # cumulative at the *next* (less noisy) level

            # Per-step α_t, β_t from adjacent cumulative bars: a_cur = a_next * alpha_t
            alpha_t = (a_cur / a_next).clamp(min=1e-12, max=1.0)
            beta_t = (1.0 - alpha_t).clamp(min=0.0)

            if tds:
                # ----- Twisted Diffusion Sampler step (replaces ancestral update) -----
                # variance difference for this ladder step (>=0): sigma_i^2 - sigma_{i+1}^2
                sigma_tilde_sq = (sigma(t_cur) ** 2 - sigma(t_next) ** 2).clamp(
                    min=1e-12
                )

                for _ in range(tds_n_steps_each):
                    # Resample particles according to current weights
                    idxs = systematic_resample(weights)
                    x_cur = x_cur.index_select(0, idxs)
                    old_p = old_p.index_select(0, idxs)

                    # Guided score at (x_cur, t_cur): score + ∇_x log N(y | x0_hat(x), sigma^2)
                    with torch.enable_grad():
                        x_req = x_cur.detach().requires_grad_(True)
                        x0_hat = net(x_req / s_cur, sig_cur, class_labels).to(
                            torch.float64
                        )
                        score_uncond = (x0_hat - x_req) / (
                            sig_cur**2
                        )  # Tweedie: s = (x0 - x)/sigma^2

                        # log N(y | x0_hat_M, sigma^2) w.r.t. x (differentiating through net)
                        diff = (x0_hat - x0_cond) * m
                        diff_l2 = flatten2(diff).pow(2).sum(dim=1)  # [B]
                        var_vec = (sig_cur**2).expand(diff_l2.shape[0])  # [B]
                        log_gauss = -diff_l2 / (2.0 * var_vec)  # [B]
                        grad_log = torch.autograd.grad(
                            log_gauss.sum(),
                            x_req,
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                        guided_score = score_uncond + grad_log  # [B,C,H,W]

                    # Proposal: single Langevin-like step with step-size sigma_tilde_sq
                    noise = S_noise * randn_like(x_cur)
                    new_x = (
                        x_req.detach()
                        + sigma_tilde_sq * guided_score
                        + sigma_tilde_sq.sqrt() * noise
                    )

                    # Twisting weights at current level t_cur
                    new_p = log_twisting_fn(new_x, t_cur)  # [B]

                    # Transition densities (per-particle):
                    with torch.no_grad():
                        # unconditional mean
                        x0_hat_u = net(x_cur / s_cur, sig_cur, class_labels).to(
                            torch.float64
                        )
                        score_u = (x0_hat_u - x_cur) / (sig_cur**2)
                        mean_uncond = x_cur + sigma_tilde_sq * score_u

                    mean_cond = (
                        x_cur + sigma_tilde_sq * guided_score
                    )  # uses guided_score from above

                    log_uncond = gaussian_log_prob(new_x, mean_uncond, sigma_tilde_sq)
                    log_cond = gaussian_log_prob(new_x, mean_cond, sigma_tilde_sq)

                    # Importance weight update: w ∝ exp( (uncond + new_p) - (cond + old_p) )
                    log_w = (log_uncond + new_p) - (log_cond + old_p)  # [B]
                    weights = torch.softmax(log_w, dim=0)

                    # Advance particles & carry twisting logp forward
                    x_cur = new_x.detach()
                    old_p = new_p.detach()

                if return_intermediate:
                    x_intermediates.append(x_cur)
                    denoised_intermediates.append(x0_hat)

                # After inner repeats, we are at the next noise level for outer loop
                x_next = x_cur
                continue  # skip the regular ancestral update below
            # ---------- GRADIENT GUIDANCE (only in ancestral mode) ----------
            elif gradient_guidance:
                # Compute ∇_{x_t} log N(y | \hat x0(x_t), var) where y := conditioning_images on mask
                # We must enable autograd through the net.
                with torch.enable_grad():
                    x_noisy = x_cur.detach().requires_grad_(True)
                    # net predicts x0; input uses VP scaling
                    x0_hat_autograd = net(x_noisy / s_cur, sig_cur, class_labels).to(
                        torch.float64
                    )
                    # Tweedie (for clarity): score = (x0_hat - x_noisy) / sigma^2
                    # denoised_x equals x0_hat_autograd; keep explicit in case you switch param later
                    denoised_x = x0_hat_autograd

                    # Masked squared error to conditioning:
                    diff = (denoised_x - x0_cond) * m  # [B,C,H,W]
                    diff_l2 = diff.flatten(1).pow(2).sum(dim=1)  # [B]

                    # use current sigma^2 as the gaussian variance
                    log_gauss = -diff_l2 / (2.0 * sig_cur**2)  # [B]
                    # Gradient wrt x_t, [B,C,H,W]
                    grad_log = torch.autograd.grad(
                        log_gauss.sum(), x_noisy, retain_graph=False, create_graph=False
                    )[0]

                # Convert guidance on log-prob to an x0 shift via Tweedie link:
                # x0_hat_guided = x0_hat + sigma^2 * grad_log
                x0_hat = x0_hat_autograd.detach() + (sig_cur**2) * grad_log.detach()
            else:
                # standard EDM denoiser (predict x0)
                x0_hat = net(x_cur / s_cur, sig_cur, class_labels).to(torch.float64)

            if return_intermediate:
                x_intermediates.append(x_cur)
                denoised_intermediates.append(x0_hat)

            # DDPM posterior: q(x_{t-1} | x_t, x0_hat) = N(tilde_mu, tilde_beta I)
            c1 = a_next.sqrt() * beta_t / (1.0 - a_cur)
            c2 = alpha_t.sqrt() * (1.0 - a_next) / (1.0 - a_cur)
            mu = c1 * x0_hat + c2 * x_cur
            tilde_beta = ((1.0 - a_next) / (1.0 - a_cur) * beta_t).clamp(min=0.0)

            # Sample (stochastic except on last step to t=0)
            if i < N - 1:
                eps = S_noise * randn_like(x_cur)
                x_next = mu + tilde_beta.sqrt() * eps
            else:
                x_next = mu  # deterministic final step

            # ==========================
            # RePaint resampling (J, R)
            # ==========================
            if (
                repaint_enable
                and repaint_repeats > 0
                and repaint_jump > 0
                and i < N - 1
            ):
                # We will "jump back" to a *more noisy* level (earlier index) then come forward again.
                # Index arithmetic: higher noise == earlier step index (j < i).
                i_up = max(i - repaint_jump, 0)
                # Pre-compute the subsequence of indices to re-run: i_up .. i (inclusive).
                # These correspond to t_up -> ... -> t_cur levels in *reverse* chain order.
                # We'll start from current x_next (already at level t_next), so first push x_next forward to t_i (current),
                # then further to t_{i_up} (more noise), and then run reverse ancestral steps back to t_next.
                for _ in range(repaint_repeats):
                    # 1) Push x_next forward to t_cur (undo the just-done reverse step).
                    #    Currently x_next is at a_next; we need a_cur (more noise): a_cur < a_next ? No.
                    # Careful: in VP, as we go forward in time (more noise), a decreases.
                    # Here a_next <= a_cur (since t_next is less noisy). So to go from a_next -> a_cur, we indeed add noise.
                    x_fwd = forward_diffuse_between_levels(
                        x_next, a_next, a_cur, rng_like=randn_like
                    )

                    # 2) Push further forward to the *up* level a_up associated with i_up.
                    t_up = t_steps[i_up]
                    a_up = alpha_bar_of_t(t_up)
                    # Now a_up <= a_cur (up is even more noisy), so we can add noise again:
                    x_fwd = forward_diffuse_between_levels(
                        x_fwd, a_cur, a_up, rng_like=randn_like
                    )

                    # 3) Run reverse ancestral steps from i_up .. i to get back to the same target level t_next.
                    x_rs = x_fwd
                    for j in range(i_up, i + 1):
                        t_j = t_steps[j]
                        t_jnext = t_steps[j + 1]
                        # Fuse mask at level t_j
                        x_rs = fuse_known_region(x_rs, t_j)

                        s_j = s(t_j)
                        sig_j = sigma(t_j)
                        a_j = alpha_bar_of_t(t_j)
                        a_jnext = alpha_bar_of_t(t_jnext)

                        alpha_j = (a_j / a_jnext).clamp(min=1e-12, max=1.0)
                        beta_j = (1.0 - alpha_j).clamp(min=0.0)

                        x0_hat_j = net(x_rs / s_j, sig_j, class_labels).to(
                            torch.float64
                        )

                        c1_j = a_jnext.sqrt() * beta_j / (1.0 - a_j)
                        c2_j = alpha_j.sqrt() * (1.0 - a_jnext) / (1.0 - a_j)
                        mu_j = c1_j * x0_hat_j + c2_j * x_rs
                        tb_j = ((1.0 - a_jnext) / (1.0 - a_j) * beta_j).clamp(min=0.0)

                        # Stochastic except when j == N-1 (but here j <= i < N-1 always)
                        x_rs = mu_j + S_noise * randn_like(x_rs) * tb_j.sqrt()

                    # After the mini reverse loop, we are back at level t_next of outer i.
                    # Update outer state with the resampled value.
                    x_next = x_rs

                    if return_intermediate:
                        x_intermediates.append(x_rs)
                        denoised_intermediates.append(x0_hat_j)

        if return_intermediate:
            return x_next, x_intermediates, denoised_intermediates

        return x_next

    else:
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(
                torch.float64
            )
            if return_intermediate:
                x_intermediates.append(x_hat)
                denoised_intermediates.append(denoised)

            d_cur = (
                sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
            ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Apply 2nd order correction.
            if solver == "euler" or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == "heun"
                denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(
                    torch.float64
                )
                d_prime = (
                    sigma_deriv(t_prime) / sigma(t_prime)
                    + s_deriv(t_prime) / s(t_prime)
                ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(
                    t_prime
                ) * denoised
                x_next = x_hat + h * (
                    (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
                )

        if return_intermediate:
            x_intermediates.append(x_next)
            denoised_intermediates.append(denoised)
            return x_next, x_intermediates, denoised_intermediates

        return x_next


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def generate_images(
    network_pkl,
    outdir,
    subdirs=False,
    seeds=None,
    class_idx=None,
    max_batch_size=64,
    device=torch.device("cuda"),
    inpaint=False,
    replacement_heuristic=False,
    return_intermediate=False,
    init_dist=True,
    conditioning_mask=None,
    inpaint_one_image_per_class=False,
    save_images=True,
    **sampler_kwargs,
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""
    if init_dist:
        dist.init()
    if seeds is None:
        seeds = list(range(64))

    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0 and dist.get_world_size() > 1:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        pkl_data = pickle.load(f)
        net = pkl_data["ema"].to(device)
        dataset_path = pkl_data["dataset_kwargs"]["path"]

    # Other ranks follow.
    if dist.get_rank() == 0 and dist.get_world_size() > 1:
        torch.distributed.barrier()

    if inpaint:
        # load test images from CIFAR/FFHQ
        from collections import defaultdict
        from training.dataset import ImageFolderDataset

        if "cifar10" in dataset_path:
            inpaint_dataset = ImageFolderDataset(
                path="datasets/cifar10-32x32-test.zip",
                use_labels=True,
                xflip=False,
                cache="cache",
            )
        elif "ffhq" in dataset_path:
            inpaint_dataset = ImageFolderDataset(
                path=(
                    f"datasets/{dataset_path}"
                    if dataset_path == "ffhq-64x64.zip"
                    else dataset_path
                ),
                use_labels=True,
                xflip=False,
                cache="cache",
            )
        else:
            raise ValueError(f"Dataset {dataset_path} not supported for inpainting")

        inpaint_data_loader = torch.utils.data.DataLoader(
            inpaint_dataset, batch_size=max_batch_size
        )
        inpaint_data_loader_iter = iter(inpaint_data_loader)

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')

    # Initialize lists to collect results when return_intermediate=True
    all_images = []
    all_x_intermediates = []
    all_conditioning_images = []
    all_denoised_intermediates = []
    end_idx = None

    for i, batch_seeds in tqdm.tqdm(
        enumerate(rank_batches), unit="batch", disable=(dist.get_rank() != 0)
    ):
        if dist.get_world_size() > 1:
            torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }
        have_ablation_kwargs = any(
            x in sampler_kwargs.keys()
            for x in ["solver", "discretization", "schedule", "scaling"]
        )
        have_langevin_kwargs = any(
            x in sampler_kwargs.keys()
            for x in [
                "step_size_init",
                "step_size_final",
                "temperature_init",
                "temperature_final",
            ]
        )

        if have_langevin_kwargs:
            print(sampler_kwargs)
            sampler_fn = langevin_sampler
        elif have_ablation_kwargs:
            sampler_fn = ablation_sampler
        else:
            sampler_fn = edm_sampler

        # print(f"Using sampler: {sampler_fn.__name__}")

        if inpaint:
            conditioning_images = next(inpaint_data_loader_iter)
            conditioning_images = (
                conditioning_images[0].to(device).to(torch.float32) / 127.5 - 1
            )  # torch.stack([x[0] for x in conditioning_images])
            # conditioning_images = conditioning_images.to(torch.float32) / 127.5 - 1

            if batch_size != max_batch_size:
                conditioning_images = conditioning_images[:batch_size]

            # print(conditioning_images.shape)
            start_idx = end_idx if end_idx is not None else 0
            end_idx = start_idx + batch_size
            # print(f"start_idx, end_idx: {start_idx, end_idx}")
            cond_mask = conditioning_mask[start_idx:end_idx]
            # print(conditioning_mask.shape)
            # print(cond_mask.shape)
            # if conditioning_mask is not None:
            #     conditioning_mask = conditioning_mask.to(device)
            #     conditioning_mask = conditioning_mask.to(torch.float32)
            # else:
            #     conditioning_mask = None

            # if inpaint_one_image_per_class:
            #     assert (
            #         "cifar10" in dataset_path
            #     ), "CIFAR-10 is the only dataset that supports inpainting one image per class"
            #     final_imgs = defaultdict(list)
            #     for i in range(100):
            #         label = int(np.argmax(dataset[i][1]))
            #         # print(label)
            #         if len(final_imgs[label]) == 0:
            #             final_imgs[label] = dataset[i][0]
            #         # break
            #         if len(list(final_imgs.keys())) == 10:
            #             break

            #     # batch_size = 10
            #     conditioning_images = torch.stack(
            #         [torch.tensor(x) for x in list(final_imgs.values())]
            #     )
            # else:
            #     conditioning_images = torch.stack(
            #         [
            #             torch.tensor(x)
            #             for x in [dataset[i][0] for i in range(batch_size)]
            #         ]
            #     )

            # conditioning_images = conditioning_images.to(device)
            # conditioning_images = conditioning_images.to(torch.float32) / 127.5 - 1
        else:
            conditioning_images = None
            cond_mask = None

        if return_intermediate:
            images, x_intermediates, denoised_intermediates = sampler_fn(
                net,
                latents,
                class_labels,
                randn_like=rnd.randn_like,
                conditioning_images=conditioning_images,
                replacement_heuristic=replacement_heuristic,
                return_intermediate=return_intermediate,
                conditioning_mask=cond_mask,
                **sampler_kwargs,
            )
            # Collect results instead of returning immediately
            all_x_intermediates.append(torch.stack(x_intermediates))
            all_denoised_intermediates.append(torch.stack(denoised_intermediates))
        else:
            images = sampler_fn(
                net,
                latents,
                class_labels,
                randn_like=rnd.randn_like,
                conditioning_images=conditioning_images,
                replacement_heuristic=replacement_heuristic,
                return_intermediate=return_intermediate,
                conditioning_mask=cond_mask,
                **sampler_kwargs,
            )

        all_images.append(images)
        all_conditioning_images.append(conditioning_images)

        if save_images:
            # Save images.
            images_np = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = (
                    os.path.join(outdir, f"{seed-seed%1000:06d}") if subdirs else outdir
                )
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"{seed:06d}.png")
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                else:
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    if dist.get_world_size() > 1:
        torch.distributed.barrier()
    dist.print0("Done.")

    # Concatenate all collected results
    all_images_tensor = torch.cat(all_images, dim=0)

    all_x_intermediates_tensor = None
    all_denoised_intermediates_tensor = None
    all_conditioning_images_tensor = None
    if return_intermediate:
        all_x_intermediates_tensor = torch.cat(all_x_intermediates, dim=0)
        all_denoised_intermediates_tensor = torch.cat(all_denoised_intermediates, dim=0)

    if inpaint:
        all_conditioning_images_tensor = torch.cat(all_conditioning_images, dim=0)

    return (
        all_images_tensor,
        all_x_intermediates_tensor,
        all_conditioning_images_tensor,
        all_denoised_intermediates_tensor,
    )


@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--seeds",
    help="Random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="0-63",
    show_default=True,
)
@click.option(
    "--subdirs", help="Create subdirectory for every 1000 seeds", is_flag=True
)
@click.option("--inpaint", help="Inpaint the left half of the image", is_flag=True)
@click.option(
    "--class",
    "class_idx",
    help="Class label  [default: random]",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of sampling steps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=18,
    show_default=True,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--sigma_max",
    help="Highest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--rho",
    help="Time step exponent",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=7,
    show_default=True,
)
@click.option(
    "--S_churn",
    "S_churn",
    help="Stochasticity strength",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_min",
    "S_min",
    help="Stoch. min noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_max",
    "S_max",
    help="Stoch. max noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default="inf",
    show_default=True,
)
@click.option(
    "--S_noise",
    "S_noise",
    help="Stoch. noise inflation",
    metavar="FLOAT",
    type=float,
    default=1,
    show_default=True,
)
@click.option(
    "--solver",
    help="Ablate ODE solver",
    metavar="euler|heun",
    type=click.Choice(["euler", "heun"]),
)
@click.option(
    "--disc",
    "discretization",
    help="Ablate time step discretization {t_i}",
    metavar="vp|ve|iddpm|edm",
    type=click.Choice(["vp", "ve", "iddpm", "edm"]),
)
@click.option(
    "--schedule",
    help="Ablate noise schedule sigma(t)",
    metavar="vp|ve|linear",
    type=click.Choice(["vp", "ve", "linear"]),
)
@click.option(
    "--scaling",
    help="Ablate signal scaling s(t)",
    metavar="vp|none",
    type=click.Choice(["vp", "none"]),
)
def main(**kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate images using Langevin dynamics sampler
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \\
        --step_size_init=1.0 --step_size_final=0.01 --temperature_init=1.0 --temperature_final=0.01
    """
    generate_images(**kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # debug
    # network_pkl = 'training-runs/00020-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch512-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01/network-snapshot-200000.pkl'
    # outdir = 'out_inpaint'
    # steps = 256
    # S_churn = 40
    # S_min = 0.05
    # S_max = 50
    # S_noise = 1.003
    # seeds = list(range(64,128))

    # generate_images(network_pkl=network_pkl, outdir=outdir, inpaint=True, num_steps=steps, S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise, seeds=seeds)

    main()

# ----------------------------------------------------------------------------
