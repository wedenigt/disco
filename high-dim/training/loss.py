# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DISCOLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, t0=0.01, gamma=1.0):
        """
        P_mean, P_std: log-normal schedule params for σ(t)
        sigma_data   : EDM sigma_data
        t0           : small σ(0) used in the posterior target
        gamma        : weight for the conditional (masked) loss
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.t0 = t0
        self.gamma = gamma

    # -------- utilities (shared) --------
    @staticmethod
    def _sample_t(B, device, P_mean, P_std):
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        return (rnd_normal * P_std + P_mean).exp()  # σ(t)

    @staticmethod
    def _left_half_mask(x):
        B, C, H, W = x.shape
        m = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        m[..., : W // 2] = 1.0
        return m

    @staticmethod
    def _sample_random_mask(
        x, pixel_obs_prob: float = 0.5, min_patch_frac: float = 0.25, modes=None
    ):
        B, C, H, W = x.shape
        m = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        modes = torch.randint(low=0, high=3, size=(B,), device=x.device)

        # Half masks (left or right)
        half_idx = (modes == 0).nonzero(as_tuple=False).flatten()
        if half_idx.numel() > 0:
            left_flags = torch.rand(half_idx.shape[0], device=x.device) < 0.5
            for i, b in enumerate(half_idx.tolist()):
                if bool(left_flags[i]):
                    m[b, :, :, : W // 2] = 1.0
                else:
                    m[b, :, :, W // 2 :] = 1.0

        # Random pixels (Bernoulli)
        pix_idx = (modes == 1).nonzero(as_tuple=False).flatten()
        if pix_idx.numel() > 0:
            rand_mask = (
                torch.rand(pix_idx.shape[0], 1, H, W, device=x.device) < pixel_obs_prob
            ).to(x.dtype)
            m[pix_idx] = rand_mask

        # Random rectangular patches
        patch_idx = (modes == 2).nonzero(as_tuple=False).flatten()
        if patch_idx.numel() > 0:
            min_area = int(min_patch_frac * H * W)
            for b in patch_idx.tolist():
                for _ in range(10):
                    h = int(
                        torch.randint(
                            low=max(1, H // 8), high=H + 1, size=(1,), device=x.device
                        ).item()
                    )
                    w = int(
                        torch.randint(
                            low=max(1, W // 8), high=W + 1, size=(1,), device=x.device
                        ).item()
                    )
                    if h * w >= min_area:
                        break
                y0 = int(
                    torch.randint(
                        low=0, high=H - h + 1, size=(1,), device=x.device
                    ).item()
                )
                x0 = int(
                    torch.randint(
                        low=0, high=W - w + 1, size=(1,), device=x.device
                    ).item()
                )
                m[b, :, y0 : y0 + h, x0 : x0 + w] = 1.0

        return m

    # -------- JOINT loss (original DISCO) --------
    @torch.no_grad()
    def _posterior_indices(self, z, x):
        """
        Minibatch posterior: p0(x | z) ∝ N(z; x, t0^2 I) over the batch.
        Returns y indices for target selection.
        """
        B, C, H, W = x.shape
        z_i = z.view(B, 1, C, H, W)  # [B,1,C,H,W]
        x_j = x.view(1, B, C, H, W)  # [1,B,C,H,W]
        sq_l2s = ((z_i - x_j) ** 2).sum(dim=[2, 3, 4])  # [B,B]
        logits = -sq_l2s / (2 * (self.t0**2))
        probs = logits.softmax(dim=1)
        y_idxs = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
        return y_idxs

    @torch.no_grad()
    def _posterior_indices_full(self, z, all_images, chunk_size=256):
        """
        Full-dataset posterior: p0(x | z) ∝ N(z; x, t0^2 I) over all images.
        Returns indices into all_images.
        """
        B, C, H, W = z.shape
        N = all_images.shape[0]

        logits = torch.empty(B, N, device=z.device, dtype=torch.float64)
        z_i = z.to(torch.float64).view(B, 1, C, H, W)

        t0 = float(self.t0)
        denom = 2.0 * t0 * t0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_j = all_images[start:end].to(torch.float64).view(1, end - start, C, H, W)
            sq_l2s = ((z_i - x_j) ** 2).sum(dim=(2, 3, 4))
            logits[:, start:end] = -sq_l2s / denom

        # Optional but useful for diagnostics.
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite posterior logits.")

        # Log-space categorical sampling.
        g = -torch.empty_like(logits).exponential_().log()
        y_idxs = torch.argmax(logits + g, dim=1)

        return y_idxs

    # @torch.no_grad()
    # def _posterior_indices_full(self, z, all_images):
    #     """
    #     Full-dataset posterior: p0(x | z) ∝ N(z; x, t0^2 I) over all images.
    #     Uses ||z-x||^2 = ||z||^2 + ||x||^2 - 2*z·x for memory efficiency.
    #     Returns indices into all_images.
    #     """
    #     B = z.shape[0]
    #     N = all_images.shape[0]
    #     z_flat = z.reshape(B, -1).double()
    #     x_flat = all_images.reshape(N, -1).double()
    #     z_sq = (z_flat**2).sum(dim=1, keepdim=True)  # [B, 1]
    #     x_sq = (x_flat**2).sum(dim=1, keepdim=True).T  # [1, N]
    #     sq_dist = (z_sq + x_sq - 2 * (z_flat @ x_flat.T)).clamp_(min=0.0)  # [B, N]
    #     logits = -sq_dist / (2 * self.t0**2)
    #     probs = logits.softmax(dim=1)
    #     y_idxs = torch.multinomial(probs.float(), num_samples=1).squeeze(1)  # [B]
    #     return y_idxs

    def prepare_joint_loss(
        self, x, labels=None, augment_labels=None, full_dataset_images=None
    ):
        """
        Prepare data for joint loss computation. Returns:
          z: [B,C,H,W] - noisy input
          sigma: [B,1,1,1] - noise levels
          y: [B,C,H,W] - targets
          weight: [B,1,1,1] - EDM weighting
          num_diff_x: diagnostic
        """
        device = x.device
        B, C, H, W = x.shape

        sigma = self._sample_t(B, device, self.P_mean, self.P_std)  # [B,1,1,1]
        z = x + torch.randn_like(x, device=device) * sigma  # x_t everywhere

        if full_dataset_images is not None:
            y_idxs = self._posterior_indices_full(z, full_dataset_images)
            y = full_dataset_images[y_idxs]
            num_diff_x = ((y - x).view(B, -1).pow(2).mean(dim=1) > 1e-4).float().mean()
        else:
            y_idxs = self._posterior_indices(z, x)
            y = x[y_idxs]
            num_diff_x = (~(y_idxs.to("cpu") == torch.arange(B))).float().sum() / B

        # EDM weighting
        weight = (sigma**2 + self.sigma_data**2) / (
            sigma * self.sigma_data
        ) ** 2  # [B,1,1,1]

        return z, sigma, y, weight, num_diff_x

    # -------- CONDITIONAL / MASKED loss (left half observed) --------
    def prepare_mask_loss(self, x, labels=None, augment_labels=None):
        """
        Prepare data for mask loss computation. Returns:
          z_in: [B,C,H,W] - input for network
          sigma: [B,1,1,1] - noise levels
          y_target: [B,C,H,W] - targets
          weight: [B,1,1,1] - EDM weighting
          m: [B,C,H,W] - mask (1=observed, 0=to denoise)
          denom: [B,1,1,1] - normalization factor
        """
        device = x.device
        B, C, H, W = x.shape

        # 1 = observed (clean), 0 = to denoise. Randomize mask shape per sample.
        m = self._sample_random_mask(x)
        sigma = self._sample_t(B, device, self.P_mean, self.P_std)  # [B,1,1,1]

        # y_t: noise only on ~m (right half)
        eps = torch.randn_like(x, device=device)
        y_t = x * m + (x + sigma * eps) * (1.0 - m)  # clean left, noisy right

        # input: (y_t on right) + (x on left)
        z_in = y_t * (1.0 - m) + x * m

        # target: empirical posterior collapses to same-image clean target on ~m
        y_target = x

        # EDM weighting
        weight = (sigma**2 + self.sigma_data**2) / (
            sigma * self.sigma_data
        ) ** 2  # [B,1,1,1]

        # normalize by number of ~m pixels per sample for scale stability
        denom = (1.0 - m).sum(dim=[1, 2, 3], keepdim=True).clamp_min(1.0)

        return z_in, sigma, y_target, weight, m, denom

    # -------- call: combine both --------
    def __call__(
        self, net, images, labels=None, augment_pipe=None, full_dataset_images=None
    ):
        # (0) optional augmentation
        x, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )

        # (A) prepare joint loss data
        z_joint, sigma_joint, y_joint, weight_joint, num_diff_x = (
            self.prepare_joint_loss(
                x, labels, augment_labels, full_dataset_images=full_dataset_images
            )
        )

        # (B) prepare mask loss data
        if self.gamma > 0.0:
            z_mask, sigma_mask, y_mask, weight_mask, m_mask, denom_mask = (
                self.prepare_mask_loss(x, labels, augment_labels)
            )
        else:
            z_mask = sigma_mask = y_mask = weight_mask = m_mask = denom_mask = None

        # (C) single network call for both losses
        if self.gamma > 0.0:
            # Concatenate inputs for both losses
            z_combined = torch.cat([z_joint, z_mask], dim=0)
            sigma_combined = torch.cat([sigma_joint, sigma_mask], dim=0)
            augment_labels_combined = (
                torch.cat([augment_labels, augment_labels], dim=0)
                if augment_labels is not None
                else None
            )

            # Single network forward pass
            D_yn_combined = net(
                z_combined,
                sigma_combined,
                labels,
                augment_labels=augment_labels_combined,
            )

            # Split outputs
            B = z_joint.shape[0]
            D_yn_joint = D_yn_combined[:B]
            D_yn_mask = D_yn_combined[B:]

            # Compute joint loss
            loss_joint_map = weight_joint * ((D_yn_joint - y_joint) ** 2)

            # Compute mask loss
            per_pix = ((D_yn_mask - y_mask) ** 2) * (1.0 - m_mask)
            C, H, W = x.shape[1], x.shape[2], x.shape[3]
            loss_mask_map = weight_mask * per_pix * ((C * H * W) / denom_mask)
        else:
            # Only joint loss
            D_yn_joint = net(
                z_joint, sigma_joint, labels, augment_labels=augment_labels
            )
            loss_joint_map = weight_joint * ((D_yn_joint - y_joint) ** 2)
            loss_mask_map = torch.zeros_like(loss_joint_map)

        # (D) total
        loss_total_map = loss_joint_map + self.gamma * loss_mask_map
        return loss_total_map, num_diff_x, loss_joint_map, loss_mask_map


@persistence.persistent_class
class EDMLossWithMasks:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.gamma = gamma

    def _sample_t(self, B, device):
        return DISCOLoss._sample_t(B, device, self.P_mean, self.P_std)

    def _sample_random_mask(self, x):
        return DISCOLoss._sample_random_mask(x)

    def prepare_joint_loss(self, x, labels=None, augment_labels=None):
        device = x.device
        B, C, H, W = x.shape
        sigma = self._sample_t(B, device)  # [B,1,1,1]
        z = x + torch.randn_like(x, device=device) * sigma
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Provide all-zero conditional mask channel to match network input shape (zero means no conditioning)
        m_all = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return z, sigma, weight, m_all

    def prepare_mask_loss(self, x, labels=None, augment_labels=None):
        device = x.device
        B, C, H, W = x.shape
        m = self._sample_random_mask(x)
        sigma = self._sample_t(B, device)
        eps = torch.randn_like(x, device=device)
        y_t = x * m + (x + sigma * eps) * (1.0 - m)
        z_in = y_t * (1.0 - m) + x * m
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        denom = (1.0 - m).sum(dim=[1, 2, 3], keepdim=True).clamp_min(1.0)
        return z_in, sigma, weight, m, denom

    def __call__(self, net, images, labels=None, augment_pipe=None):
        x, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )

        # Prepare data for both losses
        z_joint, sigma_joint, weight_joint, m_joint = self.prepare_joint_loss(
            x, labels, augment_labels
        )
        z_mask, sigma_mask, weight_mask, m_mask, denom_mask = self.prepare_mask_loss(
            x, labels, augment_labels
        )

        # Concatenate inputs for both losses
        z_combined = torch.cat([z_joint, z_mask], dim=0)
        sigma_combined = torch.cat([sigma_joint, sigma_mask], dim=0)
        m_combined = torch.cat([m_joint, m_mask], dim=0)
        augment_labels_combined = torch.cat([augment_labels, augment_labels], dim=0)

        # Single network forward pass
        D_yn_combined = net(
            z_combined,
            sigma_combined,
            labels,
            augment_labels=augment_labels_combined,
            conditional_mask=m_combined,
        )

        # Split outputs
        B = z_joint.shape[0]
        D_yn_joint = D_yn_combined[:B]
        D_yn_mask = D_yn_combined[B:]

        # Compute joint loss
        loss_joint_map = weight_joint * ((D_yn_joint - x) ** 2)

        # Compute mask loss
        per_pix = ((D_yn_mask - x) ** 2) * (1.0 - m_mask)
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        loss_mask_map = weight_mask * per_pix * ((C * H * W) / denom_mask)

        # Total loss
        loss_total_map = loss_joint_map + self.gamma * loss_mask_map
        return loss_total_map
