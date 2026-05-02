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
"""The NCSNv2 model."""

from typing import Callable
import jax
import flax.linen as nn
import functools

from .utils import get_sigmas, register_model
from .layers import (
    CondRefineBlock,
    RefineBlock,
    ResidualBlock,
    ncsn_conv3x3,
    ConditionalResidualBlock,
    get_act,
)
from .normalization import get_normalization
import ml_collections
from jax import numpy as jnp
from jax import grad
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from .layers import get_timestep_embedding

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3


def get_network(config):
    """Get the appropriate network architecture for the data."""
    if config.data.dataset == "GMM":
        return functools.partial(NCSNv2LowDim, config=config)
    elif config.data.image_size < 96:
        return functools.partial(NCSNv2, config=config)
    elif 96 <= config.data.image_size <= 128:
        return functools.partial(NCSNv2_128, config=config)
    elif 128 < config.data.image_size <= 256:
        return functools.partial(NCSNv2_256, config=config)
    else:
        raise NotImplementedError(
            f"No network suitable for {config.data.image_size}px implemented yet."
        )


@register_model(name="ncsnv2_lowdim")
class NCSNv2LowDim(nn.Module):
    """NCSNv2 model architecture for low-dimensional data."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        nf = config.model.nf
        conditional = config.model.conditional
        act = get_act(config)
        normalizer = get_normalization(config)
        sigmas = get_sigmas(config)
        use_fourier_emb = getattr(config.model, "use_fourier_emb", True)

        # h = x # we won't do any normalization here
        # Use vmap to get embeddings for each dimension of x
        dim = x.shape[-1]  # Get the dimensionality of the input
        assert nf // dim > 2, f"nf//dim > 2, got nf={nf}, dim={dim}"

        if use_fourier_emb:
            if config.data.num_channels > 2:
                embedding_fn = lambda x_i: get_timestep_embedding(x_i, nf // (dim))
            else:
                embedding_fn = lambda x_i: get_timestep_embedding(x_i, nf // (2 * dim))
            embeddings = jax.vmap(embedding_fn)(
                jnp.moveaxis(x, -1, 0)
            )  # Apply to each dimension
            h = jnp.concatenate([emb for emb in embeddings], axis=-1)
        else:
            h = x

        # Simple MLP architecture suitable for low-dimensional data
        h = nn.Dense(features=nf // 2)(h)
        if conditional:
            # Sinusoidal positional embeddings. This does not introduce any new parameters.
            emb = get_timestep_embedding(labels, nf // 2)
            # temb = temb.reshape(temb.shape[0], 1, 1, temb.shape[1])
            # assert temb.shape == h.shape, f'temb.shape={temb.shape} != h.shape={h.shape}'
        else:
            # emb0 = get_timestep_embedding(x[..., 0], nf//4) # we will use the input here, no "time"
            # emb1 = get_timestep_embedding(x[..., 1], nf//4) # we will use the input here, no "time"
            # emb = jnp.concatenate([emb0, emb1], axis=-1)
            emb = h

        h = jnp.concatenate([h, emb], axis=-1)  # concatenate time embeddings with input

        assert h.shape[-1] == nf

        h = h.reshape(h.shape[0], 1, 1, h.shape[-1])

        # Several residual blocks
        for _ in range(4):  # Number of blocks can be adjusted
            skip = h
            h = normalizer()(h)
            h = act(h)
            h = nn.Dense(features=nf)(h)
            h = normalizer()(h)
            h = act(h)
            h = nn.Dense(features=nf)(h)
            h = h + skip  # Residual connection

        # Final layers
        h = normalizer()(h)
        h = act(h)
        h = nn.Dense(features=x.shape[-1])(h)  # Output dimension matches input

        h = h.reshape(*x.shape)

        if config.model.scale_by_sigma:
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            # jax.debug.print('used_sigmas={used_sigmas}', used_sigmas=used_sigmas)
            return h / used_sigmas
        else:
            return h


@register_model(name="lowdim_transformer")
class LowDimTransformer(nn.Module):
    """Small transformer for low-dimensional vectors (D in [2, 100]).

    This module treats the D-dimensional input as a sequence of D tokens with
    scalar features. Each token is embedded to a d_model-dimensional vector,
    optionally conditioned on a diffusion time embedding, processed by several
    Transformer blocks, and projected back to scalars to match the input shape.

    Expected input/output shape: (batch, dim).
    """

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        config = self.config
        d_model = config.model.nf  # hidden size / token embedding size
        num_layers = getattr(config.model, "num_layers", 4)
        num_heads = getattr(config.model, "num_heads", 4)
        mlp_ratio = getattr(config.model, "mlp_ratio", 4)
        dropout = getattr(config.model, "dropout", 0.0)
        conditional = getattr(config.model, "conditional", True)
        sigmas = get_sigmas(config)

        assert x.ndim == 2, f"LowDimTransformer expects (batch, dim), got {x.shape}"
        batch, dim = x.shape
        assert (
            d_model >= num_heads and d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Token embedding: scalar -> d_model per dimension
        # Shape: (B, D, d_model)
        token_emb = nn.Dense(features=d_model, name="token_embed")
        h = token_emb(x[..., None])

        # Learned positional embeddings per dimension
        pos_emb = self.param(
            "pos_emb", nn.initializers.normal(stddev=0.02), (dim, d_model)
        )
        h = h + pos_emb[None, ...]

        # Optional time conditioning (additive)
        if conditional and labels is not None:
            t_emb = get_timestep_embedding(labels, d_model)  # (B, d_model)
            h = h + t_emb[:, None, :]

        # Transformer blocks (Pre-LN)
        for i in range(num_layers):
            # Self-attention block
            y = nn.LayerNorm(name=f"ln1_{i}")(h)
            y = nn.SelfAttention(
                num_heads=num_heads,
                qkv_features=d_model,
                out_features=d_model,
                dropout_rate=dropout,
                deterministic=not train,
                name=f"self_attn_{i}",
            )(y)
            h = h + y

            # MLP block
            y = nn.LayerNorm(name=f"ln2_{i}")(h)
            y = nn.Dense(d_model * mlp_ratio, name=f"mlp_fc1_{i}")(y)
            y = jax.nn.gelu(y)
            if dropout > 0:
                y = nn.Dropout(rate=dropout, name=f"mlp_drop_{i}")(
                    y, deterministic=not train
                )
            y = nn.Dense(d_model, name=f"mlp_fc2_{i}")(y)
            h = h + y

        # Final projection back to scalar per token
        h = nn.LayerNorm(name="ln_out")(h)
        out = nn.Dense(features=1, name="head")(h)
        out = out.squeeze(-1)  # (B, D)

        if getattr(config.model, "scale_by_sigma", False):
            used_sigmas = sigmas[labels].reshape((x.shape[0], *([1] * (out.ndim - 1))))
            return out / used_sigmas
        else:
            return out


@register_model(name="lowdim_transformer_ebm")
class LowDimTransformerEBM(nn.Module):
    """EBM wrapper using LowDimTransformer energy: E(x) = ||f(x)||^2 (negative sign inside).

    Returns either the gradient of the negative energy w.r.t. x (default),
    or the batched negative energy values if return_only_neg_energy=True.
    """

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True, return_only_neg_energy=False):
        def energy_fn(x_single, labels_single, train_flag):
            nn_out = LowDimTransformer(self.config)(
                x_single[None, ...],
                labels_single[None, ...] if labels_single is not None else None,
                train_flag,
            ).reshape(-1)
            energy = -jnp.sum(nn_out**2)
            return energy.squeeze()

        neg_energy = lambda x_single, labels_single: -energy_fn(
            x_single, labels_single, train
        )
        if return_only_neg_energy:
            return jax.vmap(neg_energy)(x, labels)

        neg_energy_grad = jax.grad(neg_energy)
        neg_energy_grads = jax.vmap(neg_energy_grad)(x, labels)
        return neg_energy_grads


@register_model(name="ncsnv2_lowdim_ebm")
class NCSNv2LowDimEBM(nn.Module):
    """NCSNv2 model architecture for low-dimensional data with EBM-style training."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True, return_only_neg_energy=False):
        # jax.debug.print('outer x.shape={x.shape}', x=x)
        def energy_fn(x, labels, train):
            # jax.debug.print('inner x.shape={x.shape}', x=x.shape)
            nn_out = NCSNv2LowDim(self.config)(
                x[None, ...], labels[None, ...] if labels is not None else None, train
            ).reshape(-1)
            energy = -jnp.sum(nn_out**2)  # - \| net(x) \|_2^2
            return energy.squeeze()  # make sure we return a scalar, not a (1,) array

        neg_energy = lambda x_single, labels_single: -energy_fn(
            x_single, labels_single, train
        )
        if return_only_neg_energy:
            return jax.vmap(neg_energy)(x, labels)

        neg_energy_grad = jax.grad(neg_energy)

        # vmap over batch dim
        neg_energy_grads = jax.vmap(neg_energy_grad)(x, labels)
        return neg_energy_grads


@register_model(name="ncsnv2_64")
class NCSNv2(nn.Module):
    """NCSNv2 model architecture."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        nf = config.model.nf
        act = get_act(config, get_wrapper_fn=True)
        normalizer = get_normalization(config)
        sigmas = get_sigmas(config)
        interpolation = config.model.interpolation
        positive = config.model.positive

        if not config.data.centered:
            h = 2 * x - 1.0
        else:
            h = x

        h = conv3x3(h, nf, stride=1, bias=True, positive=positive)
        # ResNet backbone
        h = ResidualBlock(
            nf, resample=None, act=act, normalization=normalizer, positive=positive
        )(h)
        layer1 = ResidualBlock(
            nf, resample=None, act=act, normalization=normalizer, positive=positive
        )(h)
        h = ResidualBlock(
            2 * nf,
            resample="down",
            act=act,
            normalization=normalizer,
            positive=positive,
        )(layer1)
        layer2 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer, positive=positive
        )(h)
        h = ResidualBlock(
            2 * nf,
            resample="down",
            act=act,
            normalization=normalizer,
            dilation=2,
            positive=positive,
        )(layer2)
        layer3 = ResidualBlock(
            2 * nf,
            resample=None,
            act=act,
            normalization=normalizer,
            dilation=2,
            positive=positive,
        )(h)
        h = ResidualBlock(
            2 * nf,
            resample="down",
            act=act,
            normalization=normalizer,
            dilation=4,
            positive=positive,
        )(layer3)
        layer4 = ResidualBlock(
            2 * nf,
            resample=None,
            act=act,
            normalization=normalizer,
            dilation=4,
            positive=positive,
        )(h)
        # U-Net with RefineBlocks
        ref1 = RefineBlock(
            layer4.shape[1:3],
            2 * nf,
            act=act,
            interpolation=interpolation,
            start=True,
            positive=positive,
        )([layer4])
        ref2 = RefineBlock(
            layer3.shape[1:3],
            2 * nf,
            interpolation=interpolation,
            act=act,
            positive=positive,
        )([layer3, ref1])
        ref3 = RefineBlock(
            layer2.shape[1:3],
            2 * nf,
            interpolation=interpolation,
            act=act,
            positive=positive,
        )([layer2, ref2])
        ref4 = RefineBlock(
            layer1.shape[1:3],
            nf,
            interpolation=interpolation,
            act=act,
            positive=positive,
            end=True,
        )([layer1, ref3])

        h = normalizer()(ref4)
        activation = act(h.shape[-1])
        h = activation(h)
        h = conv3x3(h, x.shape[-1], positive=positive)

        # When using the DDPM loss, no need of normalizing the output
        if config.model.scale_by_sigma:
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            return h / used_sigmas
        else:
            return h


@register_model(name="ncsn")
class NCSN(nn.Module):
    """NCSNv1 model architecture."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        normalizer = get_normalization(config, conditional=True)
        sigmas = get_sigmas(config)
        interpolation = config.model.interpolation

        if not config.data.centered:
            h = 2 * x - 1.0
        else:
            h = x

        h = conv3x3(h, nf, stride=1, bias=True)
        # ResNet backbone
        h = CondResidualBlock(nf, resample=None, act=act, normalization=normalizer)(
            h, labels
        )
        layer1 = CondResidualBlock(
            nf, resample=None, act=act, normalization=normalizer
        )(h, labels)
        h = CondResidualBlock(
            2 * nf, resample="down", act=act, normalization=normalizer
        )(layer1, labels)
        layer2 = CondResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h, labels)
        h = CondResidualBlock(
            2 * nf, resample="down", act=act, normalization=normalizer, dilation=2
        )(layer2, labels)
        layer3 = CondResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer, dilation=2
        )(h, labels)
        h = CondResidualBlock(
            2 * nf, resample="down", act=act, normalization=normalizer, dilation=4
        )(layer3, labels)
        layer4 = CondResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer, dilation=4
        )(h, labels)
        # U-Net with RefineBlocks
        ref1 = CondRefineBlock(
            layer4.shape[1:3],
            2 * nf,
            act=act,
            normalizer=normalizer,
            interpolation=interpolation,
            start=True,
        )([layer4], labels)
        ref2 = CondRefineBlock(
            layer3.shape[1:3],
            2 * nf,
            normalizer=normalizer,
            interpolation=interpolation,
            act=act,
        )([layer3, ref1], labels)
        ref3 = CondRefineBlock(
            layer2.shape[1:3],
            2 * nf,
            normalizer=normalizer,
            interpolation=interpolation,
            act=act,
        )([layer2, ref2], labels)
        ref4 = CondRefineBlock(
            layer1.shape[1:3],
            nf,
            normalizer=normalizer,
            interpolation=interpolation,
            act=act,
            end=True,
        )([layer1, ref3], labels)

        h = normalizer()(ref4, labels)
        h = act(h)
        h = conv3x3(h, x.shape[-1])

        # When using the DDPM loss, no need of normalizing the output
        if config.model.scale_by_sigma:
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            return h / used_sigmas
        else:
            return h


@register_model(name="ncsnv2_128")
class NCSNv2_128(nn.Module):  # pylint: disable=invalid-name
    """NCSNv2 model architecture for 128px images."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        normalizer = get_normalization(config)
        sigmas = get_sigmas(config)
        interpolation = config.model.interpolation

        if not config.data.centered:
            h = 2 * x - 1.0
        else:
            h = x

        h = conv3x3(h, nf, stride=1, bias=True)
        # ResNet backbone
        h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
        layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
        h = ResidualBlock(2 * nf, resample="down", act=act, normalization=normalizer)(
            layer1
        )
        layer2 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h)
        h = ResidualBlock(2 * nf, resample="down", act=act, normalization=normalizer)(
            layer2
        )
        layer3 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h)
        h = ResidualBlock(
            4 * nf, resample="down", act=act, normalization=normalizer, dilation=2
        )(layer3)
        layer4 = ResidualBlock(
            4 * nf, resample=None, act=act, normalization=normalizer, dilation=2
        )(h)
        h = ResidualBlock(
            4 * nf, resample="down", act=act, normalization=normalizer, dilation=4
        )(layer4)
        layer5 = ResidualBlock(
            4 * nf, resample=None, act=act, normalization=normalizer, dilation=4
        )(h)
        # U-Net with RefineBlocks
        ref1 = RefineBlock(
            layer5.shape[1:3], 4 * nf, interpolation=interpolation, act=act, start=True
        )([layer5])
        ref2 = RefineBlock(
            layer4.shape[1:3], 2 * nf, interpolation=interpolation, act=act
        )([layer4, ref1])
        ref3 = RefineBlock(
            layer3.shape[1:3], 2 * nf, interpolation=interpolation, act=act
        )([layer3, ref2])
        ref4 = RefineBlock(layer2.shape[1:3], nf, interpolation=interpolation, act=act)(
            [layer2, ref3]
        )
        ref5 = RefineBlock(
            layer1.shape[1:3], nf, interpolation=interpolation, act=act, end=True
        )([layer1, ref4])

        h = normalizer()(ref5)
        h = act(h)
        h = conv3x3(h, x.shape[-1])

        if config.model.scale_by_sigma:
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            return h / used_sigmas
        else:
            return h


@register_model(name="ncsnv2_256")
class NCSNv2_256(nn.Module):  # pylint: disable=invalid-name
    """NCSNv2 model architecture for 256px images."""

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, labels, train=True):
        # config parsing
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        normalizer = get_normalization(config)
        sigmas = get_sigmas(config)
        interpolation = config.model.interpolation

        if not config.data.centered:
            h = 2 * x - 1.0
        else:
            h = x

        h = conv3x3(h, nf, stride=1, bias=True)
        # ResNet backbone
        h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
        layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
        h = ResidualBlock(2 * nf, resample="down", act=act, normalization=normalizer)(
            layer1
        )
        layer2 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h)
        h = ResidualBlock(2 * nf, resample="down", act=act, normalization=normalizer)(
            layer2
        )
        layer3 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h)
        h = ResidualBlock(2 * nf, resample="down", act=act, normalization=normalizer)(
            layer3
        )
        layer31 = ResidualBlock(
            2 * nf, resample=None, act=act, normalization=normalizer
        )(h)
        h = ResidualBlock(
            4 * nf, resample="down", act=act, normalization=normalizer, dilation=2
        )(layer31)
        layer4 = ResidualBlock(
            4 * nf, resample=None, act=act, normalization=normalizer, dilation=2
        )(h)
        h = ResidualBlock(
            4 * nf, resample="down", act=act, normalization=normalizer, dilation=4
        )(layer4)
        layer5 = ResidualBlock(
            4 * nf, resample=None, act=act, normalization=normalizer, dilation=4
        )(h)
        # U-Net with RefineBlocks
        ref1 = RefineBlock(
            layer5.shape[1:3], 4 * nf, interpolation=interpolation, act=act, start=True
        )([layer5])
        ref2 = RefineBlock(
            layer4.shape[1:3], 2 * nf, interpolation=interpolation, act=act
        )([layer4, ref1])
        ref31 = RefineBlock(
            layer31.shape[1:3], 2 * nf, interpolation=interpolation, act=act
        )([layer31, ref2])
        ref3 = RefineBlock(
            layer3.shape[1:3], 2 * nf, interpolation=interpolation, act=act
        )([layer3, ref31])
        ref4 = RefineBlock(layer2.shape[1:3], nf, interpolation=interpolation, act=act)(
            [layer2, ref3]
        )
        ref5 = RefineBlock(
            layer1.shape[1:3], nf, interpolation=interpolation, act=act, end=True
        )([layer1, ref4])

        h = normalizer()(ref5)
        h = act(h)
        h = conv3x3(h, x.shape[-1])

        if config.model.scale_by_sigma:
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            return h / used_sigmas
        else:
            return h
