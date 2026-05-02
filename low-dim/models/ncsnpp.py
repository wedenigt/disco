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

from . import utils, layers, layerspp, normalization
import flax.linen as nn
import functools
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, time_cond, train=True):
    # config parsing
    config = self.config
    act = get_act(config)
    sigmas = utils.get_sigmas(config)

    nf = config.model.nf
    ch_mult = config.model.ch_mult
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    num_resolutions = len(ch_mult)

    explicit_sigma_pred = config.model.explicit_sigma_pred if 'explicit_sigma_pred' in config.model else False
    conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    skip_rescale = config.model.skip_rescale
    resblock_type = config.model.resblock_type.lower()
    progressive = config.model.progressive.lower()
    progressive_input = config.model.progressive_input.lower()
    embedding_type = config.model.embedding_type.lower()
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.model.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    if conditional:
      # timestep/noise_level embedding; only for continuous training
      if embedding_type == 'fourier':
        # Gaussian Fourier features embeddings.
        assert config.training.continuous, "Fourier features are only used for continuous training."
        used_sigmas = time_cond
        temb = layerspp.GaussianFourierProjection(
          embedding_size=nf,
          scale=config.model.fourier_scale)(jnp.log(used_sigmas))

      elif embedding_type == 'positional':
        # Sinusoidal positional embeddings.
        timesteps = time_cond
        used_sigmas = sigmas[time_cond.astype(jnp.int32)]
        temb = layers.get_timestep_embedding(timesteps, nf)
      else:
        raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
      temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
    else:
      temb = None

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    if not config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block

    input_pyramid = None
    if progressive_input != 'none':
      input_pyramid = x

    hs = [conv3x3(x, nf)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
        if h.shape[1] in attn_resolutions:
          h = AttnBlock()(h)
        hs.append(h)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          h = Downsample()(hs[-1])
        else:
          h = ResnetBlock(down=True)(hs[-1], temb, train)

        if progressive_input == 'input_skip':
          input_pyramid = pyramid_downsample()(input_pyramid)
          h = combiner()(input_pyramid, h)

        elif progressive_input == 'residual':
          input_pyramid = pyramid_downsample(out_ch=h.shape[-1])(input_pyramid)
          if skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = ResnetBlock()(h, temb, train)
    h = AttnBlock()(h)
    h = ResnetBlock()(h, temb, train)

    if explicit_sigma_pred:
      sigma_latent = conv3x3(act(h),
        out_planes=1, # single channel output
        bias=True,
        init_scale=init_scale)
      # flatten the output
      sigma_latent = jnp.reshape(sigma_latent, (sigma_latent.shape[0], -1))
      # feed through a linear layer
      sigma_log = nn.Dense(features=1)(sigma_latent)
      sigma = jnp.exp(jnp.reshape(sigma_log, (sigma_log.shape[0], -1))).squeeze(axis=-1)
      # jax.debug.print("sigma: {}", sigma)
    else:
      sigma = 1.

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(out_ch=nf * ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1),
                                                      temb,
                                                      train)

      if h.shape[1] in attn_resolutions:
        h = AttnBlock()(h)

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            pyramid = conv3x3(
              act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              x.shape[-1],
              bias=True,
              init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = conv3x3(
              act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              h.shape[-1],
              bias=True)
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            pyramid = pyramid_upsample()(pyramid)
            pyramid = pyramid + conv3x3(
              act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              x.shape[-1],
              bias=True,
              init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
            if skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          h = Upsample()(h)
        else:
          h = ResnetBlock(up=True)(h, temb, train)

    assert not hs

    if progressive == 'output_skip':
      h = pyramid
    else:
      h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
      h = conv3x3(h, x.shape[-1], init_scale=init_scale)

    if config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    if explicit_sigma_pred:
      h = h * sigma.reshape((x.shape[0], *([1] * len(x.shape[1:]))))

    return h


@utils.register_model(name='ncsnpp_ebm')
class NCSNppEBM(nn.Module):
    """NCSN++ model architecture wrapped into an EBM."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, time_cond, train=True, return_only_neg_energy=False):
        # x is a batch of images
        # time_cond is a batch of time steps
        def energy_fn(x, time_cond, train):
            # x is a single image
            # time_cond is a single time step
            nn_out = NCSNpp(self.config)(x[None, ...], time_cond[None, ...], train).reshape(-1)
            energy = -jnp.sum(nn_out**2) # - \| net(x) \|_2^2
            return energy.squeeze() # make sure we return a scalar, not a (1,) array

        neg_energy = lambda x_single, t_single: -energy_fn(x_single, t_single, train)
        if return_only_neg_energy:
            return jax.vmap(neg_energy)(x, time_cond)
        
        neg_energy_grad = jax.grad(neg_energy)

        # vmap over batch dim
        neg_energy_grads = jax.vmap(neg_energy_grad)(x, time_cond)
        return neg_energy_grads


@utils.register_model(name='ncsnpp_uedm_precondition')
class NCSNppUEDM(nn.Module):
    """NCSN++ model architecture wrapped into the uEDM preconditioner."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, used_sigmas, train=True):
        assert self.config.training.continuous, "uEDM preconditioner only works for continuous training."

        sigma_data = getattr(self.config.model, 'sigma_data', 0.5)
        # sigmas = utils.get_sigmas(self.config)
        # used_sigmas = sigmas[time_cond.astype(jnp.int32)]

        c_skip = sigma_data**2 / (used_sigmas**2 + sigma_data**2)
        # c_out = (used_sigmas * sigma_data) / jnp.sqrt(used_sigmas**2 + sigma_data**2)
        # c_in = 1 / jnp.sqrt(used_sigmas**2 + sigma_data**2) # EDM
        # c_noise = jnp.log(used_sigmas) / 4. # this is different from uEDM paper, but it's the same as the original EDM paper

        c_out = 1. # uEDM paper
        c_in = 1 / jnp.sqrt(used_sigmas**2 + 1) # uEDM paper
        c_noise = used_sigmas # uEDM paper

        F_x = NCSNpp(self.config)(c_in * x, c_noise, train)
        D_x = c_skip * x + c_out * F_x
        return D_x