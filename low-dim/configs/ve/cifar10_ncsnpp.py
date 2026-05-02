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

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with SMLD."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = False
  training.ebm = False # train an EBM
  training.uEDM = False # train as unconditional EDM model (see Sun et al., 2025)

  # # sampling
  # sampling = config.sampling
  # sampling.method = 'pc'
  # sampling.predictor = 'reverse_diffusion'
  # sampling.corrector = 'langevin'

  # sampling
  sampling = config.sampling
  sampling.method = 'edm'
  sampling.s_churn = 80
  sampling.s_min = 0
  sampling.s_max = 50
  sampling.s_noise = 1.007
  sampling.num_steps = 18
  sampling.inv_temp_scaler = 1.0

  # model
  model = config.model
  model.sigma_min = 0.01
  model.sigma_max = 50 # original paper trains with 50, but we use 10 to match DISCO
  model.num_scales = 1000
  model.net_output = 'eps'

  model.loss_variant = 'None' # this must be a string such that we can override it on the command line
  model.name = 'ncsnpp'
  model.scale_by_sigma = False # we do eps-prediction, so we do not scale by sigma
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.embedding_type = 'positional'
  model.conv_size = 3

  return config
