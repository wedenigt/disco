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
"""Training NCSN++ on CIFAR-10 with DISCO."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training

  training = config.training
  training.batch_size = 192
  training.n_jitted_steps = 5
  training.sde = 'vesde'
  training.continuous = False
  training.fixed_sigma_is = 0.01 # DISCO uses this
  training.weight_free_disco = True # use weight-free DISCO
  training.ebm = False # train an EBM

  # sampling
  sampling = config.sampling
  sampling.method = 'edm'
  sampling.rho = 7
  sampling.s_churn = 80
  sampling.s_min = 0
  sampling.s_max = 50
  sampling.s_noise = 1.007
  sampling.num_steps = 18
  sampling.inv_temp_scaler = 1.0

  # model
  model = config.model
  model.sigma_min = 0.01
  model.sigma_max = 50 # at 10, ~50% of the weights are 0. at 50, ~95% are 0. on average, linspace(0.01, 10, 1000) -> 13% of weights are 0.
  model.num_scales = 1000

  model.name = 'ncsnpp'
  model.loss_variant = 'None' # this must be a string such that we can override it on the command line
  model.explicit_sigma_pred = False # If set to True, we explicitly predict the noise level and return \sigma_\theta(x) * \eps_\theta(x)
  model.scale_by_sigma = False # we do not scale by the noise level in DISCO
  model.net_output = 'sigma_times_eps' # default for DISCO

  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = False # we do not condition on the noise level in DISCO
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
