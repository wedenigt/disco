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
"""Config file for reproducing the results of DDPM on cifar-10."""

from configs.default_gmm_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.fixed_sigma_is = None

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # model
  model = config.model
  model.sigma_min = 0.1
  model.sigma_max = 2.0
  model.num_scales = 100
  
  # Using the low-dimensional network
  model.name = 'ncsnv2_lowdim' 
  model.scale_by_sigma = False # in DDPM, we scale by 1/sigma in the *loss* function, not in the model
  model.ema_rate = 0.999
  model.normalization = 'None'
  model.positive = False
  model.nonlinearity = 'lrelu'
  model.nf = 256  # Increased since we have fewer layers
  model.interpolation = 'bilinear'

    # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = -1.

  return config
