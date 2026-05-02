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
"""Config file for training NCSNv2 on MNIST with Importance Sampling."""

from configs.default_mnist_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = False
  training.eval_freq = 5000
  training.fixed_sigma_is = 0.01 # if this is set, we'll use the importance-sampling trick

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'none'
  sampling.corrector = 'ald' # annealed Langevin dynamics for sampling
  sampling.n_steps_each = 5
  sampling.snr = 0.15 # typical values are 0.05 to 0.2, not sure what to pick for mnist
  # model
  model = config.model
  model.sigma_min = 0.01 # this should be at least as large as fixed_sigma_is (if used)
  model.sigma_max = 5.
  model.num_scales = 100

  model.name = 'ncsnv2_64'
  model.scale_by_sigma = False # we'll use the importance-sampling trick
  model.ema_rate = 0.999
  model.normalization = 'InstanceNorm++'
  model.positive = False
  # model.nonlinearity = '3way_lrelu'
  model.nonlinearity = 'lrelu'
  model.nf = 16
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
