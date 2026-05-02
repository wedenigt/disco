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

"""Config file for training NCSNv2 on D-dimensional GMM data."""

from configs.default_gmm_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = False
    training.eval_freq = 1000
    training.fixed_sigma_is = 0.1  # DISCO uses this
    training.weight_free_disco = True  # use weight-free DISCO
    training.true_gmm_weights = False  # use true GMM weights
    training.ebm = False  # train an EBM
    training.use_full_dataset = (
        False  # use the full dataset for sampling from the posterior
    )
    training.sample_posterior = True  # Sample from the posterior (if False, we compute the posterior expectation)

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "none"
    sampling.corrector = "ald"
    sampling.n_steps_each = 5
    sampling.snr = 0.01

    # model
    model = config.model
    model.sigma_min = 0.1
    model.sigma_max = 2.0
    model.num_scales = 100

    # Using the low-dimensional network
    model.name = "ncsnv2_lowdim"  # Changed from ncsnv2_64
    model.scale_by_sigma = (
        False  # we'll use the importance-sampling trick, this must be turned off
    )
    model.ema_rate = 0.999
    model.normalization = "InstanceNorm++"
    model.positive = False
    model.nonlinearity = "lrelu"
    model.nf = 256  # Increased since we have fewer layers
    model.interpolation = "bilinear"
    model.loss_variant = (
        "None"  # this must be a string such that we can override it on the command line
    )
    model.conditional = False  # we do *not* condition on the noise level (the global minimizer is independent of the noise level)
    model.net_output = "sigma_times_eps"  # default for DISCO
    # optim
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.amsgrad = False
    optim.eps = 1e-8
    optim.warmup = 0
    optim.grad_clip = -1.0

    return config
