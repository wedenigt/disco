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
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import math
from typing import Any, Dict, Optional, TypeVar

import flax
import jax
import jax.numpy as jnp
from PIL import Image
import tensorflow as tf
import numpy as np
import jax.random as random
from gmm_utils import get_gmm_score_vmap

T = TypeVar("T")


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def load_training_state(filepath, state):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        state = flax.serialization.from_bytes(state, f.read())
    return state


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and save it into an image file.

    Pixel values are assumed to be within [0, 1].

    Args:
      ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
      fp: A filename(string) or file object.
      nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
      padding (int, optional): amount of padding. Default: ``2``.
      pad_value (float, optional): Value for the padded pixels. Default: ``0``.
      format(Optional):  If omitted, the format to use is determined from the
        filename extension. If a file object was used instead of a filename, this
        parameter should always be used.
    """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and
             all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError("array_like of tensors expected, got {}".format(
            type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                         padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                          x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    im = Image.fromarray(np.array(ndarr.copy()))
    im.save(fp, format=format)


def flatten_dict(config):
    """Flatten a hierarchical dict to a simple dict."""
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_dict = flatten_dict(value)
            for subkey, subvalue in sub_dict.items():
                new_dict[key + "/" + subkey] = subvalue
        elif isinstance(value, tuple):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict


def weighted_gmm_fisher_divergence(eval_data, score_fn_model, vesde,
                                   means, covs, weights, rng, only_sigma_min=False, 
                                   max_sigma_index=None):
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    sigma_min = smld_sigma_array[-1]
    assert sigma_min < smld_sigma_array[0], "sigma_min must be less than sigma_max"

    if only_sigma_min:
        sigmas = jnp.ones(eval_data.shape[0]) * sigma_min
    else:
        N = max_sigma_index if max_sigma_index is not None else vesde.N

        rng, step_rng = random.split(rng)
        labels = vesde.N - random.choice(step_rng, N, shape=(eval_data.shape[0],))
        # labels = jnp.zeros_like(labels) + N
        sigmas = smld_sigma_array[labels]
        # jax.debug.print("sigmas: {}", sigmas)

    rng, step_rng = random.split(rng)
    eps = random.normal(step_rng, eval_data.shape)
    sigma_times_eps = batch_mul(eps, sigmas)
    perturbed_data = sigma_times_eps + eval_data

    score_fn_gmm = get_gmm_score_vmap(means, covs, weights, sigma=sigma_min)
    true_scores = score_fn_gmm(perturbed_data)
    # model scores at t=0
    t = jnp.zeros(eval_data.shape[0]) 
    model_scores = score_fn_model(perturbed_data, t)
    # model_scores = jnp.zeros_like(true_scores) # debug
    score_diff = model_scores - true_scores

    weighted_fisher_divergence = jnp.sum(score_diff**2, axis=-1)  # Squared L2 norm
    weighted_fisher_divergence = jnp.mean(weighted_fisher_divergence)  # Take expectation over samples

    # weighed cos sim
    normalized_model_scores = model_scores / jnp.linalg.norm(model_scores, axis=-1, keepdims=True)
    normalized_true_scores = true_scores / jnp.linalg.norm(true_scores, axis=-1, keepdims=True)
    weighted_cos_sim = jnp.sum(normalized_model_scores * normalized_true_scores, axis=-1)

    # weighted magnitude difference
    model_scores_norms = jnp.linalg.norm(model_scores, axis=-1)
    true_scores_norms = jnp.linalg.norm(true_scores, axis=-1)
    weighted_magnitude_difference = jnp.square(model_scores_norms - true_scores_norms)

    fisher_alt = weighted_magnitude_difference + 2 * model_scores_norms * true_scores_norms * (1 - weighted_cos_sim)

    weighted_cos_sim = jnp.mean(weighted_cos_sim)
    weighted_magnitude_difference = jnp.mean(weighted_magnitude_difference)

    # print(jnp.mean(fisher_alt), weighted_fisher_divergence)
    return weighted_fisher_divergence, weighted_cos_sim, weighted_magnitude_difference
