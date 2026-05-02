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
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from jax import random


def sample_checkerboard_shifted(
    key: jax.Array,
    n: int,
    *,
    tiles_x: int = 8,
    tiles_y: int = 8,
    parity: str = "even",  # which tiles are "filled": (i + j) % 2 == 0 or 1
    shift_frac: float = 0.5,  # horizontal shift (in tile widths) for every odd row
) -> jnp.ndarray:
    """
    Samples exactly `n` points from a staggered-row checkerboard in [-1, 1]^2.
    - Filled tiles satisfy (i + j) % 2 == p, where p=0 ('even') or 1 ('odd').
    - Every odd row (j % 2 == 1) is shifted by `shift_frac * tile_width`.
    - No rejection; uniform within each chosen tile; no vertical corridors.

    Args:
        key: jax.random.PRNGKey
        n: number of samples
        tiles_x, tiles_y: number of tiles per axis (even recommended)
        parity: 'even' or 'odd'
        shift_frac: row shift as a fraction of tile width (0.5 = brick pattern)

    Returns:
        (n, 2) samples in [-1, 1]^2
    """
    if tiles_x < 2 or tiles_y < 2:
        raise ValueError("tiles_x and tiles_y must be >= 2.")
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'.")
    if not (0.0 <= shift_frac < 1.0):
        raise ValueError("shift_frac must be in [0, 1).")

    key_j, key_a, key_u, key_v = random.split(key, 4)
    p = 0 if parity == "even" else 1

    # --- 1) Sample ALL rows uniformly: j in {0,...,tiles_y-1}
    j = random.randint(key_j, (n,), 0, tiles_y)

    # --- 2) Enforce (i + j) % 2 == p by choosing i's parity from j
    # Desired i_parity = (p - (j % 2)) mod 2
    j_par = j & 1
    i_par = (p ^ j_par).astype(jnp.int32)  # XOR == addition mod 2

    # Sample the "half-grid" column index a, then lift to full-grid i = 2*a + i_par
    half_x = (tiles_x + 1) // 2  # works for even/odd tiles_x
    a = random.randint(key_a, (n,), 0, half_x)
    i = 2 * a + i_par
    # If tiles_x is odd, i can equal tiles_x on the last a when i_par=1; clip it.
    i = jnp.minimum(i, tiles_x - 1)

    # --- 3) Geometry
    w = 2.0 / tiles_x
    h = 2.0 / tiles_y

    # Uniform offsets inside the tile
    u = random.uniform(key_u, (n,), minval=0.0, maxval=1.0)
    v = random.uniform(key_v, (n,), minval=0.0, maxval=1.0)

    # Base corners
    x0 = -1.0 + i.astype(jnp.float32) * w
    y0 = -1.0 + j.astype(jnp.float32) * h

    # --- 4) Shift every odd row horizontally by shift_frac * w (then wrap)
    row_is_odd = (j & 1).astype(jnp.float32)
    x = x0 + row_is_odd * (shift_frac * w) + u * w
    y = y0 + v * h

    # Wrap on torus to keep all points in [-1, 1]^2 (robust near boundaries)
    x = -1.0 + jnp.mod(x + 1.0, 2.0)
    y = -1.0 + jnp.mod(y + 1.0, 2.0)

    return jnp.stack([x, y], axis=-1)


def sample_rings_jax(
    key: jax.Array,
    n: int,
    *,
    c: float = 0.2,  # constant spacing between successive radii
    r1: float | None = None,  # inner radius; if None we set r4=r_max and back off
    r_max: float = 0.95,  # ensure r4 <= r_max < 1 to stay inside [-1,1]^2
    probs=(
        0.4,
        0.3,
        0.2,
        0.1,
    ),  # priors for rings 1..4 (inner→outer); will be normalized
    radial_std: float = 0.0,  # radial thickness (0 = infinitesimally thin ring)
    enforce_proportions: bool = True,  # if True, match exact counts to probs
    scale: float = 2.0,  # scale the points to [-scale, scale]^2
) -> jnp.ndarray:
    """
    Return exactly `n` samples from 4 concentric rings centered at the origin, all within [-1,1]^2.

    Radii: r_i = r1 + (i-1)*c, i=1..4, with r4 <= r_max < 1.
    Priors: probs (inner→outer). If enforce_proportions, the sample counts per ring
            match probs exactly (up to rounding).

    Args:
        key: jax.random.PRNGKey
        n: number of points
        c: spacing between rings
        r1: inner radius; if None, set r4=r_max and r1=r_max-3c
        r_max: maximum allowed radius to ensure bounds
        probs: length-4 nonnegative ring weights (inner→outer)
        radial_std: std dev of radial noise (Gaussian along radial direction)
        enforce_proportions: control exact counts per ring

    Returns:
        (n,2) JAX array of (x,y) samples in [-1,1]^2.
    """
    probs = jnp.array(probs, dtype=jnp.float32)
    probs = probs / jnp.sum(probs)
    if jnp.any(probs < 0) or probs.shape[0] != 4:
        raise ValueError("`probs` must be length-4 and nonnegative.")
    if c <= 0:
        raise ValueError("`c` must be > 0.")
    if r1 is None:
        r1 = r_max - 3.0 * c
    if r1 <= 0 or (r1 + 3.0 * c) > r_max or r_max >= 1.0:
        raise ValueError("Invalid radii: need 0 < r1 and r1+3c <= r_max < 1.")

    r = jnp.array([r1 + k * c for k in range(4)], dtype=jnp.float32)  # [r1, r2, r3, r4]

    # Decide how many points per ring
    if enforce_proportions:
        # Deterministic rounding with largest-remainder method
        raw = probs * n
        k_floor = jnp.floor(raw).astype(jnp.int32)
        rem = n - jnp.sum(k_floor)
        # assign leftover to the largest fractional parts
        frac = raw - k_floor.astype(jnp.float32)
        order = jnp.argsort(-frac)  # descending
        add = (
            jnp.zeros(4, dtype=jnp.int32).at[order[:rem]].add(1)
            if rem > 0
            else jnp.zeros(4, dtype=jnp.int32)
        )
        counts = k_floor + add
    else:
        # Multinomial draw (random counts)
        (key_cnt,) = random.split(key, 1)
        counts = random.multinomial(key_cnt, n=n, p=probs)

    # Prepare per-ring sampling
    keys = random.split(key, 1 + 4)  # [main_unused, k1, k2, k3, k4]
    thetas = []
    radii = []
    for idx in range(4):
        m = counts[idx]
        if m == 0:
            continue
        k_theta, k_eps = random.split(keys[idx + 1])
        theta = random.uniform(k_theta, (m,), minval=0.0, maxval=2.0 * jnp.pi)
        # radial noise along radial direction (kept small so we stay within r_max)
        eps = random.normal(k_eps, (m,)) * radial_std
        ri = jnp.clip(r[idx] + eps, a_min=0.0, a_max=r_max)
        thetas.append(theta)
        radii.append(ri)

    if len(radii) == 0:
        return jnp.empty((0, 2), dtype=jnp.float32)

    theta_all = jnp.concatenate(thetas)
    r_all = jnp.concatenate(radii)

    x = r_all * jnp.cos(theta_all)
    y = r_all * jnp.sin(theta_all)

    # All points are guaranteed inside the unit disk with r_max < 1 ⇒ inside [-scale,scale]^2.
    return scale * jnp.stack([x, y], axis=-1)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        jax.debug.print("returning inverse scaler")
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def create_random_gmm_params(k, seed=42, dim=2):
    """Create random parameters for a GMM with k components.

    Args:
        k: Number of GMM components
        seed: Random seed for reproducibility
        dim: Dimension of the GMM (default: 2)

    Returns:
        means: Array of shape (k, dim) containing component means
        covs: Array of shape (k, dim, dim) containing component covariances
        weights: Array of shape (k,) containing component weights
    """
    np.random.seed(seed)

    # Generate random means in [-4, 4]^dim
    means = np.random.uniform(-4, 4, (k, dim))

    # Generate random covariance matrices
    covs = []
    for _ in range(k):
        A = np.random.randn(dim, dim)
        cov = np.dot(A, A.T) / 10  # Scale down to get reasonable spreads
        covs.append(cov)
    covs = np.array(covs)

    # Generate random weights and normalize
    weights = np.random.rand(k)
    weights = weights / weights.sum()

    # sample from the GMM
    samples = sample_gmm(means, covs, weights, 10_000, dim=dim)
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)

    means = scaler.transform(means)  # change the means to the normalized space
    A = np.diag(1.0 / scaler.scale_)
    covs = A.T @ covs @ A

    return means, covs, weights


def sample_gmm(means, covs, weights, n_samples, dim=2, rng=None):
    """Generate samples from a GMM with the given parameters.

    Args:
        means: Array of shape (k, 2) containing component means
        covs: Array of shape (k, 2, 2) containing component covariances
        weights: Array of shape (k,) containing component weights
        n_samples: Number of samples to generate

    Returns:
        samples: Array of shape (n_samples, dim)
    """
    if rng is None:
        rng = np.random.default_rng(1)

    k = len(weights)
    # Choose components based on weights
    components = rng.choice(k, size=n_samples, p=weights)

    # Generate samples
    samples = np.zeros((n_samples, dim))
    for i in range(k):
        mask = components == i
        n_comp = mask.sum()
        if n_comp > 0:
            samples[mask] = rng.multivariate_normal(means[i], covs[i], size=n_comp)

    return samples


def prepare_2d_dataset(
    config,
    name,
    batch_dims,
    num_epochs,
    shuffle_buffer_size,
    prefetch_size,
    standardize=False,
    rng_key=None,
):
    if rng_key is None:
        rng_key = 42

    def resize_op(img):
        return img  # no need to resize, we're working with 2D points

    # Create a synthetic moons dataset
    n_samples = getattr(config.data, "samples_per_epoch", 10000)
    noise = getattr(config.data, "noise", 0.1)
    test_size = getattr(config.data, "test_size", 0.2)
    dim = getattr(config.data, "num_channels", 2)  # data dimension

    # Generate the moons data
    if name == "Moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=rng_key)
    elif name == "Checkerboard":
        key = random.PRNGKey(rng_key)
        X = sample_checkerboard_shifted(
            key, n_samples, tiles_x=4, tiles_y=4, parity="even", shift_frac=0.0
        )
    elif name == "Rings":
        key = random.PRNGKey(rng_key)
        X = sample_rings_jax(
            key,
            n_samples,
            c=0.18,
            r1=None,
            r_max=0.95,
            probs=(0.5, 0.25, 0.15, 0.10),
            radial_std=0.01,
            enforce_proportions=True,
        )
    else:
        raise ValueError(f"Dataset {name} not supported")

    if standardize:
        # Standardize the data using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    # Split into train and test sets
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=rng_key)

    def preprocess_fn(sample):
        """Convert GMM samples to the expected format."""
        # Reshape to match expected format (treating 2D points as 1x1 images with dim channels)
        img = tf.reshape(sample, (dim,))
        return dict(image=img, label=None)

    # Create synthetic dataset builder (just a placeholder for compatibility)
    class TwoDBaseBuilder:
        def __init__(self):
            self.name = name

        def as_dataset(self, split="train", shuffle_files=True, read_config=None):
            if split == "train":
                ds = tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32))
            else:
                ds = tf.data.Dataset.from_tensor_slices(X_test.astype(np.float32))

            if read_config and hasattr(read_config, "options"):
                ds = ds.with_options(read_config.options)
            return ds

    dataset_builder = TwoDBaseBuilder()
    train_split_name = "train"
    eval_split_name = "test"

    train_ds = create_custom_dataset(
        dataset_builder,
        train_split_name,
        preprocess_fn,
        batch_dims,
        num_epochs,
        shuffle_buffer_size,
        prefetch_size,
    )
    eval_ds = create_custom_dataset(
        dataset_builder,
        eval_split_name,
        preprocess_fn,
        batch_dims,
        num_epochs,
        shuffle_buffer_size,
        prefetch_size,
    )

    return train_ds, eval_ds, dataset_builder


def get_dataset(
    config,
    additional_dim=None,
    uniform_dequantization=False,
    evaluation=False,
    return_gmm_params=False,
    rng_key=None,  # if supplied, we'll generate the data using this key. should only be used for eval purposes, not during training.
):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
        which equals the number of steps jitted together.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    per_device_batch_size = batch_size // jax.device_count()
    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    # Create additional data dimension when jitting multiple steps together
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

    # Create dataset builders for each dataset.
    if config.data.dataset == "MNIST":
        dataset_builder = tfds.builder("mnist")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "CIFAR10":
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "Moons":
        # We standardize moons
        train_ds, eval_ds, dataset_builder = prepare_2d_dataset(
            config,
            config.data.dataset,
            batch_dims,
            num_epochs,
            shuffle_buffer_size,
            prefetch_size,
            standardize=True,
            rng_key=rng_key,
        )
        return train_ds, eval_ds, dataset_builder
    elif config.data.dataset == "Checkerboard":
        train_ds, eval_ds, dataset_builder = prepare_2d_dataset(
            config,
            config.data.dataset,
            batch_dims,
            num_epochs,
            shuffle_buffer_size,
            prefetch_size,
            standardize=False,
            rng_key=rng_key,
        )
        return train_ds, eval_ds, dataset_builder
    elif config.data.dataset == "Rings":
        train_ds, eval_ds, dataset_builder = prepare_2d_dataset(
            config,
            config.data.dataset,
            batch_dims,
            num_epochs,
            shuffle_buffer_size,
            prefetch_size,
            standardize=False,
            rng_key=rng_key,
        )
        return train_ds, eval_ds, dataset_builder
    elif config.data.dataset == "SVHN":
        dataset_builder = tfds.builder("svhn_cropped")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "CELEBA":
        dataset_builder = tfds.builder("celeb_a")
        train_split_name = "train"
        eval_split_name = "validation"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == "LSUN":
        dataset_builder = tfds.builder(f"lsun/{config.data.category}")
        train_split_name = "train"
        eval_split_name = "validation"

        if config.data.image_size == 128:

            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:

            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ["FFHQ", "CelebAHQ"]:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = eval_split_name = "train"

    elif config.data.dataset == "GMM":

        def resize_op(img):
            return img  # GMM is already 2D, no need to resize

        # For GMM, we'll create a synthetic dataset
        k = getattr(
            config.data, "gmm_components", 5
        )  # Default to 5 components if not specified
        samples_per_epoch = getattr(config.data, "samples_per_epoch", 10000)
        dim = getattr(config.data, "num_channels", 2)  # data dimension
        means, covs, weights = create_random_gmm_params(k, dim=dim, seed=rng_key)

        def create_tf_dataset(n_samples=None, rng=None):
            """Create a TensorFlow dataset from GMM samples."""
            if n_samples is None:
                n_samples = samples_per_epoch
            samples = sample_gmm(means, covs, weights, n_samples, dim=dim, rng=rng)
            # Convert to tensorflow dataset
            ds = tf.data.Dataset.from_tensor_slices(samples.astype(np.float32))
            return ds

        def preprocess_fn(sample):
            """Convert GMM samples to the expected format."""
            # Reshape to match expected format (treating 2D points as 1x1 images with dim channels)
            img = tf.reshape(sample, (dim,))
            return dict(image=img, label=None)

        # Create synthetic dataset builder (just a placeholder for compatibility)
        class GMMBuilder:
            def __init__(self):
                self.name = "GMM"

            def as_dataset(self, split="train", shuffle_files=True, read_config=None):
                n_samples = (
                    samples_per_epoch if split == "train" else samples_per_epoch // 5
                )
                rng = (
                    np.random.default_rng(1)
                    if split == "train"
                    else np.random.default_rng(2)
                )
                ds = create_tf_dataset(n_samples, rng=rng)
                if read_config and hasattr(read_config, "options"):
                    ds = ds.with_options(read_config.options)
                return ds

        dataset_builder = GMMBuilder()
        train_split_name = "train"
        eval_split_name = "test"

        train_ds = create_custom_dataset(
            dataset_builder,
            train_split_name,
            preprocess_fn,
            batch_dims,
            num_epochs,
            shuffle_buffer_size,
            prefetch_size,
        )
        eval_ds = create_custom_dataset(
            dataset_builder,
            eval_split_name,
            preprocess_fn,
            batch_dims,
            num_epochs,
            shuffle_buffer_size,
            prefetch_size,
        )

        if return_gmm_params:
            return train_ds, eval_ds, dataset_builder, means, covs, weights
        else:
            return train_ds, eval_ds, dataset_builder

    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ["FFHQ", "CelebAHQ"]:

        def preprocess_fn(d):
            sample = tf.io.parse_single_example(
                d,
                features={
                    "shape": tf.io.FixedLenFeature([3], tf.int64),
                    "data": tf.io.FixedLenFeature([], tf.string),
                },
            )
            data = tf.io.decode_raw(sample["data"], tf.uint8)
            data = tf.reshape(data, sample["shape"])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0
            return dict(image=img, label=None)

    else:

        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d["image"])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0

            return dict(image=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=True, read_config=read_config
            )
        else:
            ds = dataset_builder.with_options(dataset_options)

        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        for batch_size in reversed(batch_dims):
            ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder


# Override the create_dataset function for GMM
def create_custom_dataset(
    dataset_builder,
    split,
    preprocess_fn,
    batch_dims,
    num_epochs,
    shuffle_buffer_size,
    prefetch_size,
):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)

    ds = dataset_builder.as_dataset(split=split, read_config=read_config)
    ds = ds.repeat(count=num_epochs)
    # Use a fixed seed for shuffling based on the split
    seed = 42 if split == "train" else 43
    ds = ds.shuffle(shuffle_buffer_size, seed=seed)
    ds = ds.map(
        preprocess_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    )
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)
