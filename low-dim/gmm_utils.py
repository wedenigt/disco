import jax
import jax.numpy as jnp
from tqdm import tqdm
from ot import emd2  # optimal transport library
from jax import config as jax_config
from jax.scipy.linalg import solve
from jax import jit


def mvn_logpdf(x, mu, cov):
    """
    Batched log-density of multivariate normal for fixed x:
      x:   shape (Do,)
      mu:  shape (K, Do)
      cov: shape (K, Do, Do)
    returns shape (K,)
    """
    K, Do = mu.shape
    # Compute Cholesky factors L_i such that cov_i = L_i @ L_i^T
    L = jnp.linalg.cholesky(cov)  # (K,Do,Do)
    diff = x - mu  # (K,Do)
    # solve L y = diff^T  => y = L^{-1} diff^T
    y = solve(L, diff[..., None], lower=True)  # (K,Do,1)
    maha = jnp.sum(y**2, axis=(1, 2))  # (K,)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)), axis=1)
    return -0.5 * (maha + Do * jnp.log(2 * jnp.pi) + logdet)


def get_conditional_gmm_params(
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
    cond_values: jnp.ndarray,
    cond_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Given a GMM defined by (weights, means, covs), compute the conditional GMM
    parameters for the given conditional values.
    """
    K, D = means.shape
    obs = cond_indices
    unobs = ~cond_indices
    Do = obs.sum()
    Du = unobs.sum()
    x_o = cond_values[obs]  # (Do,)

    # Slice component-wise parameters
    mu_o = means[:, obs]  # (K, Do)
    mu_u = means[:, unobs]  # (K, Du)
    Σ_oo = covs[:, obs][:, :, obs]  # (K, Do, Do)
    Σ_uo = covs[:, unobs][:, :, obs]  # (K, Du, Do)
    Σ_ou = covs[:, obs][:, :, unobs]  # (K, Do, Du)
    Σ_uu = covs[:, unobs][:, :, unobs]  # (K, Du, Du)

    # Precompute inverse of Σ_oo for each component
    inv_Σ_oo = jnp.linalg.inv(Σ_oo)  # (K, Do, Do)

    # Conditional means and covariances for each component
    diff = (x_o - mu_o)[..., None]  # (K, Do, 1)
    cond_mu = mu_u + jnp.matmul(Σ_uo, jnp.matmul(inv_Σ_oo, diff))[..., 0]  # (K, Du)
    cond_cov = Σ_uu - jnp.matmul(Σ_uo, jnp.matmul(inv_Σ_oo, Σ_ou))  # (K, Du, Du)

    # Component log-weights ∝ log prior + log p(x_o | comp)
    log_p_xo = mvn_logpdf(x_o, mu_o, Σ_oo)  # (K,)
    log_w = jnp.log(weights) + log_p_xo  # (K,)
    w_post = jax.nn.softmax(log_w)  # (K,)

    return w_post, cond_mu, cond_cov


def sample_conditional_gmm(
    key,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
    cond_values: jnp.ndarray,
    cond_indices: jnp.ndarray,
):
    """
    Sample x ~ p(x_u | x_o = cond_values[cond_indices])
    from a GMM defined by (weights, means, covs).
    """
    K, D = means.shape
    obs = cond_indices
    unobs = ~cond_indices
    Do = obs.sum()
    Du = unobs.sum()
    x_o = cond_values[obs]  # (Do,)

    w_post, cond_mu, cond_cov = get_conditional_gmm_params(
        means, covs, weights, cond_values, cond_indices
    )

    # Sample component
    key, subkey = jax.random.split(key)
    comp = jax.random.choice(subkey, a=K, p=w_post)

    # Sample from the chosen conditional Gaussian
    key, subkey = jax.random.split(key)
    L = jnp.linalg.cholesky(cond_cov[comp])  # (Du, Du)
    z = jax.random.normal(subkey, shape=(Du,))  # (Du,)
    x_u = cond_mu[comp] + L @ z  # (Du,)

    # Reconstruct full sample
    x = jnp.empty((D,))
    x = x.at[obs].set(x_o)
    x = x.at[unobs].set(x_u)

    return x


def eval_gmm(x, means, covs, weights):
    """
    Evaluates the GMM distribution at a given point.
    """
    log_pdfs = jax.vmap(
        jax.scipy.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0)
    )(x, means, covs)
    return jax.scipy.special.logsumexp(jnp.log(weights) + log_pdfs, axis=0)


def eval_gmm_conditional(x, means, covs, weights, cond_values, cond_indices):
    """
    Evaluates the conditional GMM distribution at a given point.
    """
    w_post, cond_mu, cond_cov = get_conditional_gmm_params(
        means, covs, weights, cond_values, cond_indices
    )
    return eval_gmm(x, cond_mu, cond_cov, w_post)


def gmm_responsibilities(x, means, covs, weights):
    """
    Computes the responsibilities of a GMM for a given point.
    """
    log_pdfs = jax.vmap(
        jax.scipy.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0)
    )(
        x, means, covs
    )  # p(x | z)
    log_pdfs = log_pdfs + jnp.log(weights)  # p(x, z) = p(z) p(x | z)
    log_total = jax.scipy.special.logsumexp(log_pdfs, axis=0)  # p(x)
    return jnp.exp(log_pdfs - log_total)  # p(z | x)


def compute_wasserstein_distance(samples, true_data, num_iter_max=100_000):
    assert (
        samples.shape == true_data.shape
    ), "Samples and true data must have the same shape"

    jax_config.update("jax_enable_x64", True)
    # Compute 2D Wasserstein-1 distance using the Earth Mover's Distance
    # First compute pairwise distances between all points using JAX on GPU
    samples_final = jnp.array(samples)  # Move to GPU
    true_data_gpu = jnp.array(true_data)  # Move to GPU

    # Compute all pairwise distances efficiently using broadcasting
    diff = (
        samples_final[:, None, :] - true_data_gpu[None, :, :]
    )  # Shape: (n_samples, n_true, 2)
    M = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # Shape: (n_samples, n_true)

    # Uniform weights for both distributions
    a = jnp.ones(len(samples)) / len(samples)
    b = jnp.ones(len(true_data)) / len(true_data)

    w1_2d = emd2(a, b, M, numItermax=num_iter_max)
    jax_config.update("jax_enable_x64", False)
    return w1_2d


def sliced_wasserstein(
    samples, true_data, key, M=2 * 4096, p=1, no_slicing=False, num_iter_max=1_000_000
):
    """
    samples, true_data: (n, d) with the same n
    key:  jax.random.PRNGKey
    M:    number of random directions
    returns: scalar, representing Sliced p-Wasserstein distance
    """
    assert p in [1, 2], "Only supports 1-Wasserstein and 2-Wasserstein."
    X = jnp.asarray(samples, jnp.float32)
    Y = jnp.asarray(true_data, jnp.float32)
    n, d = X.shape

    if no_slicing:
        return compute_wasserstein_distance(X, Y, num_iter_max=num_iter_max)

    @jit
    def _sliced_wasserstein(X, Y, key):
        # sample M unit directions, i.e., points on the sphere
        dirs = jax.random.normal(key, (M, d))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)  # (M,d)

        # project and sort along each direction
        U = jnp.sort(X @ dirs.T, axis=0)  # (n, M)
        V = jnp.sort(Y @ dirs.T, axis=0)  # (n, M)

        if p == 1:
            # 1D W1 per direction (mean absolute diff), then average directions
            dist = jnp.mean(jnp.mean(jnp.abs(U - V), axis=0))
        else:
            # # 1D W2^2 per direction then average over directions; finally sqrt
            w2_sq = jnp.mean(jnp.mean((U - V) ** 2, axis=0))
            dist = jnp.sqrt(jnp.maximum(w2_sq, 0.0))

        return dist

    return _sliced_wasserstein(X, Y, key)


def get_is_weights(xs, xts, std=0.001, std_prime=3.0, log=False):
    """Computes importance sampling weights based on posterior ratios.

    Args:
        xs: Array of shape (N, D) containing N data points of dimension D.
            These are the original, unperturbed data points.
        xts: Array of shape (N, D) containing N perturbed data points.
            These should be noisy versions of xs.
        std: Float, standard deviation for the small noise distribution.
        std_prime: Float, standard deviation for the large noise distribution.

    Returns:
        weights: Array of shape (N,) containing importance sampling weights
            computed as exp(log p(x|xt,std) - log p(x|xt,std_prime)) for each
            data point, where p(x|xt,std) is the posterior probability under
            a Gaussian noise model with standard deviation std.
    """

    def compute_posterior_ratio(xs, x_idx, xt, std):
        """Computes the log ratio of posterior probabilities for a given data point.

        Args:
            xs: Array of shape (N, D) containing N data points of dimension D
            x_idx: Integer index of the reference point in xs that produced xt
            xt: Array of shape (D,) containing the perturbed point \tilde{x}
            std: Float standard deviation of the distribution

        Returns:
            ratio: Float containing log(p(xs[x_idx]|xt)/sum_i p(xs[i]|xt)) where p(x|xt)
                is the Gaussian probability of x given xt with variance std^2
        """
        diff = xs - xt
        log_pdf = -0.5 * jnp.sum(diff**2, axis=1) / (std**2)

        # Compute logsumexp of log probabilities
        log_sum_gaussians = jax.scipy.special.logsumexp(log_pdf)
        x_log_pdf = log_pdf[x_idx]

        ratio = x_log_pdf - log_sum_gaussians
        return ratio

    post_ratio_fn = lambda x_idx, xt, sigma: compute_posterior_ratio(
        xs, x_idx, xt, std=sigma
    )

    # # Vectorize post_ratio_std over both x_idx and xts
    post_ratios_std = jax.vmap(post_ratio_fn, in_axes=(0, 0, None))(
        jnp.arange(xts.shape[0]), xts, std
    )
    std_prime = (
        jnp.repeat(std_prime, xts.shape[0])
        if isinstance(std_prime, float)
        else std_prime
    )
    post_ratios_std_prime = jax.vmap(post_ratio_fn, in_axes=(0, 0, 0))(
        jnp.arange(xts.shape[0]), xts, std_prime
    )
    log_weights = post_ratios_std - post_ratios_std_prime

    return log_weights if log else jnp.exp(log_weights)


def eval_log_gmm(xs, means, covs, weights, perturb_std=None):
    """Evaluate log probability of a Gaussian Mixture Model.

    Args:
        xs: Array of shape (B, D) containing B D-dimensional points
        means: Array of shape (k, D) containing k component means
        covs: Array of shape (k, D, D) containing k covariance matrices
        weights: Array of shape (k,) containing mixture weights
        perturb_std: Optional standard deviation to add to covariances

    Returns:
        Array of shape (B,) containing log probabilities
    """
    # Input validation
    B, D = xs.shape
    k, D2 = means.shape
    # Check dimension match
    jax.lax.cond(
        D != D2,
        lambda: jax.debug.print(
            "Dimension mismatch: xs has dim {} but means has dim {}", D, D2
        ),
        lambda: None,
    )

    # Check covariance shape
    jax.lax.cond(
        jnp.any(covs.shape != (k, D, D)),
        lambda: jax.debug.print(
            "Expected covs shape ({}, {}, {}), got {}", k, D, D, covs.shape
        ),
        lambda: None,
    )

    # Check weights shape
    jax.lax.cond(
        jnp.any(weights.shape != (k,)),
        lambda: jax.debug.print(
            "Expected weights shape ({},), got {}", k, weights.shape
        ),
        lambda: None,
    )

    # Check weights sum
    jax.lax.cond(
        jnp.abs(jnp.sum(weights) - 1.0) >= 1e-6,
        lambda: jax.debug.print("Weights must sum to approximately 1"),
        lambda: None,
    )

    if perturb_std is not None:
        covs = (
            covs + perturb_std**2 * jnp.eye(D)[None, :, :]
        )  # Broadcasting to (k, D, D)

    # Vectorized computation over components
    log_prob_fn = lambda m, c: jax.scipy.stats.multivariate_normal.logpdf(xs, m, c)
    log_probs = jax.vmap(log_prob_fn)(means, covs)  # Shape: (k, B)

    # Add log weights and compute logsumexp
    log_probs = log_probs + jnp.log(weights)[:, None]  # Shape: (k, B)
    return jax.scipy.special.logsumexp(log_probs, axis=0)  # Shape: (B,)


def get_gmm_score(means, covs, weights):
    """Returns function that computes ∇ₓ log p(x) for a Gaussian Mixture Model (GMM).
    When calling score, if sigma is not None, the covariance matrices are perturbed by sigma²I.
    This then corresponds to ∇ₓ log p_σ(x).

    Args:
        xs: Array of shape (B, D) containing B D-dimensional points
        means: Array of shape (k, D) containing k component means
        covs: Array of shape (k, D, D) containing k covariance matrices
        weights: Array of shape (k,) containing mixture weights
        perturb_std: Optional standard deviation to add to covariances

    Returns:
        Array of shape (B,) containing log probabilities
    """
    # Input validation
    k, D = means.shape

    # Check shapes
    jax.lax.cond(
        jnp.any(covs.shape != (k, D, D)),
        lambda: jax.debug.print(
            "Expected covs shape ({}, {}, {}), got {}", k, D, D, covs.shape
        ),
        lambda: None,
    )

    jax.lax.cond(
        jnp.any(weights.shape != (k,)),
        lambda: jax.debug.print(
            "Expected weights shape ({},), got {}", k, weights.shape
        ),
        lambda: None,
    )

    # Check weights sum
    jax.lax.cond(
        jnp.abs(jnp.sum(weights) - 1.0) >= 1e-6,
        lambda: jax.debug.print("Weights must sum to approximately 1"),
        lambda: None,
    )

    # Check diagonal covariances
    # diag = jnp.diagonal(covs, axis1=1, axis2=2)

    # For some reason, this is not working. Not sure why.
    # jax.lax.cond(
    #     (jnp.abs(diag.reshape(-1) - diag.reshape(-1)[0]) > 1e-6).sum() > 0,
    #     lambda: jax.debug.print("Covariances must be diagonal, got {}", diag.reshape(-1)),
    #     lambda: None
    # )

    def score_fn(x, sigma=None):
        if sigma is not None:
            covs_perturbed = (
                covs + sigma**2 * jnp.eye(D)[None, :, :]
            )  # Broadcasting to (k, D, D)
        else:
            covs_perturbed = covs

        # For diagonal covariances, we only need the diagonal elements
        inv_vars = 1.0 / jnp.diagonal(covs_perturbed, axis1=1, axis2=2)  # Shape: (k, D)
        log_det = jnp.sum(
            jnp.log(jnp.diagonal(covs_perturbed, axis1=1, axis2=2)), axis=1
        )  # Shape: (k,)

        # Compute differences and squared mahalanobis distances
        diff = x[None, :] - means  # Shape: (k, D)

        # Use jax.scipy.stats.multivariate_normal.logpdf for each component
        log_prob_fn = lambda m, c: jax.scipy.stats.multivariate_normal.logpdf(x, m, c)
        log_probs = jax.vmap(log_prob_fn)(means, covs_perturbed)  # Shape: (k,)
        log_probs = log_probs + jnp.log(weights)  # Shape: (k,)
        # Compute responsibilities (posterior probabilities)
        log_total = jax.scipy.special.logsumexp(log_probs)
        responsibilities = jnp.exp(log_probs - log_total)  # Shape: (k,)

        # Compute score as weighted sum of component-wise scores
        # For diagonal covariances, score is simply -diff * inv_vars
        component_scores = -diff * inv_vars  # Shape: (k, D)
        score = jnp.sum(
            responsibilities[:, None] * component_scores, axis=0
        )  # Shape: (D,)

        return score

    return score_fn


def get_gmm_score_vmap(means, covs, weights, sigma=None):
    score_fn = get_gmm_score(means, covs, weights)
    return jax.vmap(lambda x: score_fn(x, sigma))


def test_gmm_score():
    """Test that our analytical GMM score matches JAX's autograd."""
    # Create a simple test GMM
    D = 2  # dimension
    k = 20  # number of components
    rng = jax.random.PRNGKey(0)

    # Generate random parameters
    means = jax.random.normal(rng, (k, D))
    covs = jax.random.uniform(rng, (k, D)) * 0.1  # diagonal elements
    covs = jnp.stack([jnp.diag(cov) for cov in covs])  # make diagonal matrices
    weights = jax.random.uniform(rng, (k,))
    weights = weights / jnp.sum(weights)  # normalize weights

    # Generate test points
    num_points = 100
    test_points = jax.random.normal(rng, (num_points, D))

    # Get our analytical score function

    # Define GMM log probability for autograd
    def gmm_log_prob(x, sigma=None):
        return eval_log_gmm(x[None, :], means, covs, weights, perturb_std=sigma)[0]

    score_fn = get_gmm_score(means, covs, weights)
    # Get autograd score function
    autograd_score_fn = jax.grad(gmm_log_prob)

    # Compare scores for different sigma values
    sigma_values = [None, 0.01, 0.1, 1.0]
    max_errors = []

    for sigma in sigma_values:
        errors = []
        for x in test_points:
            analytical = score_fn(x, sigma)
            automatic = autograd_score_fn(x, sigma)
            error = jnp.max(jnp.abs(analytical - automatic))
            errors.append(error)

        max_error = jnp.max(jnp.array(errors))
        max_errors.append(max_error)
        print(f"Maximum error for sigma={sigma}: {max_error:.2e}")

    # Assert that errors are small
    assert all(
        err < 1e-3 for err in max_errors
    ), "Large discrepancy between analytical and automatic differentiation"
    print("All tests passed!")


if __name__ == "__main__":
    test_gmm_score()
