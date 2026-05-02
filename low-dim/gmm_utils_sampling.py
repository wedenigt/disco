import jax
import jax.numpy as jnp
from tqdm import tqdm
import os
from pathlib import Path
import glob
from datetime import datetime


def get_latest_checkpoints(checkpoint_path, n=5):
    """
    Get the n most recent checkpoint directories.
    
    Args:
        checkpoint_path (Path): Path to the checkpoint directory
        n (int): Number of most recent checkpoints to return
        
    Returns:
        list: List of Path objects for the n most recent checkpoint directories
    """
    # Get all directories in checkpoint path
    checkpoint_dirs = [d for d in checkpoint_path.glob("*") if d.is_dir() and len(list(d.glob("config.yml"))) > 0]
    
    # Sort directories by modification time (most recent first)
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Get the n most recent directories
    return [c.name for c in checkpoint_dirs[:n]]


def conditional_gmm(means, covs, weights, cond_x1):
    """
    Computes the conditional GMM distribution of the second RV, conditioned on the first RV.
    
    Args:
        means: Array of shape (K, 2) containing the means of K components
        covs: Array of shape (K, 2, 2) containing the covariance matrices
        weights: Array of shape (K,) containing the mixture weights
        cond_x1: Float, the value of x1 to condition on
    
    Returns:
        cond_means: Array of shape (K,) for the conditional means of x2|x1
        cond_vars: Array of shape (K,) for the conditional variances of x2|x1
        cond_weights: Array of shape (K,) for the updated mixture weights
    """
    K = len(weights)
    
    # Compute conditional means and variances for each component
    cond_means = jnp.zeros(K)
    cond_vars = jnp.zeros(K)
    
    for k in range(K):
        # Extract components for current mixture
        mu = means[k]
        sigma = covs[k]
        
        # Conditional mean: mu2 + sigma12/sigma11 * (x1 - mu1)
        cond_means = cond_means.at[k].set(
            mu[1] + sigma[1,0]/sigma[0,0] * (cond_x1 - mu[0])
        )
        
        # Conditional variance: sigma22 - sigma12^2/sigma11
        cond_vars = cond_vars.at[k].set(
            sigma[1,1] - sigma[1,0]**2/sigma[0,0]
        )
    
    # Update weights based on likelihood of x1 under each component
    log_probs = -0.5 * jnp.log(2*jnp.pi*covs[:,0,0]) - \
                0.5 * (cond_x1 - means[:,0])**2/covs[:,0,0]
    log_probs = log_probs + jnp.log(weights)
    
    # Normalize weights
    cond_weights = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs))
    
    return cond_means, cond_vars, cond_weights

def eval_gmm(x, means, covs, weights):
    """
    Evaluates the GMM distribution at a given point. x is of shape (2,)
    """
    # log_pdfs = jax.vmap(lambda m, c: jax.scipy.stats.multivariate_normal.logpdf(x, m, c))(means, covs)

    log_pdfs = jax.vmap(jax.scipy.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0))(x, means, covs)
    return jax.scipy.special.logsumexp(jnp.log(weights) + log_pdfs, axis=0)

def eval_cond_gmm(x, means, covs, weights, cond_x1, perturb_sigma=jnp.zeros(2)):
    """
    Evaluates the conditional GMM distribution at a given point. x is of shape (2,)
    """
    covs = covs + jnp.diag(perturb_sigma**2)
    cond_means, cond_vars, cond_weights = conditional_gmm(means, covs, weights, cond_x1)
    cond_means = cond_means[:, None]
    cond_vars = cond_vars[:, None, None]
    return eval_gmm(x, cond_means, cond_vars, cond_weights)