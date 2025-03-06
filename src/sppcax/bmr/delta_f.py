"""Functions for computing changes in variational free energy for different distributions."""
from typing import Any

import jax.numpy as jnp
from jax.scipy.special import gammaln
from multipledispatch import dispatch

from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array


@dispatch(object, object)
def compute_delta_f(posterior: Any, prior: Any, indices=None) -> Array:
    """Compute change in variational free energy when switching off parameters.

    Args:
        posterior: Posterior distribution
        prior: Prior distribution
        indices: Indices of parameters to consider for pruning. If None, all parameters are considered.

    Returns:
        Array of delta F values for each parameter
    """
    raise NotImplementedError(f"No implementation for {type(posterior)} and {type(prior)}")


@dispatch(MultivariateNormalGamma, MultivariateNormalGamma, object)
def compute_delta_f(posterior: MultivariateNormalGamma, prior: MultivariateNormalGamma, indices=None) -> Array:
    """Compute delta F for MultivariateNormalGamma distribution (used for loading matrix W).

    For each parameter, calculates the change in free energy that would result
    from setting its prior precision to infinity (effectively pruning the parameter).

    Args:
        posterior: Posterior MultivariateNormalGamma distribution
        prior: Prior MultivariateNormalGamma distribution
        indices: Indices of parameters to consider for pruning. If None, all parameters are considered.

    Returns:
        Array of delta F values for each parameter
    """
    # Extract parameters
    post_mean = posterior.mvn.mean
    post_prec = posterior.mvn.precision
    post_alpha = posterior.gamma.alpha
    post_beta = posterior.gamma.beta

    prior_mean = prior.mvn.mean
    prior_prec = prior.mvn.precision
    prior_alpha = prior.gamma.alpha
    prior_beta = prior.gamma.beta

    # If indices not provided, consider all parameters
    if indices is None:
        indices = jnp.ndindex(post_mean.shape)

    # Calculate delta F for each parameter
    results = []
    for idx in indices:
        # Free energy with current model (F1)
        F1 = _free_energy_term_mvn_gamma(
            idx, post_mean, post_prec, post_alpha, post_beta, prior_mean, prior_prec, prior_alpha, prior_beta
        )

        # Free energy with pruned parameter (F0)
        # When pruning, we set the prior precision to effectively infinity for that parameter
        pruned_prior_prec = prior_prec.at[idx].set(jnp.inf)
        F0 = _free_energy_term_mvn_gamma(
            idx, post_mean, post_prec, post_alpha, post_beta, prior_mean, pruned_prior_prec, prior_alpha, prior_beta
        )

        # Delta F = F1 - F0
        delta_f = F1 - F0
        results.append((idx, delta_f))

    return jnp.array(results)


def _free_energy_term_mvn_gamma(
    idx, post_mean, post_prec, post_alpha, post_beta, prior_mean, prior_prec, prior_alpha, prior_beta
):
    """Calculate free energy contribution for a specific parameter in MVN-Gamma distribution."""
    # Implement specific free energy calculation for MVN-Gamma
    # This is a simplified version - actual implementation would need complete derivation

    # Extract parameter-specific values
    mu_post = post_mean[idx]
    lambda_post = post_prec[idx, idx]
    alpha_post = post_alpha[idx[0]]  # Assuming alpha is per-feature
    beta_post = post_beta[idx[0]]  # Assuming beta is per-feature

    mu_prior = prior_mean[idx]
    lambda_prior = prior_prec[idx, idx]
    alpha_prior = prior_alpha[idx[0]]
    beta_prior = prior_beta[idx[0]]

    # Calculate log evidence terms
    # This is just illustrative - real implementation needs full Bayesian math
    kl_divergence = (
        0.5 * jnp.log(lambda_post / lambda_prior)
        - 0.5
        + 0.5 * lambda_prior / lambda_post
        + 0.5 * lambda_post * (mu_post - mu_prior) ** 2 / lambda_prior
    )

    # Gamma component of KL
    kl_gamma = (
        (alpha_post - alpha_prior) * jnp.digamma(alpha_post)
        - gammaln(alpha_post)
        + gammaln(alpha_prior)
        + alpha_prior * jnp.log(beta_post / beta_prior)
        + alpha_post * (beta_prior - beta_post) / beta_post
    )

    # Combine terms
    free_energy = kl_divergence + kl_gamma

    return free_energy
