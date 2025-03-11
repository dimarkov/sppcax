"""Functions for computing changes in variational free energy for different distributions."""

import jax.numpy as jnp
from jax import lax, nn, vmap
from jax import random as jr
from jax.scipy.special import gammaln
from multipledispatch import dispatch

from ..models.factor_analysis_params import PFA
from ..types import Array, PRNGKey


def ln_c(alpha: Array, beta: Array):
    return gammaln(alpha - 1 / 2) - gammaln(alpha) + jnp.log(beta) / 2


def compute_delta_f(lam_d, lm_mean_d, lm_prec_d, alpha_d, beta_d, ln_c_k) -> Array:
    delta_f = jnp.inner(lam_d, ln_c_k)
    L = jnp.linalg.cholesky(lm_prec_d)
    tilde_mu = L.mT @ (lam_d * lm_mean_d)

    delta_f += jnp.inner(jnp.log(jnp.diag(L)), lam_d)

    delta_f += alpha_d * jnp.log(beta_d)
    delta_f -= alpha_d * jnp.log(beta_d + jnp.inner(tilde_mu, tilde_mu) / 2)

    return delta_f


@dispatch(PFA, float, Array, Array)
def gibbs_sampler(key: PRNGKey, model: PFA, pi: float, lam: Array, delta_f: Array) -> Array:
    """Sample sparsity matrix lambda based on the local change in the variational free energy.

    For each element in the loading matrix, calculates the change in free energy that would result
    from setting its prior mean to zero and prior precision to infinity (effectively pruning the parameter).

    Args:
        model: Probabilistic Factor Analysis
        pi: Prior probability of element being pruned out.
        lam: Current sparisty matrix
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix

    Returns:
        A tuple of new delta F values for each dimension d, and a new sparsity matrix
    """

    # MultivariateNormal-Gamma posterior q(W, \rho, \tau)
    posterior = model.W_dist
    rho = model.noise_precision

    # Extract parameters
    mask = posterior.mvn.mask  # initial mask over loading matrix
    lm_mean = posterior.mvn.mean  # loading matrix mean
    lm_prec = posterior.mvn.precision  # loading matrix precision
    ln_c_k = ln_c(posterior.gamma.alpha, posterior.gamma.beta)
    eta = jnp.log(pi) - jnp.log(1 - pi)
    D, K = lam.shape

    def step_fn(carry, k):
        lam, delta_f, key = carry

        _lam_k = lam[:, k]
        lam = lam.at[:, k].set(~_lam_k)
        delta_f_d = vmap(compute_delta_f, in_axes=(0, 0, 0, 0, 0, None))(
            lam, lm_mean, lm_prec, rho.alpha, rho.beta, ln_c_k
        )

        tmp = delta_f_d - delta_f
        delta_f_d_iim1 = jnp.where(_lam_k == 1, -tmp, tmp)

        p = nn.sigmoid(delta_f_d_iim1 + eta)
        key, _key = jr.split(key)
        lam_k = jr.bernoulli(key, p=p) * mask[:, k]
        lam = lam.at[:, k].set(lam_k)

        delta_f = jnp.where(lam_k == _lam_k, delta_f, delta_f_d)
        return (lam, delta_f, key), None

    init = (lam, delta_f, key)
    (new_lam, new_delta_f, _), _ = lax.step(step_fn, init, jnp.arange(K))

    return new_delta_f, new_lam


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
