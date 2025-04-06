"""Functions for computing changes in variational free energy for different distributions."""

import jax.numpy as jnp
from jax import lax, nn, vmap
from jax import random as jr
from jax.scipy.special import gammaln

from ..distributions import MultivariateNormal
from ..models.factor_analysis_params import PFA
from ..types import Array, Matrix, PRNGKey, Scalar, Tuple, Vector


def ln_c(alpha: Vector, beta: Vector, r: Vector):
    return gammaln(alpha - r / 2) - gammaln(alpha) + r * jnp.log(beta) / 2


def compute_delta_f(pruned_d, lm_mean_d, lm_prec_d, alpha_d, beta_d, ln_c) -> Array:
    delta_f = ln_c
    G = jnp.diag(pruned_d)
    L = jnp.linalg.cholesky(G @ lm_prec_d @ G + jnp.diag(1 - pruned_d))
    tilde_mu = L.mT @ (pruned_d * lm_mean_d)

    delta_f += jnp.log(jnp.diag(L)).sum()

    delta_f += alpha_d * jnp.log(beta_d)
    delta_f -= alpha_d * jnp.log(beta_d + jnp.inner(tilde_mu, tilde_mu) / 2)

    return delta_f


def gibbs_sampler_pfa(key: PRNGKey, model: PFA, pi: Scalar, lam: Matrix, delta_f: Vector) -> Tuple[Vector, Matrix]:
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
    eta = jnp.log(pi) - jnp.log(1 - pi)
    D, K = lam.shape

    def step_fn(carry, k):
        lam, delta_f, key = carry

        _lam_k = lam[:, k]
        lam = mask * lam.at[:, k].set(~_lam_k)
        pruned = mask.astype(jnp.int8) - lam
        r = pruned.sum(0)
        ln_c_d = ln_c(posterior.gamma.alpha, posterior.gamma.beta, r).sum() / D
        delta_f_d = vmap(compute_delta_f, in_axes=(0, 0, 0, 0, 0, None))(
            pruned, lm_mean, lm_prec, rho.alpha, rho.beta, ln_c_d
        )

        tmp = delta_f_d - delta_f
        delta_f_d_iim1 = jnp.where(_lam_k == 1, -tmp, tmp)

        p = nn.sigmoid(delta_f_d_iim1 + eta[k])
        key, _key = jr.split(key)
        lam_k = jr.bernoulli(key, p=p) * mask[:, k]
        lam = lam.at[:, k].set(lam_k)

        delta_f = jnp.where(lam_k == _lam_k, delta_f, delta_f_d)
        return (lam, delta_f, key), None

    init = (lam, delta_f, key)
    (new_lam, new_delta_f, _), _ = lax.scan(step_fn, init, jnp.arange(K))

    return new_delta_f, new_lam


def compute_delta_f_mvn(pruned, mean, prec) -> Array:
    G = jnp.diag(pruned)
    L = jnp.linalg.cholesky(G @ prec @ G + jnp.diag(1 - pruned))
    tilde_mu = L.mT @ (pruned * mean)

    delta_f = jnp.log(jnp.diag(L)).sum()
    delta_f -= jnp.inner(tilde_mu, tilde_mu) / 2

    return delta_f


def gibbs_sampler_mvn(
    key: PRNGKey, mvn_dist: MultivariateNormal, pi: Scalar, lam: Matrix, delta_f: Vector
) -> Tuple[Vector, Matrix]:
    """Sample sparsity constrains based on the local change in the variational free energy.

    For each element of the multivariate normal distribution, calculates the change in free energy that would result
    from setting its prior mean to zero and prior precision to infinity (effectively pruning the parameter).

    Args:
        mvn_dist: MultivariateNormal distribution
        pi: Prior probability of element being pruned out.
        lam: Current sparisty matrix
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix

    Returns:
        A tuple of new delta F values for each dimension d, and new sparsity constrains
    """

    # Extract parameters
    mask = mvn_dist.mask  # initial mask over loading matrix
    lm_mean = mvn_dist.mean  # posterior mean
    lm_prec = mvn_dist.precision  # posterior precision
    eta = jnp.log(pi) - jnp.log(1 - pi)
    _, D = mask.shape

    def step_fn(carry, k):
        lam, delta_f, key = carry

        _lam_k = lam[:, k]
        lam = mask * lam.at[:, k].set(~_lam_k)
        pruned = mask.astype(jnp.int8) - lam
        delta_f_d = vmap(compute_delta_f_mvn)(pruned, lm_mean, lm_prec)

        tmp = delta_f_d - delta_f
        delta_f_d_iim1 = jnp.where(_lam_k == 1, -tmp, tmp)

        p = nn.sigmoid(delta_f_d_iim1 + eta[k])
        key, _key = jr.split(key)
        lam_k = jr.bernoulli(key, p=p) * mask[:, k]
        lam = lam.at[:, k].set(lam_k)

        delta_f = jnp.where(lam_k == _lam_k, delta_f, delta_f_d)
        return (lam, delta_f, key), None

    init = (lam, delta_f, key)
    (new_lam, new_delta_f, _), _ = lax.scan(step_fn, init, jnp.arange(D))

    return new_delta_f, new_lam
