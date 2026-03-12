"""Functions for computing changes in variational free energy for different distributions."""

import jax.numpy as jnp
from jax import lax, nn, vmap
from jax import random as jr
from ..distributions import MultivariateNormal as MVN, MultivariateNormalInverseGamma as MVNIG
from ..types import Array, Matrix, PRNGKey, Scalar, Tuple, Vector


def compute_delta_f(
    pruned_d, lm_mean_d, lm_prec_d, ln_c=0.0, alpha_d=None, beta_d=None, prior_prec_d=None, prior_mean_d=None
) -> Array:
    delta_f = ln_c
    G = jnp.diag(pruned_d)
    L = jnp.linalg.cholesky(G @ lm_prec_d @ G + jnp.diag(1 - pruned_d))
    tilde_mu = L.mT @ (pruned_d * lm_mean_d)

    delta_f += jnp.log(jnp.diag(L)).sum()

    if alpha_d is not None and beta_d is not None:
        delta_f += alpha_d * jnp.log(beta_d)

    tmp = jnp.inner(tilde_mu, tilde_mu) / 2
    if beta_d is not None:
        tmp += beta_d

    if prior_prec_d is not None:
        L = jnp.linalg.cholesky(G @ prior_prec_d @ G + jnp.diag(1 - pruned_d))
        tilde_mu = L.mT @ (pruned_d * prior_mean_d)
        tmp -= jnp.inner(tilde_mu, tilde_mu) / 2
        delta_f -= jnp.log(jnp.diag(L)).sum()

    if alpha_d is not None:
        delta_f -= alpha_d * jnp.log(tmp)
    else:
        delta_f -= tmp

    return delta_f


def gibbs_sampler_mvn(
    key: PRNGKey, post: MVN, pi: Scalar, lam: Matrix, delta_f: Vector, prior: MVN = None
) -> Tuple[Vector, Matrix]:
    """Sample sparsity matrix lambda based on the local change in the variational free energy.

    For each element in the loading matrix, calculates the change in free energy that would result
    from setting its prior mean to zero and prior precision to infinity (effectively pruning the parameter).

    Args:
        key: PRNGkey used for sampling
        poster: Posterior Multivariate Normal distribution
        prior: Prior Multivariate Normal distribution
        pi: Prior probability of element being pruned out.
        lam: Current sparisty matrix
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix

    Returns:
        A tuple of new delta F values for each dimension d, and a new sparsity matrix
    """

    # Extract parameters
    mask = prior.mask  # initial mask over loading matrix

    lm_mean = post.mean  # loading matrix mean
    lm_prec = post.precision  # loading matrix precision

    if prior is not None:
        lm_prior_mean = prior.mean  # loading matrix mean
        lm_prior_prec = prior.precision  # loading matrix precision

    eta = jnp.log(pi) - jnp.log(1 - pi)
    D, K = lam.shape

    def step_fn(carry, k):
        lam, delta_f, key = carry

        pruned = mask.astype(jnp.int8) * (1 - lam.at[:, k].set(~lam[:, k]))
        if prior is not None:
            delta_f_d = vmap(compute_delta_f)(
                pruned,
                lm_mean,
                lm_prec,
                prior_mean_d=lm_prior_mean,
                prior_prec_d=lm_prior_prec,
            )
        else:
            delta_f_d = vmap(compute_delta_f)(pruned, lm_mean, lm_prec)

        tmp = delta_f_d - delta_f
        delta_f_d_iim1 = jnp.where(lam[:, k] == 1, -tmp, tmp)

        p = nn.sigmoid(delta_f_d_iim1 + eta[k])
        key, _key = jr.split(key)
        lam_k = jr.bernoulli(key, p=p) * mask[:, k]
        delta_f = jnp.where(lam_k == lam[:, k], delta_f, delta_f_d)

        return (lam.at[:, k].set(lam_k), delta_f, key), None

    init = (lam, delta_f, key)
    (new_lam, new_delta_f, _), _ = lax.scan(step_fn, init, jnp.arange(K))

    return new_delta_f, new_lam


def gibbs_sampler_mvnig(
    key: PRNGKey, post: MVNIG, prior: MVNIG, pi: Scalar, lam: Matrix, delta_f: Vector
) -> Tuple[Vector, Matrix]:
    """Sample sparsity matrix lambda based on the local change in the variational free energy.

    For each element in the loading matrix, calculates the change in free energy that would result
    from setting its prior mean to zero and prior precision to infinity (effectively pruning the parameter).

    Args:
        key: PRNGkey used for sampling
        poster: Posterior MVNIG distribution
        prior: Prior mVNIG distribution
        pi: Prior probability of element being pruned out.
        lam: Current sparisty matrix
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix

    Returns:
        A tuple of new delta F values for each dimension d, and a new sparsity matrix
    """

    # Extract parameters
    mask = prior.mvn.mask  # initial mask over loading matrix

    lm_mean = post.mean  # loading matrix mean
    lm_prec = post.precision  # loading matrix precision

    lm_prior_mean = prior.mean  # loading matrix mean
    lm_prior_prec = prior.precision  # loading matrix precision
    eta = jnp.log(pi) - jnp.log(1 - pi)
    D, K = lam.shape

    # Broadcast alpha/beta to (D,) to handle both isotropic (scalar) and
    # per-dimension cases, so vmap always has a batch axis to map over.
    alpha_d = jnp.broadcast_to(post.inv_gamma.alpha, (D,))
    beta_d = jnp.broadcast_to(post.inv_gamma.beta, (D,))

    def step_fn(carry, k):
        lam, delta_f, key = carry

        pruned = mask.astype(jnp.int8) * (1 - lam.at[:, k].set(~lam[:, k]))
        delta_f_d = vmap(compute_delta_f)(
            pruned,
            lm_mean,
            lm_prec,
            alpha_d=alpha_d,
            beta_d=beta_d,
            prior_mean_d=lm_prior_mean,
            prior_prec_d=lm_prior_prec,
        )

        tmp = delta_f_d - delta_f
        delta_f_d_iim1 = jnp.where(lam[:, k] == 1, -tmp, tmp)

        p = nn.sigmoid(delta_f_d_iim1 + eta[k])
        key, _key = jr.split(key)
        lam_k = jr.bernoulli(key, p=p) * mask[:, k]
        delta_f = jnp.where(lam_k == lam[:, k], delta_f, delta_f_d)

        return (lam.at[:, k].set(lam_k), delta_f, key), None

    init = (lam, delta_f, key)
    (new_lam, new_delta_f, _), _ = lax.scan(step_fn, init, jnp.arange(K))

    return new_delta_f, new_lam
