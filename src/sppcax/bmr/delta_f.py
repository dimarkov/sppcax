"""Functions for computing changes in variational free energy for different distributions."""

import jax.numpy as jnp
from jax import lax, nn, vmap
from jax import random as jr
from ..distributions import MultivariateNormal as MVN, MultivariateNormalInverseGamma as MVNIG
from ..types import Matrix, PRNGKey, Scalar, Tuple, Vector


def compute_delta_f(
    pruned_d: Vector,
    lm_mean_d: Vector,
    lm_prec_d: Matrix,
    ln_c: Scalar = 0.0,
    alpha_d: Scalar | None = None,
    beta_d: Scalar | None = None,
    prior_prec_d: Matrix | None = None,
    prior_mean_d: Vector | None = None,
) -> Scalar:
    """Compute the change in variational free energy from pruning loading matrix elements.

    Evaluates the free energy difference when a subset of loading matrix elements
    (indicated by pruned_d) are set to zero for a single observation dimension d.

    Args:
        pruned_d: Binary vector indicating which elements are pruned (0) or active (1).
        lm_mean_d: Posterior mean of the loading matrix row d.
        lm_prec_d: Posterior precision matrix for row d.
        ln_c: Log-constant offset (default: 0.0).
        alpha_d: InverseGamma shape parameter for row d (None for MVN-only models).
        beta_d: InverseGamma scale parameter for row d (None for MVN-only models).
        prior_prec_d: Prior precision matrix for row d (None if no prior correction).
        prior_mean_d: Prior mean for row d (None if no prior correction).

    Returns:
        Scalar change in variational free energy.
    """
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
        key: PRNG key used for sampling.
        post: Posterior Multivariate Normal distribution.
        pi: Prior probability of element being active (not pruned).
        lam: Current sparsity matrix.
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix.
        prior: Prior Multivariate Normal distribution.

    Returns:
        A tuple of (new_delta_f, new_lam) — updated free energy changes and sparsity matrix.
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
        key: PRNG key used for sampling.
        post: Posterior MVNIG distribution.
        prior: Prior MVNIG distribution.
        pi: Prior probability of element being active (not pruned).
        lam: Current sparsity matrix.
        delta_f: Change in the variational free energy for the rows of the current sparsity matrix.

    Returns:
        A tuple of (new_delta_f, new_lam) — updated free energy changes and sparsity matrix.
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
