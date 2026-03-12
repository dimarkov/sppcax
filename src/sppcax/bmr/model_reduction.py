"""Functions for Bayesian model reduction."""

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import digamma
from multipledispatch import dispatch

from ..distributions import Beta, MultivariateNormal as MVN, MultivariateNormalInverseGamma as MVNIG
from ..types import PRNGKey
from .delta_f import gibbs_sampler_mvn, gibbs_sampler_mvnig, gibbs_sampler_with_ard

from dynamax.utils.distributions import NormalInverseWishart as NIW


@dispatch(NIW, NIW)
def prune_params(post: NIW, prior: NIW, *, key: PRNGKey, max_iter: int = 4) -> NIW:
    """Applies Bayesian model reduction to distribution, where Inverse Gamma priors
    get optimized, and individual elements of the Multivariate Normal get pruned to
    maximize change in the variational free energy, hence minimize the upper bound
    on marginal log-likelihood.

    Args:
        post: Posterior MultivariateNormalInverseGamma distribution
        prior: Prior MultivariateNormalInverseGamma distribution
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Optimized posterior MultivariateNormalInverseGamma distribution
    """

    # optimize Inverse Gamma priors and consequently posterior estimates

    # nu = post.df
    # nu_0 = prior.df
    # diff_scale = post.scale - prior.scale
    # new_scale = nu * diff_scale / (nu - nu_0)

    # opt_post = NIW(post.loc, post.mean_concentration, nu, new_scale)

    # TODO: prune parameters
    # Decide if we should to force a priori some latents to zero.

    return post


@dispatch(MVNIG, MVNIG)
def prune_params(post: MVNIG, prior: MVNIG, *, key: PRNGKey, max_iter: int = 8, ard_post=None) -> MVNIG:  # noqa: F811
    """Applies Bayesian model reduction to distribution, where Inverse Gamma priors
    get optimized, and individual elements of the Multivariate Normal get pruned to
    maximize change in the variational free energy, hence minimize the upper bound
    on marginal log-likelihood.

    Args:
        post: Posterior MultivariateNormalInverseGamma distribution
        prior: Prior MultivariateNormalInverseGamma distribution
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler
        ard_post: Optional ARD Gamma posterior over column precisions. When provided,
            uses ARD-aware Gibbs sampler that accounts for column precision in
            the free energy change computation.

    Returns:
        Optimized posterior MultivariateNormalInverseGamma distribution
    """

    # prune parameters
    def step_fn(carry, t):
        delta_f, mask, alpha_0, alpha, beta, key = carry

        key, _key = jr.split(key)
        pi = alpha / (alpha + beta)  # Beta(alpha, beta).sample(_key)

        key, _key = jr.split(key)
        if ard_post is not None:
            delta_f, mask = gibbs_sampler_with_ard(_key, post, ard_post, pi, mask, delta_f)
        else:
            delta_f, mask = gibbs_sampler_mvnig(_key, post, prior, pi, mask, delta_f)

        alpha = alpha_0 / len(alpha) + mask.sum(0)
        beta = 1.0 + (1 - mask).sum(0)
        alpha_0 = -jnp.square(len(alpha)) / jnp.sum(digamma(alpha) - digamma(alpha + beta))

        return (delta_f, mask, alpha_0, alpha, beta, key), delta_f

    alpha_0 = 1.0
    alpha = jnp.ones(post.event_shape)
    beta = jnp.ones(post.event_shape)
    init = (jnp.zeros(post.batch_shape), prior.mvn.mask, alpha_0, alpha, beta, key)
    (last_df, mask, *_), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter), unroll=2)

    # correct beta parameter of inverse-gamma distribution
    prior_nat1 = prior.mvn.nat1
    pruned_prior_mean = (~mask) * prior.mvn.mean

    post_nat1 = post.mvn.nat1
    pruned_post_mean = (~mask) * post.mvn.mean

    dnat2 = post.inv_gamma.dnat2
    dnat2 -= jnp.sum(pruned_post_mean * post_nat1, -1) / 2
    dnat2 += jnp.sum(pruned_prior_mean * prior_nat1, -1) / 2

    # optimize Inverse Gamma priors
    nat2_0 = jnp.minimum(dnat2 / (post.inv_gamma.alpha - 1), -1 / (post.inv_gamma.alpha - 1))
    dnat2 = jnp.minimum(dnat2, 0.0)
    opt_post = eqx.tree_at(lambda d: (d.mvn.mask, d.inv_gamma.nat2_0, d.inv_gamma.dnat2), post, (mask, nat2_0, dnat2))

    return opt_post


@dispatch(MVN, MVN)
def prune_params(post: MVN, prior: MVN, *, key: PRNGKey, max_iter: int = 8) -> MVN:  # noqa: F811
    """Applies Bayesian model reduction to distribution, where the individual elements of a
    Multivariate Normal get pruned to maximize change in the variational free energy,
    hence minimize the upper bound on marginal log-likelihood.

    Args:
        post: Posterior MultivariateNormal distribution
        prior: Prior MultivariateNormal distribution
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Optimized posterior MultivariateNormal distribution
    """

    # prune parameters
    def step_fn(carry, t):
        delta_f, mask, alpha_0, alpha, beta, key = carry

        pi = alpha / (alpha + beta)

        key, _key = jr.split(key)
        delta_f, mask = gibbs_sampler_mvn(_key, post, pi, mask, delta_f, prior=prior)

        alpha = alpha_0 / len(alpha) + mask.sum(0)
        beta = 1.0 + (1 - mask).sum(0)

        alpha_0 = -jnp.square(len(alpha)) / jnp.sum(digamma(alpha) - digamma(alpha + beta))

        return (delta_f, mask, alpha_0, alpha, beta, key), delta_f

    alpha_0 = 1.0
    alpha = jnp.ones(post.event_shape)
    beta = jnp.ones(post.event_shape)
    init = (jnp.zeros(post.batch_shape), prior.mask, alpha_0, alpha, beta, key)
    (last_df, mask, *_), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter), unroll=2)

    opt_post = eqx.tree_at(lambda d: (d.mask,), post, (mask,))

    return opt_post


@dispatch(MVN)
def reduce_model(model: MVN, *, key: PRNGKey, pi: float = 0.5, max_iter: int = 1) -> MVN:  # noqa: F811
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        model: MultivariateNormal distribution
        pi: Prior expected probability of parameter being pruned
        max_iter: Maximal number of iterations for the Gibbs sampler
        key: Random state for initialization

    Returns:
        Pruned MultivariateNormal distribution
    """

    def step_fn(carry, t):
        delta_f, sparsity_dist, lam, key = carry

        # Compute delta F and sparsity matrix lambda for each parameter
        key, _key = jr.split(key)
        pi = sparsity_dist.sample(_key)
        key, _key = jr.split(key)
        delta_f, lam = gibbs_sampler_mvn(_key, model, pi, lam, delta_f)

        dnat1 = lam.astype(delta_f.dtype).sum(0)
        dnat2 = (1 - lam).astype(delta_f.dtype).sum(0)
        sparsity_dist = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), sparsity_dist, (dnat1, dnat2))

        return (delta_f, sparsity_dist, lam, key), delta_f

    K = model.mask.shape[-1]
    sparity_prior = Beta(10 * pi * jnp.ones(K), 10 * (1 - pi) * jnp.ones(K))
    init = (jnp.zeros(model.batch_shape), sparity_prior, model.mask, key)
    (delta_f, sparsity_dist, lam, key), _ = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the MVN distribution
    model = eqx.tree_at(lambda x: (x.mask, x.nat1), model, (lam, jnp.where(lam, model.nat1, 0.0)))

    return model
