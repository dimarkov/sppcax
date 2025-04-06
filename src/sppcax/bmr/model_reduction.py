"""Functions for Bayesian model reduction."""

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from multipledispatch import dispatch

from ..distributions import Beta, MultivariateNormal, Gamma
from ..models.factor_analysis_params import BayesianFactorAnalysisParams, PFA
from ..types import PRNGKey
from .delta_f import gibbs_sampler_mvn, gibbs_sampler_pfa


@dispatch(PFA)
def reduce_model(model: PFA, *, key: PRNGKey, max_iter: int = 4) -> PFA:
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        model: Probabilistic factor analysis model
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Pruned PFA model
    """

    def step_fn(carry, t):
        delta_f, sparsity_post, lam, key = carry

        # Compute delta F and sparsity matrix lambda for each parameter
        key, _key = jr.split(key)
        pi = sparsity_post.sample(_key)
        key, _key = jr.split(key)
        delta_f, lam = gibbs_sampler_pfa(_key, model, pi, lam, delta_f)

        dnat1 = lam.astype(delta_f.dtype).sum(0)
        dnat2 = (1 - lam).astype(delta_f.dtype).sum(0)
        sparsity_post = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), sparsity_post, (dnat1, dnat2))

        return (delta_f, sparsity_post, lam, key), delta_f

    init = (jnp.zeros(model.n_features), model.sparsity_prior, model.W_dist.mvn.mask, key)
    (last_df, sparsity_post, lam, key), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the model
    mvn_W = eqx.tree_at(lambda x: x.mask, model.W_dist.mvn, lam)  # update the mask of the loading matrix
    dnat1 = lam.sum(0) / 2
    tau = eqx.tree_at(lambda x: x.dnat1, model.W_dist.gamma, dnat1)
    W_dist = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.W_dist, (mvn_W, tau))

    dnat2 = model.noise_precision.dnat2
    pruned = model.W_dist.mvn.mask.astype(jnp.int8) - lam
    tilde_mu = pruned * model.W_dist.mvn.mean
    dnat2 -= 0.5 * (tilde_mu[..., None, :] @ (model.W_dist.mvn.covariance @ tilde_mu[..., None])).squeeze((-1, -2))
    noise_precision = eqx.tree_at(lambda x: x.dnat2, model.noise_precision, dnat2)
    model = eqx.tree_at(
        lambda x: (x.W_dist, x.sparsity_prior, x.noise_precision),
        model,
        (W_dist, sparsity_post, noise_precision),
    )

    return model


@dispatch(MultivariateNormal)
def reduce_model(  # noqa: F811
    model: MultivariateNormal, *, key: PRNGKey, pi: float = 0.5, max_iter: int = 4
) -> MultivariateNormal:
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
    (_, _, lam, key), _ = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the MVN distribution
    model = eqx.tree_at(lambda x: x.mask, model, lam)  # update the mask of the loading matrix

    return model


def remove_redundant_latents(model: BayesianFactorAnalysisParams) -> BayesianFactorAnalysisParams:
    """Remove redundant latent variables from the model."""
    remaining_components = jnp.nonzero(model.W_dist.mvn.mask.sum(0) > 0)

    alpha = model.W_dist.gamma.alpha[remaining_components]
    beta = model.W_dist.gamma.beta[remaining_components]
    gamma = Gamma(alpha, beta)

    W_dist = eqx.tree_at(lambda x: x.gamma, model.W_dist, gamma)
    model = eqx.tree_at(lambda x: x.W_dist, model, W_dist)
    return model
