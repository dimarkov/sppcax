"""Functions for Bayesian model reduction."""

from typing import Any

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from multipledispatch import dispatch

from ..distributions import MultivariateNormalGamma
from ..models.factor_analysis_params import PFA
from ..types import Array
from .delta_f import gibbs_sampler_pfa


@dispatch(PFA)
def reduce_model(model: PFA, max_iter: int = 4) -> PFA:
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        model: Probabilistic factor analysis model
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

        dnat1 = lam.sum() * jnp.ones(())
        dnat2 = (1 - lam).sum() * jnp.ones(())
        sparsity_post = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), sparsity_post, (dnat1, dnat2))

        return (delta_f, sparsity_post, lam, key), delta_f

    init = (jnp.zeros(model.n_features), model.sparsity_prior, model.W_dist.mvn.mask, model.random_state)
    (last_df, sparsity_post, lam, key), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the model
    mvn_W = eqx.tree_at(lambda x: x.mask, model.W_dist.mvn, lam)  # update the mask of the loading matrix
    dnat1 = lam.sum(0) / 2
    tau = eqx.tree_at(lambda x: x.dnat1, model.W_dist.gamma, dnat1)
    W_dist = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.W_dist, (mvn_W, tau))

    dnat2 = model.noise_precision.dnat2  # TODO: add BMR correction to beta values
    noise_precision = eqx.tree_at(lambda x: x.dnat2, model.noise_precision, dnat2)
    model = eqx.tree_at(
        lambda x: (x.W_dist, x.sparsity_prior, x.noise_precision, x.random_state),
        model,
        (W_dist, sparsity_post, noise_precision, key),
    )

    return model


@dispatch(object, object)
def _apply_pruning(prior: Any, indices_to_prune: Array) -> Any:
    """Apply pruning to the prior by setting specified parameters to zero with high precision.

    Args:
        prior: Prior distribution
        indices_to_prune: Array of indices to prune

    Returns:
        Updated prior with pruned parameters
    """
    raise NotImplementedError(f"Pruning not implemented for {type(prior)}")


@dispatch(MultivariateNormalGamma, object)
def _apply_pruning(prior: MultivariateNormalGamma, indices_to_prune: Array) -> MultivariateNormalGamma:
    """Apply pruning to MultivariateNormalGamma prior.

    Sets mean to zero and precision to very high value for pruned parameters.

    Args:
        prior: MultivariateNormalGamma prior
        indices_to_prune: Array of indices to prune

    Returns:
        Updated MultivariateNormalGamma prior with pruned parameters
    """
    # Create copies of parameters for updating
    mean = prior.mvn.mean.copy()
    precision = prior.mvn.precision.copy()

    # Set pruned parameters to have zero mean and high precision
    for idx in indices_to_prune:
        mean = mean.at[tuple(idx)].set(0.0)
        precision = precision.at[tuple(idx), tuple(idx)].set(1e10)  # Very high precision

    # Create updated prior
    updated_mvn = eqx.tree_at(lambda x: (x.mean, x.precision), prior.mvn, (mean, precision))
    updated_prior = eqx.tree_at(lambda x: x.mvn, prior, updated_mvn)

    return updated_prior
