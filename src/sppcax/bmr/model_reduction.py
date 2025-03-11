"""Functions for Bayesian model reduction."""

from typing import Any

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from multipledispatch import dispatch

from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..models.factor_analysis_params import PFA
from ..types import Array
from .delta_f import compute_delta_f


@dispatch(PFA)
def reduce_model(model: PFA, max_iter: int = 4) -> PFA:
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        posterior: Posterior distribution
        prior: Prior distribution
        threshold: Log evidence threshold for pruning (larger = more aggressive pruning)
        indices: Optional list of parameter indices to consider for pruning

    Returns:
        Tuple containing:
        - Updated prior with pruned parameters
        - Array of log evidence values for pruned vs. unpruned models
    """

    def step_fn(carry, t):
        delta_f, sparsity_post, lam = carry

        # Compute delta F and sparsity matrix lambda for each parameter
        pi = sparsity_post.sample()
        delta_f, lam = compute_delta_f(model, pi, lam, delta_f)

        sparsity_post = eqx.tree_at(lambda x: x.count, sparsity_post, lam.sum())

        return (delta_f, sparsity_post, lam), delta_f

    init = (jnp.zeros(PFA.n_features), PFA.sparsity_prior, PFA.W_dist.mvn.mask)
    (last_df, sparsity_post, lam), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the model
    mvn_W = eqx.tree_at(lambda x: x.mask, model.W_dist.mvn, lam)  # update the mask of the loading matrix
    W_dist = eqx.tree_at(lambda x: x.mvn, model.W_dist, mvn_W)
    dnat2 = model.noise_precision.dnat2  # TODO: add correction to beta values
    noise_precision = eqx.tree_at(lambda x: x.dnat2, model.noise_precision, dnat2)
    model = eqx.tree_at(
        lambda x: (x.W_dist, x.prior_sparsity, x.noise_precision), model, (W_dist, sparsity_post, noise_precision)
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
