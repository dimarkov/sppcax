"""Functions for Bayesian model reduction."""

from typing import Any, List, Optional, Tuple

import equinox as eqx
from multipledispatch import dispatch

from ..distributions.base import Distribution
from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array
from .delta_f import compute_delta_f


def reduce_model(
    posterior: Distribution, prior: Distribution, threshold: float = 3.0, indices: Optional[List[Tuple]] = None
) -> Tuple[Distribution, Array]:
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
    # Compute delta F for each parameter
    delta_f_values = compute_delta_f(posterior, prior, indices)

    # Determine which parameters to prune based on threshold
    pruning_mask = delta_f_values[:, 1] < threshold

    # Create updated prior with pruned parameters
    updated_prior = _apply_pruning(prior, delta_f_values[pruning_mask, 0].astype(int))

    return updated_prior, delta_f_values


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
