"""Utility functions for handling mixed likelihoods."""

from typing import Dict, List, Optional

import jax.numpy as jnp

from ..distributions import Categorical, ExponentialFamily, Normal, Poisson
from ..types import Array


def create_likelihood(likelihood_type: str, n_features: int, n_categories: Optional[int] = None) -> ExponentialFamily:
    """Create likelihood object for given type.

    Args:
        likelihood_type: Type of likelihood ('normal', 'categorical', or 'poisson').
        n_features: Number of features for this likelihood.
        n_categories: Number of categories for categorical likelihood.

    Returns:
        Likelihood object.
    """
    if likelihood_type == "normal":
        return Normal(n_features)
    elif likelihood_type == "categorical":
        if n_categories is None:
            raise ValueError("n_categories must be specified for categorical likelihood")
        return Categorical(jnp.zeros((n_features, n_categories - 1)))
    elif likelihood_type == "poisson":
        return Poisson(jnp.zeros(n_features))
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood_type}")


def parse_feature_groups(feature_types: Dict[str, List[int]]) -> Dict[int, ExponentialFamily]:
    """Create likelihood objects for feature groups.

    Args:
        feature_types: Dictionary mapping likelihood type to list of feature indices.
            Example: {'bernoulli': [0, 1], 'categorical': [2, 3]}

    Returns:
        Dictionary mapping feature index to likelihood object.
    """
    likelihoods = {}

    for likelihood_type, feature_indices in feature_types.items():
        n_features = len(feature_indices)

        # Handle categorical features
        n_categories = None
        if likelihood_type == "categorical":
            # Infer number of categories from data
            raise NotImplementedError("Categorical likelihood not yet supported")

        likelihood = create_likelihood(likelihood_type, n_features, n_categories)

        # Assign likelihood to each feature
        for idx in feature_indices:
            likelihoods[idx] = likelihood

    return likelihoods


def compute_expected_natural_parameters(
    W_mean: Array, W_cov: Array, z_mean: Array, z_cov: Array, feature_idx: int
) -> Array:
    """Compute expected natural parameters for a feature.

    For linear Gaussian model:
    E[Wz] = E[W]E[z]
    Var[Wz] = E[W]Var[z]E[W]^T + Tr(Var[z]Var[W])

    Args:
        W_mean: Mean of loading matrix (n_features, n_components).
        W_cov: Covariance of loading matrix.
        z_mean: Mean of latent variables (n_samples, n_components).
        z_cov: Covariance of latent variables.
        feature_idx: Index of feature to compute parameters for.

    Returns:
        Expected natural parameters for the feature.
    """
    # Extract parameters for this feature
    w_mean = W_mean[feature_idx]
    w_cov = W_cov[feature_idx]

    # Compute mean
    mean = jnp.dot(w_mean, z_mean.T)

    # Compute variance
    var = jnp.dot(jnp.dot(w_mean, z_cov), w_mean.T) + jnp.trace(jnp.dot(z_cov, w_cov))

    return mean, var


def compute_expected_log_likelihood(
    X: Array, likelihoods: Dict[int, ExponentialFamily], W_mean: Array, W_cov: Array, z_mean: Array, z_cov: Array
) -> Array:
    """Compute expected log likelihood under variational distribution.

    Args:
        X: Data matrix (n_samples, n_features).
        likelihoods: Dictionary mapping feature index to likelihood object.
        W_mean: Mean of loading matrix.
        W_cov: Covariance of loading matrix.
        z_mean: Mean of latent variables.
        z_cov: Covariance of latent variables.

    Returns:
        Expected log likelihood.
    """
    n_samples, n_features = X.shape
    total_ll = 0.0

    for j in range(n_features):
        # Get likelihood for this feature
        likelihood = likelihoods[j]

        # Compute expected natural parameters
        nat_params = compute_expected_natural_parameters(W_mean, W_cov, z_mean, z_cov, j)

        # Update likelihood with natural parameters
        likelihood.natural_parameters = nat_params

        # Compute expected log likelihood
        total_ll += likelihood.log_prob(X[:, j])

    return total_ll
