from typing import Optional

from jax import numpy as jnp, random as jr
from sppcax.distributions import MultivariateNormalInverseGamma
from sppcax.distributions import MultivariateNormal
from sppcax.types import PRNGKey


def _make_mvnig_prior(
    n_features: int,
    n_components: int,
    input_dim: int,
    has_bias: bool = True,
    isotropic_noise: bool = False,
    key: Optional[PRNGKey] = None,
) -> MultivariateNormalInverseGamma:
    """Create a default MVNIG emission prior for FA/PCA.

    Args:
        n_features: Number of observed features (emission_dim).
        n_components: Number of latent components (state_dim).
        input_dim: Dimensionality of control inputs.
        has_bias: Whether the model has an emission bias term.
        isotropic_noise: If True, use shared noise precision (PCA).
        key: Optional random key for initialization.

    Returns:
        Default MVNIG emission prior.
    """
    dim = n_components + has_bias + input_dim  # columns: [H, U, d] or just [H, U]
    if key is not None:
        loc = jr.normal(key, (n_features, dim)) * 0.01
    else:
        loc = jnp.zeros((n_features, dim))

    return MultivariateNormalInverseGamma(loc=loc, alpha0=2.0, beta0=1.0, isotropic_noise=isotropic_noise)


def _make_mvn_prior(
    state_dim: int,
    input_dim: int,
    has_bias: bool = True,
    key: Optional[PRNGKey] = None,
) -> MultivariateNormal:
    """Create a default MVN dynamics prior for DFA.

    Args:
        state_dim: Number of latent components (state_dim).
        input_dim: Dimensionality of control inputs.
        has_bias: Whether the model has a dynamics bias term.
        key: Optional random key for initialization.

    Returns:
        Default MVN dynamics prior.
    """
    dim = state_dim + input_dim + has_bias  # columns: [F, U, b] or just [F, U]
    loc = jnp.zeros((state_dim, dim))
    if key is not None:
        loc += jr.normal(key, (state_dim, dim)) * 0.01

    return MultivariateNormal(loc=loc)
