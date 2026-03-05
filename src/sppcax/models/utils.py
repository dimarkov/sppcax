from jax import numpy as jnp, random as jr
from sppcax.distributions import MultivariateNormalInverseGamma
from sppcax.distributions import MultivariateNormal


def _make_mvnig_prior(n_features, n_components, input_dim, has_bias=True, isotropic_noise=False, key=None):
    """Create a default MVNIG emission prior for FA/PCA.

    Args:
        n_features: Number of observed features (emission_dim).
        n_components: Number of latent components (state_dim).
        has_bias: Whether the model has an emission bias term.
        isotropic_noise: If True, use shared noise precision (PCA).
        key: Optional random key for initialization.
    """
    dim = n_components + has_bias + input_dim  # columns: [H, U, d] or just [H, U]
    if key is not None:
        loc = jr.normal(key, (n_features, dim)) * 0.01
    else:
        loc = jnp.zeros((n_features, dim))

    # Default mask: lower-triangular-like for identifiability
    mask = jnp.clip(jnp.arange(n_features), max=n_components)[..., None] >= jnp.arange(n_components)
    mask = jnp.pad(mask, [(0, 0), (0, int(has_bias) + input_dim)], constant_values=1)

    return MultivariateNormalInverseGamma(
        loc=loc, mask=mask, alpha0=2.0, beta0=1.0, isotropic_noise=isotropic_noise
    )


def _make_mvn_prior(state_dim, input_dim, has_bias=True, key=None):
    """Create a default MVN dynamics prior for DFA.

    Args:
        state_dim: Number of latent components (state_dim).
        has_bias: Whether the model has a dynamics bias term.
        key: Optional random key for initialization.
    """

    dim = state_dim + input_dim + has_bias  # columns: [F, U, b] or just [F, U]
    loc = jnp.pad(jnp.eye(state_dim), [(0, 0), (0, int(has_bias) + input_dim)])
    if key is not None:
        loc += jr.normal(key, (state_dim, dim)) * 0.01
 
    return MultivariateNormal(loc=loc)