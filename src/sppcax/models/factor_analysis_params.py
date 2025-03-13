"""Bayesian Factor Analysis parameter containers."""

from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions import Beta, Delta, Distribution, Gamma
from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array, PRNGKey


def _to_distribution(X: Union[Array, Distribution]) -> Distribution:
    """Convert input to a Distribution if it isn't already.

    Args:
        X: Input data, either an Array or Distribution

    Returns:
        Distribution instance
    """
    if isinstance(X, Distribution):
        return X
    return Delta(X)


class BayesianFactorAnalysisParams(eqx.Module):
    """Base class for Bayesian Factor Analysis models (parameter container)."""

    n_components: int
    n_features: int
    isotropic_noise: bool
    W_dist: MultivariateNormalGamma  # batched over features
    noise_precision: Gamma  # Single precision for PPCA or per-feature for FA
    sparsity_prior: Beta  # Prior over sparsity probaility
    mean_: Array  # Data mean for centering
    data_mask: Optional[Array] = None  # Mask for missing data (True for observed, False for missing)
    random_state: Optional[PRNGKey] = None

    def __init__(
        self,
        n_components: int,
        n_features: int,
        isotropic_noise: bool = False,
        data_mask: Optional[Array] = None,
        random_state: Optional[PRNGKey] = None,
    ):
        """Initialize BayesianFactorAnalysis model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            isotropic_noise: If True, use same noise precision for all features (PPCA)
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
            random_state: Random state for initialization
        """
        self.n_components = n_components
        self.n_features = n_features
        self.isotropic_noise = isotropic_noise
        self.random_state = random_state
        self.data_mask = data_mask

        # Initialize mean
        self.mean_ = jnp.zeros(n_features)

        # Initialize parameters
        self._init_params()

    def _init_params(self) -> None:
        """Initialize model parameters."""
        key = self.random_state
        if key is None:
            key = jr.PRNGKey(0)

        # Initialize loading matrix columns
        loc = jr.normal(key, (self.n_features, self.n_components)) * 0.01
        mask = jnp.clip(jnp.arange(self.n_features), max=self.n_components)[..., None] >= jnp.arange(self.n_components)

        # set initial alpha to the value of the posterior
        alpha = 2 + (self.n_features - jnp.arange(self.n_components)) / 2
        self.W_dist = MultivariateNormalGamma(loc=loc, mask=mask, alpha=alpha, beta=1.0)

        self.sparsity_prior = Beta(alpha0=1.0, beta0=1.0)

        # Initialize noise precision
        if self.isotropic_noise:
            # Single precision for all features (PPCA)
            self.noise_precision = Gamma(alpha0=2.0, beta0=1.0)
        else:
            # Per-feature precision (FA)
            self.noise_precision = Gamma(alpha0=2 * jnp.ones(self.n_features), beta0=jnp.ones(self.n_features))

    def _validate_mask(self, X: Array) -> Array:
        """Validate and process the data mask.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Validated mask of shape (n_samples, n_features)
        """
        if self.data_mask is None:
            return jnp.ones_like(X, dtype=bool)

        if self.data_mask.shape != X.shape:
            raise ValueError(f"data_mask shape {self.data_mask.shape} does not match data shape {X.shape}")

        return self.data_mask


class PPCA(BayesianFactorAnalysisParams):
    """Probabilistic Principal Component Analysis (parameter container)."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        random_state: Optional[PRNGKey] = None,
        data_mask: Optional[Array] = None,
    ):
        """Initialize PPCA model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=True,
            random_state=random_state,
            data_mask=data_mask,
        )


class PFA(BayesianFactorAnalysisParams):
    """Probabilistic Factor Analysis with per-feature noise (parameter container)."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        random_state: Optional[PRNGKey] = None,
        data_mask: Optional[Array] = None,
    ):
        """Initialize Factor Analysis model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=False,
            random_state=random_state,
            data_mask=data_mask,
        )
