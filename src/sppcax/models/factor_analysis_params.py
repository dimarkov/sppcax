"""Bayesian Factor Analysis parameter containers."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions import Beta, Gamma
from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array, PRNGKey


class BMROptions(eqx.Module):
    use: bool = False
    vals: tuple = ()

    @property
    def opts(self):
        if len(self.vals) == 2:
            return {self.vals[0]: self.vals[1]}
        elif len(self.vals) == 4:
            return {self.vals[0]: self.vals[1], self.vals[2]: self.vals[3]}
        else:
            raise NotImplementedError


class BayesianFactorAnalysisParams(eqx.Module):
    """Base class for Bayesian Factor Analysis models (parameter container)."""

    n_components: int
    n_features: int
    isotropic_noise: bool
    W_dist: MultivariateNormalGamma  # batched over features
    noise_precision: Gamma  # Single precision for PPCA or per-feature for FA
    sparsity_prior: Beta  # Prior over sparsity probaility
    mean_: Array  # Data mean for centering
    optimize_with_bmr: bool  # Whether to apply Bayesian Model Reduction optimization for tau and psi at every step
    bmr_e_step: BMROptions
    bmr_m_step: BMROptions
    data_mask: Optional[Array] = None  # Mask for missing data (True for observed, False for missing)

    def __init__(
        self,
        n_components: int,
        n_features: int,
        isotropic_noise: bool = False,
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        bmr_e_step_opts: Optional[tuple] = ("max_iter", 4, "pi", 0.5),
        bmr_m_step_opts: Optional[tuple] = ("max_iter", 4),
        data_mask: Optional[Array] = None,
        *,
        key: Optional[PRNGKey] = None,
    ):
        """Initialize BayesianFactorAnalysis model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            isotropic_noise: If True, use same noise precision for all features (PPCA)
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.n_components = n_components
        self.n_features = n_features
        self.isotropic_noise = isotropic_noise
        self.optimize_with_bmr = optimize_with_bmr
        self.data_mask = data_mask
        self.bmr_e_step = BMROptions(bmr_e_step, bmr_e_step_opts)
        self.bmr_m_step = BMROptions(bmr_m_step, bmr_m_step_opts)

        # Initialize mean
        self.mean_ = jnp.zeros(n_features)

        # Initialize parameters
        if key is None:
            key = jr.PRNGKey(0)
        self._init_params(key)

    def _init_params(self, key: PRNGKey) -> None:
        """Initialize model parameters."""

        # Initialize loading matrix columns
        loc = jr.normal(key, (self.n_features, self.n_components)) * 0.01
        mask = jnp.clip(jnp.arange(self.n_features), max=self.n_components)[..., None] >= jnp.arange(self.n_components)

        # set initial alpha to the value of the posterior
        alpha = 1
        beta = 1
        self.W_dist = MultivariateNormalGamma(loc=loc, mask=mask, alpha=alpha, beta=beta)

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
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        bmr_e_step_opts: Optional[tuple] = ("max_iter", 4, "pi", 0.5),
        bmr_m_step_opts: Optional[tuple] = ("max_iter", 4),
        data_mask: Optional[Array] = None,
        *,
        key: Optional[PRNGKey] = None,
    ):
        """Initialize PPCA model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=True,
            optimize_with_bmr=optimize_with_bmr,
            bmr_e_step=bmr_e_step,
            bmr_m_step=bmr_m_step,
            bmr_e_step_opts=bmr_e_step_opts,
            bmr_m_step_opts=bmr_m_step_opts,
            data_mask=data_mask,
            key=key,
        )


class PFA(BayesianFactorAnalysisParams):
    """Probabilistic Factor Analysis with per-feature noise (parameter container)."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        bmr_e_step_opts: Optional[tuple] = ("max_iter", 4, "pi", 0.5),
        bmr_m_step_opts: Optional[tuple] = ("max_iter", 4),
        data_mask: Optional[Array] = None,
        *,
        key: Optional[PRNGKey] = None,
    ):
        """Initialize Factor Analysis model parameters.

        Args:
            n_components: Number of components
            n_features: Number of features
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=False,
            optimize_with_bmr=optimize_with_bmr,
            bmr_e_step=bmr_e_step,
            bmr_m_step=bmr_m_step,
            bmr_e_step_opts=bmr_e_step_opts,
            bmr_m_step_opts=bmr_m_step_opts,
            data_mask=data_mask,
            key=key,
        )
