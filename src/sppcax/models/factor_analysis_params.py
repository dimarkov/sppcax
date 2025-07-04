"""Bayesian Factor Analysis parameter containers."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions import Beta, Gamma
from ..distributions.mvn_gamma import MultivariateNormalInverseGamma
from ..types import Array, PRNGKey
from .regression_params import RegressionParams


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
    n_controls: int
    isotropic_noise: bool
    update_ard: bool
    q_w_psi: MultivariateNormalInverseGamma  # Posterior over W (mvn) and psi (gamma), batched over features
    q_tau: Gamma  # Posterior over tau (ARD prior precision), batched over components
    sparsity_prior: Beta  # Prior over sparsity probaility
    optimize_with_bmr: bool  # Whether to apply Bayesian Model Reduction optimization for tau and psi at every step
    bmr_e_step: BMROptions
    bmr_m_step: BMROptions
    control: Optional[RegressionParams] = None
    data_mask: Optional[Array] = None  # Mask for missing data (True for observed, False for missing)

    def __init__(
        self,
        n_components: int,
        n_features: int,
        n_controls: int = 0,
        prior_prec_control: float = 1.0,
        isotropic_noise: bool = False,
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        update_ard: bool = True,
        use_bias: bool = True,
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
        self.n_controls = n_controls
        self.isotropic_noise = isotropic_noise
        self.optimize_with_bmr = optimize_with_bmr
        self.data_mask = data_mask
        self.bmr_e_step = BMROptions(bmr_e_step, bmr_e_step_opts)
        self.bmr_m_step = BMROptions(bmr_m_step, bmr_m_step_opts)
        self.update_ard = update_ard

        # Initialize parameters
        if key is None:
            key = jr.PRNGKey(0)
        key, _key = jr.split(key)
        self._init_params(_key)

        if n_controls - (1 - use_bias) >= 0:
            key, _key = jr.split(key)
            self.control = RegressionParams(
                n_controls, n_features, use_bias=use_bias, prior_prec=prior_prec_control, key=_key
            )

    def _init_params(self, key: PRNGKey) -> None:
        """Initialize model parameters."""
        key_w, key_psi, key_tau = jr.split(key, 3)

        # Initialize q(W|psi) - MVN part
        loc = jr.normal(key_w, (self.n_features, self.n_components)) * 0.01
        mask = jnp.clip(jnp.arange(self.n_features), max=self.n_components)[..., None] >= jnp.arange(self.n_components)

        # Initialize q(W, psi) - join posterior over the loading matrix, W, and noise precision, psi
        self.q_w_psi = MultivariateNormalInverseGamma(
            loc=loc, mask=mask, alpha0=2.0, beta0=1.0, isotropic_noise=self.isotropic_noise
        )

        # Initialize q(tau) - ARD prior
        alpha_tau = 0.5
        beta_tau = 0.5
        self.q_tau = Gamma(alpha0=alpha_tau * jnp.ones(self.n_components), beta0=beta_tau * jnp.ones(self.n_components))

        # Initialize sparsity prior
        self.sparsity_prior = Beta(alpha0=jnp.ones(self.n_components), beta0=jnp.ones(self.n_components))

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
        n_controls: int = 0,
        prior_prec_control: float = 1.0,
        use_bias: bool = True,
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        update_ard: bool = True,
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
            n_controls=n_controls,
            prior_prec_control=prior_prec_control,
            use_bias=use_bias,
            isotropic_noise=True,
            update_ard=update_ard,
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
        n_controls: int = 0,
        prior_prec_control: float = 1.0,
        use_bias: bool = True,
        optimize_with_bmr: bool = False,
        bmr_e_step: bool = False,
        bmr_m_step: bool = False,
        update_ard: bool = True,
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
            n_controls=n_controls,
            prior_prec_control=prior_prec_control,
            use_bias=use_bias,
            isotropic_noise=False,
            update_ard=update_ard,
            optimize_with_bmr=optimize_with_bmr,
            bmr_e_step=bmr_e_step,
            bmr_m_step=bmr_m_step,
            bmr_e_step_opts=bmr_e_step_opts,
            bmr_m_step_opts=bmr_m_step_opts,
            data_mask=data_mask,
            key=key,
        )
