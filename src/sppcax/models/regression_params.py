"""Bayesian Factor Analysis parameter containers."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions import Beta, MultivariateNormal
from ..types import PRNGKey


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


class RegressionParams(eqx.Module):
    """Base class for Bayesian Factor Analysis models (parameter container)."""

    n_features: int
    n_controls: int
    q_b: MultivariateNormal  # Posterior over B (mvn) batched over features
    prior_prec: float  # prior precision for regression coefficients
    sparsity_prior: Beta  # Prior over sparsity probaility
    optimize_with_bmr: bool  # Whether to apply Bayesian Model Reduction optimization for tau and psi at every step

    def __init__(
        self,
        n_controls: int,
        n_features: int,
        prior_prec: float = 1.0,
        optimize_with_bmr: bool = False,
        use_bias: bool = True,
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
        self.n_controls = n_controls
        self.n_features = n_features
        self.prior_prec = prior_prec
        self.optimize_with_bmr = optimize_with_bmr

        # Initialize parameters
        if key is None:
            key = jr.PRNGKey(0)
        self._init_params(key, use_bias)

    def _init_params(self, key: PRNGKey, use_bias: bool) -> None:
        """Initialize model parameters."""

        # Initialize q(B|psi) - MVN part - q(psi) is tracked in PFA/PPCA models

        if self.n_controls > 0:
            loc = jr.normal(key, (self.n_features, self.n_controls)) / self.n_controls
            # add constant
            if use_bias:
                loc = jnp.pad(loc, [(0, 0), (0, 1)])
        else:
            if use_bias:
                loc = jnp.zeros((self.n_features, 1))
            else:
                raise Exception("Either number of predictors has to be large than zero or one has to use bias")

        nc = self.n_controls + int(use_bias)
        self.q_b = MultivariateNormal(loc=loc, precision=self.prior_prec * jnp.eye(nc))

        # Initialize sparsity prior
        self.sparsity_prior = Beta(alpha0=jnp.ones(nc), beta0=jnp.ones(nc))
