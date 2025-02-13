"""Multivariate Normal-Gamma distribution implementation."""

from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily
from .gamma import Gamma
from .mvn import MultivariateNormal


class MultivariateNormalGamma(ExponentialFamily):
    """Multivariate Normal-Gamma distribution.

    This distribution combines:
    p(x|τ) = N(x; μ, (τΛ)⁻¹)  # Multivariate Normal
    p(τ) = Gamma(τ; α, β)      # Precision scalar

    where:
    - x is a vector (can be batched)
    - μ is the location parameter
    - Λ is a base precision matrix
    - τ is a scalar precision parameter
    - α, β are Gamma distribution parameters
    """

    mvn: MultivariateNormal
    gamma: Gamma

    def __init__(
        self,
        loc: Array,
        mask: Optional[Array] = None,
        alpha: float = 2.0,
        beta: float = 1.0,
        scale_tril: Optional[Array] = None,
        covariance: Optional[Array] = None,
        precision: Optional[Array] = None,
    ):
        """Initialize MultivariateNormalGamma distribution.

        Args:
            loc: Location parameter
            mask: Optional boolean mask for active dimensions
            alpha: Shape parameter for Gamma prior
            beta: Rate parameter for Gamma prior
            scale_tril: Optional scale matrix (lower triangular)
            covariance: Optional covariance matrix
            precision: Optional precision matrix

        Note:
            Only one of scale_tril, covariance, or precision should be provided.
            If none are provided, identity matrix is used.
        """
        # Initialize MVN distribution
        self.mvn = MultivariateNormal(
            loc=loc, mask=mask, scale_tril=scale_tril, covariance=covariance, precision=precision
        )

        # Initialize Gamma parameters
        self.gamma = Gamma(alpha0=alpha * jnp.ones(self.mvn.event_shape), beta0=beta * jnp.ones(self.mvn.event_shape))

        # Set shapes from MVN
        super().__init__(batch_shape=self.mvn.batch_shape, event_shape=self.mvn.event_shape)

    def log_prob(self, x: Tuple[Array, Array]) -> Array:
        """Compute log probability.

        Args:
            x: Tuple of (value, tau) where:
                value: Value to compute probability for
                tau: Precision scalar

        Returns:
            Log probability
        """
        value, tau = x

        # MVN term: p(x|τ)
        # Scale precision matrix by tau
        mvn_log_prob = self.mvn.log_prob(value)
        mvn_log_prob = mvn_log_prob + 0.5 * jnp.sum(self.mvn.mask, -1) * jnp.log(tau)

        # Gamma term: p(τ)
        gamma_log_prob = self.gamma.log_prob(tau)

        return mvn_log_prob + gamma_log_prob

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Tuple[Array, Array]:
        """Sample from the distribution.

        Args:
            key: PRNG key
            sample_shape: Shape of samples to draw

        Returns:
            Tuple of (value, tau) samples
        """
        key_tau, key_mvn = jr.split(key)

        # Sample tau ~ Gamma(α, β)
        tau = self.gamma.sample(key_tau, sample_shape=sample_shape)

        # Sample x|tau ~ MVN(μ, (τΛ)⁻¹)
        # We can sample from base MVN and scale by sqrt(tau)
        value = self.mvn.sample(key_mvn, sample_shape=sample_shape)
        value = value / jnp.sqrt(tau)[..., None]

        return value, tau

    @property
    def mean(self) -> Array:
        """Get mean of the marginal distribution p(x)."""
        return self.mvn.mean

    @property
    def expected_covariance(self) -> Array:
        return self.mvn.covariance * (self.gamma.beta / (self.gamma.alpha - 1))[..., None]

    @property
    def expected_precision(self) -> Array:
        """Compute expected precision E[τ]."""
        return self.gamma.mean

    @property
    def expected_log_precision(self) -> Array:
        """Compute expected log precision E[log(τ)]."""
        return jsp.digamma(self.gamma.alpha) - jnp.log(self.gamma.beta)
