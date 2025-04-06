"""Multivariate Normal-Gamma distribution implementation."""

from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import equinox as eqx

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
        *,
        isotropic_noise,
        mask: Optional[Array] = None,
        alpha0: float = 2.0,
        beta0: float = 1.0,
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
        if isotropic_noise:
            self.gamma = Gamma(alpha0=alpha0, beta0=beta0)
        else:
            self.gamma = Gamma(
                alpha0=alpha0 * jnp.ones(self.mvn.batch_shape), beta0=beta0 * jnp.ones(self.mvn.batch_shape)
            )

        # Set shapes from MVN-Gamma
        super().__init__(batch_shape=self.mvn.batch_shape, event_shape=self.mvn.event_shape)

    def log_prob(self, x: Tuple[Array, Array]) -> Array:
        """Compute log probability.

        Args:
            x: Tuple of (w, psi) where:
                w: Value to compute probability for
                psi: Precision scalar

        Returns:
            Log probability
        """
        w, psi = x

        # MVN term: p(w|psi)
        sqrt_psi = jnp.sqrt(psi)
        mvn = eqx.tree_at(lambda x: x.nat1, self.mvn, self.mvn.nat1 * sqrt_psi[..., None])
        mvn_log_prob = mvn.log_prob(w * sqrt_psi[..., None])
        mvn_log_prob = mvn_log_prob + 0.5 * jnp.sum(self.mvn.mask, -1) * jnp.log(psi)

        # Gamma term: p(psi)
        gamma_log_prob = self.gamma.log_prob(psi)

        return mvn_log_prob + gamma_log_prob

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Tuple[Array, Array]:
        """Sample from the distribution.

        Args:
            key: PRNG key
            sample_shape: Shape of samples to draw

        Returns:
            Tuple of (value, tau) samples
        """
        key_psi, key_mvn = jr.split(key)

        # Sample psi ~ Gamma(α, β)
        psi = self.gamma.sample(key_psi, sample_shape=sample_shape)
        sqrt_psi = jnp.sqrt(psi)

        # Sample x|psi ~ MVN(μ, (psiΛ)⁻¹)
        # We can sample from base MVN and scale by sqrt(psi)
        mvn = eqx.tree_at(lambda x: x.nat1, self.mvn, self.mvn.nat1 * sqrt_psi[..., None])
        value = mvn.sample(key_mvn, sample_shape=sample_shape)
        value = value / sqrt_psi[..., None]

        return value, psi

    @property
    def mean(self) -> Array:
        """Get mean of the marginal distribution p(x)."""
        return self.mvn.mean

    @property
    def expected_covariance(self) -> Array:
        # mean of the inverse of precision (variance)
        exp_variance = jnp.broadcast_to(self.gamma.beta / (self.gamma.alpha - 1), self.batch_shape)
        return self.mvn.covariance * exp_variance[..., None, None]

    @property
    def expected_psi(self) -> Array:
        """Compute expected precision E[psi]."""
        return jnp.broadcast_to(self.gamma.mean, self.batch_shape)

    @property
    def expected_log_psi(self) -> Array:
        """Compute expected log precision E[log(psi)]."""
        return jnp.broadcast_to(jsp.digamma(self.gamma.alpha) - jnp.log(self.gamma.beta), self.batch_shape)

    @property
    def expected_sufficient_statistics_psi(self) -> Array:
        """Compute expected sufficient statistics of psi."""
        suff_stats = self.gamma.expected_sufficient_statistics
        return jnp.broadcast_to(suff_stats, self.batch_shape + (2,))
