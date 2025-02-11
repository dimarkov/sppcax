"""Poisson distribution in natural parameterization."""

from typing import ClassVar, Optional

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Poisson(ExponentialFamily):
    """Poisson distribution parameterized by log rate.

    The Poisson distribution has natural parameter η = log(λ)
    where λ is the rate parameter, and sufficient statistic T(x) = x.
    """

    nat1: Array
    natural_param_shape: ClassVar[Shape] = ()  # [log_rate]

    def __init__(self, log_rate: Array):
        """Initialize Poisson distribution.

        Args:
            log_rate: Natural parameter η = log(λ).
        """
        super().__init__(batch_shape=jnp.shape(log_rate), event_shape=())
        self.nat1 = log_rate

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Poisson":
        """Create Poisson from natural parameters.

        Args:
            log_rate: Natural parameter η = log(λ).

        Returns:
            Poisson instance.
        """
        return cls(log_rate=eta)

    @property
    def log_rate(self) -> Array:
        """Get log(rate) parameter."""
        return self.nat1

    @property
    def rate(self) -> Array:
        """Get rate parameter."""
        return jnp.exp(self.log_rate)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters (log rate).

        Returns:
            Natural parameters η = log(λ).
        """
        return self.nat1

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = x.

        Args:
            x: Count data.

        Returns:
            Sufficient statistics T(x) = x.
        """
        return x

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = E[x] = λ = exp(η).

        Returns:
            Expected sufficient statistics E[x].
        """
        return jnp.exp(self.nat1)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = exp(η).

        Returns:
            Log normalizer A(η).
        """
        return jnp.exp(self.nat1)

    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure log(h(x)) = -log(x!).

        Args:
            x: Count data.

        Returns:
            Log base measure -log(x!).
        """
        return -gammaln(x + 1)

    def sample(self, key: PRNGKey, sample_shape: Optional[Shape] = ()) -> Array:
        """Sample from Poisson distribution.

        Args:
            key: PRNG key.
            sample_shape: Shape of samples to draw.

        Returns:
            Count samples from distribution.
        """
        rate = jnp.broadcast_to(jnp.exp(self.log_rate), sample_shape + self.shape)
        return jr.poisson(key, rate)

    @property
    def entropy(self) -> Array:
        """Compute entropy of Poisson distribution.

        Returns:
            Entropy H(λ) = λ(1 - log(λ)) + exp(-λ)sum_{k=0}^∞ λ^k log(k!)/k!
        """
        # Approximate entropy using rate and log rate
        inv_rate = jnp.exp(-self.log_rate)
        return 0.5 * (jnp.log(2 * jnp.pi) + 1 + self.log_rate) - inv_rate / 12
