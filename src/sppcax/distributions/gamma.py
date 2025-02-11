"""Gamma distribution implementation."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Gamma(ExponentialFamily):
    """Gamma distribution in natural parameters.

    The gamma distribution has density:
    p(x|α,β) = β^α * x^(α-1) * exp(-βx) / Γ(α)

    In exponential family form:
    h(x) = 1
    η = [α-1, -β]
    T(x) = [log(x), x]
    A(η) = log(Γ(η₁ + 1)) - (η₁ + 1)*log(-η₂)
    """

    nat1: Array  # First natural parameter (α-1)
    nat2: Array  # Second natural parameter (-β)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha: Array = 1.0, beta: Array = 1.0):
        """Initialize gamma distribution with alpha (shape) and beta (rate) parameters.

        Args:
            alpha: Shape parameter α (default: 1.0)
            beta: Rate parameter β (default: 1.0)
        """
        # Convert to arrays
        shape = jnp.asarray(alpha)
        rate = jnp.asarray(beta)

        # Convert to natural parameters
        self.nat1 = shape - 1.0  # α-1
        self.nat2 = -rate  # -β

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Broadcast parameters
        self.nat1 = jnp.broadcast_to(self.nat1, self.batch_shape)
        self.nat2 = jnp.broadcast_to(self.nat2, self.batch_shape)

    @property
    def alpha(self) -> Array:
        """Get shape parameter α."""
        return self.nat1 + 1.0

    @property
    def beta(self) -> Array:
        """Get rate parameter β."""
        return -self.nat2

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [α-1, -β].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [log(x), x].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [log(x), x] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([jnp.log(x), x], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [ψ(α) - log(β), α/β].

        Returns:
            Expected sufficient statistics [E[log(x)], E[x]] with shape:
            batch_shape + (2,)
        """
        alpha = self.alpha
        beta = self.beta
        return jnp.stack([jsp.digamma(alpha) - jnp.log(beta), alpha / beta], axis=-1)  # E[log(x)]  # E[x]

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(Γ(η₁ + 1)) - (η₁ + 1)*log(-η₂).

        Returns:
            Log normalizer with shape: batch_shape
        """
        alpha = self.alpha
        beta = self.beta
        return jsp.gammaln(alpha) - alpha * jnp.log(beta)

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            zero
        """
        return jnp.zeros(())

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        return jr.gamma(key, self.alpha, shape=shape) / self.beta

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Gamma":
        """Create gamma distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            Gamma distribution.
        """
        shape = eta[..., 0] + 1.0  # α = η₁ + 1
        rate = -eta[..., 1]  # β = -η₂

        return cls(alpha=shape, beta=rate)
