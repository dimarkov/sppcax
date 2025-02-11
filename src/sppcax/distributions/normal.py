"""Normal distribution implementations."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax import lax

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Normal(ExponentialFamily):
    """Univariate normal distribution in natural parameters.

    The normal distribution has density:
    p(x|μ,σ) = 1/√(2πσ²) * exp(-(x-μ)²/(2σ²))

    In exponential family form:
    η = [μ/σ², -1/(2σ²)]
    T(x) = [x, x²]
    A(η) = -η₁²/(4η₂) - (1/2)log(-2η₂) + (1/2)log(2π)
    """

    nat1: Array  # First natural parameter (precision * mean)
    nat2: Array  # Second natural parameter (-0.5 * precision)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, loc: Array = 0.0, scale: Array = 1.0):
        """Initialize normal distribution.

        Args:
            loc: Location parameter μ (default: 0.0)
            scale: Scale parameter σ (default: 1.0)
        """
        # Convert to arrays
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        precision = 1.0 / (scale * scale)
        self.nat1 = jnp.broadcast_to(precision * loc, self.batch_shape)
        self.nat2 = jnp.broadcast_to(-0.5 * precision, self.batch_shape)

    @property
    def precision(self) -> Array:
        """Get precision parameter"""
        return -2.0 * self.nat2

    @property
    def loc(self) -> Array:
        """Get location parameter."""
        return self.nat1 / self.precision

    @property
    def scale(self) -> Array:
        """Get scale parameter."""
        return lax.rsqrt(self.precision)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [x, x²].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [x, x²] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([x, x**2], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [μ, μ² + σ²].

        Returns:
            Expected sufficient statistics [E[x], E[x²]] with shape:
            batch_shape + (2,)
        """
        loc = self.loc
        scale = self.scale
        return jnp.stack([loc, loc**2 + scale**2], axis=-1)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [precision*mean, -0.5*precision].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            Log base measure log(h(x)) with shape: batch_shape
        """
        return self.broadcast_to_shape(-0.5 * jnp.log(2.0 * jnp.pi), ignore_event=True)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η).

        Returns:
            Log normalizer with shape: batch_shape
        """
        return 0.25 * self.nat1**2 / self.nat2 - 0.5 * jnp.log(-2.0 * self.nat2)

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jr.normal(key, shape)
        return self.loc + self.scale * eps

    @property
    def entropy(self) -> Array:
        """Compute entropy of the distribution.

        Returns:
            Entropy with shape: batch_shape
        """
        return -self.log_base_measure() + 0.5 + jnp.log(self.scale)

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Normal":
        """Create normal distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            Normal distribution.
        """
        precision = -2.0 * eta[..., 1]
        loc = eta[..., 0] / precision
        scale = lax.rsqrt(precision)

        return cls(loc=loc, scale=scale)
