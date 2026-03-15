"""Delta distribution implementation."""

from typing import Callable, Optional

import jax.numpy as jnp

from ..types import Array, PRNGKey, Shape
from .base import Distribution


def default_ss(x: Array) -> Array:
    """Compute default sufficient statistics [x, vec(xx^T)] for MVN.

    Args:
        x: Input vector with shape (..., d).

    Returns:
        Concatenation of x and vectorized outer product with shape (..., d + d*d).
    """
    return jnp.concatenate([x, (x[..., None] * x[..., None, :]).reshape(*x.shape[:-1], -1)], axis=-1)


class Delta(Distribution):
    """Delta distribution (Dirac delta) concentrated at a single point."""

    mean: Array
    sufficient_statistics: Callable

    def __init__(
        self,
        location: Array,
        sufficient_statistics_fn: Optional[Callable] = None,
    ):
        """Initialize delta distribution.

        Args:
            location: Point where probability mass is concentrated.
                     Shape: batch_shape + event_shape
            sufficient_statistics_fn: Optional function to compute sufficient statistics.
                                    If None, uses MVN sufficient statistics.
        """
        *batch_shape, event_dim = location.shape
        super().__init__(batch_shape=tuple(batch_shape), event_shape=(event_dim,))

        self.mean = location
        self.sufficient_statistics = sufficient_statistics_fn if sufficient_statistics_fn is not None else default_ss

    def log_prob(self, x: Array) -> Array:
        """Compute log probability.

        Args:
            x: Value to compute log probability for.
               Shape: batch_shape + event_shape

        Returns:
            Log probability with shape: batch_shape
            Returns 0 at location, -inf elsewhere.
        """
        # Check if x equals location across event dimensions
        equal = jnp.all(x == self.mean, axis=tuple(range(-len(self.event_shape), 0)))
        return jnp.where(equal, 0.0, -jnp.inf)

    @property
    def covariance(self) -> Array:
        """Covariance matrix (always zero for delta distribution)."""
        zeros = jnp.zeros(self.shape)
        return zeros[..., None] * jnp.eye(zeros.shape[-1])

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution (always returns location).

        Args:
            key: PRNG key (unused).
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
            All samples are equal to location.
        """
        return jnp.broadcast_to(self.mean, sample_shape + self.shape)

    def entropy(self) -> Array:
        """Compute entropy (always 0 for delta distribution).

        Returns:
            Entropy with shape: batch_shape
        """
        return jnp.zeros(self.batch_shape)

    @property
    def precision(self) -> Array:
        """Precision (infinite for delta, but return large finite value for compatibility)."""
        return jnp.inf * jnp.ones_like(self.covariance)

    @property
    def expected_precision(self) -> Array:
        """Expected precision E[1/X] for Delta is 1/location (element-wise)."""
        return 1.0 / self.mean

    @property
    def expected_log_precision(self) -> Array:
        """Expected log-precision E[-log(X)] for Delta is -log(location)."""
        return -jnp.log(self.mean)

    @property
    def expected_second_moment(self) -> Array:
        """Expected second moment E[XX^T] = location @ location^T (no variance)."""
        return self.mean[..., :, None] * self.mean[..., None, :]

    def mf_expectations(self) -> dict:
        """Return expectations for mean-field coordinate ascent partner."""
        return {
            "mean": self.mean,
            "nat1": jnp.zeros_like(self.mean),
            "second_moment": self.expected_second_moment,
            "expected_precision": self.expected_precision,
            "expected_log_precision": self.expected_log_precision,
        }

    def mf_update(self, prior, stats, partner_expectations) -> "Delta":
        """Mean-field update for Delta is a no-op (fixed component)."""
        return self

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute expected sufficient statistics.

        For delta distribution, this is just sufficient_statistics(location)
        since all probability mass is concentrated at location.

        Returns:
            Expected sufficient statistics with shape determined by sufficient_statistics_fn.
        """
        return self.sufficient_statistics(self.mean)
