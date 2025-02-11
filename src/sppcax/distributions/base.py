"""Base distribution class."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from ..types import Array, PRNGKey, Shape


class Distribution(eqx.Module):
    """Base distribution class in natural parameters.

    Attributes:
        batch_shape: Shape of batch dimensions.
        event_shape: Shape of event dimensions.
    """

    batch_shape: Shape
    event_shape: Shape

    def __init__(self, batch_shape, event_shape):
        """Initialize distribution with empty shapes."""
        self.batch_shape = batch_shape
        self.event_shape = event_shape

    @property
    def shape(self):
        return self.batch_shape + self.event_shape

    def broadcast_to_shape(self, x: Array, ignore_event: bool = False) -> Array:
        """Broadcast array to match distribution shape.

        Args:
            x: Array to broadcast.
            ignore_event: If True, only broadcast batch dimensions.

        Returns:
            Broadcasted array.
        """
        target_shape = self.batch_shape
        if not ignore_event:
            target_shape = target_shape + self.event_shape
        return jnp.broadcast_to(x, target_shape)

    def log_prob(self, x: Array) -> Array:
        """Compute log probability of x.

        Args:
            x: Value to compute log probability for. Should have shape:
               batch_shape + event_shape

        Returns:
            Log probability with shape: batch_shape
        """
        raise NotImplementedError

    def sample(self, key: PRNGKey, sample_shape: Optional[Shape] = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Additional sample dimensions.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
        """
        raise NotImplementedError

    def entropy(self) -> Array:
        """Compute entropy of the distribution.

        Returns:
            Entropy with shape: batch_shape
        """
        raise NotImplementedError

    def __call__(self, x: Array) -> Array:
        """Compute log probability of x.

        Args:
            x: Value to compute log probability for.

        Returns:
            Log probability of x.
        """
        return self.log_prob(x)
