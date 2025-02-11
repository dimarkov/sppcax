"""Bernoulli distribution in natural parameterization."""

from typing import ClassVar

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Bernoulli(ExponentialFamily):
    """Bernoulli distribution parameterized by logits.

    The Bernoulli distribution has density:
    p(x|p) = p^x * (1-p)^(1-x)

    In exponential family form:
    η = logit(p)
    T(x) = x
    A(η) = log(1 + exp(η))
    """

    logits: Array  # Natural parameter η = logit(p)
    natural_param_shape: ClassVar[Shape] = (1,)  # [η]

    def __init__(self, logits: Array):
        """Initialize Bernoulli distribution.

        Args:
            logits: Log-odds of success probability, η = log(p/(1-p)).
        """
        # Convert to array
        logits = jnp.asarray(logits)

        # Ensure logits has natural_param_shape at the end
        if logits.ndim == 0:
            self.logits = logits[None]
            batch_shape = ()
        else:
            self.logits = logits[..., None] if logits.shape[-1:] != (1,) else logits
            batch_shape = jnp.shape(self.logits)[:-1]

        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def probs(self) -> Array:
        """Get success probability."""
        return jnn.sigmoid(self.logits[..., 0])

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters (logits).

        Returns:
            Natural parameters η with shape:
            batch_shape + (1,)
        """
        return self.logits

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = x.

        Args:
            x: Binary data (0 or 1).
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics T(x) with shape:
            batch_shape + (1,)
        """
        return x[..., None]

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = E[x] = p = sigmoid(η).

        Returns:
            Expected sufficient statistics E[x] with shape:
            batch_shape + (1,)
        """
        return jnn.sigmoid(self.logits)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(1 + exp(η)).

        Returns:
            Log normalizer with shape: batch_shape
        """
        return jnn.softplus(self.logits[..., 0])

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
        """Sample from Bernoulli distribution.

        Args:
            key: PRNG key.
            sample_shape: Shape of samples to draw.

        Returns:
            Binary samples with shape:
            sample_shape + batch_shape + event_shape
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        return jr.bernoulli(key, self.probs, shape=shape)

    @property
    def entropy(self) -> Array:
        """Compute entropy of Bernoulli distribution.

        Returns:
            Entropy with shape: batch_shape
        """
        p = self.probs
        return -(p * jnp.log(p + 1e-12) + (1 - p) * jnp.log(1 - p + 1e-12))

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Bernoulli":
        """Create Bernoulli distribution from natural parameters.

        Args:
            eta: Natural parameters η with shape:
                batch_shape + (1,)

        Returns:
            Bernoulli distribution.
        """
        return cls(logits=eta)
