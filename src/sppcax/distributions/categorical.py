"""Categorical distribution in natural parameterization."""

from typing import ClassVar, Optional

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily

MAXF = jnp.finfo(jnp.dtype(jnp.array(0.0).dtype)).max


class Categorical(ExponentialFamily):
    """Categorical distribution parameterized by logits.

    The Categorical distribution has K-1 natural parameters
    η_k = log(p_k/p_K) for k=1,...,K-1 where p_K is the probability
    of the last category.
    """

    nat1: Array  # Shape (..., K-1)
    natural_param_shape: ClassVar[Shape] = (1,)

    def __init__(self, logits: Array):
        """Initialize Categorical distribution.

        Args:
            logits: Log-probs
                Shape (..., K) where K is number of categories.
        """
        super().__init__(batch_shape=logits.shape[:-1], event_shape=())
        self.nat1 = logits[..., :-1] - logits[..., -1:]

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Categorical":
        """Create Categorical from natural parameters.

        Args:
            logits: Log-odds relative to last category.
                Shape (..., K-1) where K is number of categories.

        Returns:
            Categorical instance.
        """
        pad_width = [(0, 0)] * eta.ndim
        pad_width[-1] = (0, 1)
        return cls(logits=jnp.pad(eta, pad_width))

    @property
    def full_logits(self) -> Array:
        pad_width = [(0, 0)] * self.nat1.ndim
        pad_width[-1] = (0, 1)
        return jnp.pad(self.nat1, pad_width)

    @property
    def probs(self) -> Array:
        """Get probability parameters."""
        return jnn.softmax(self.full_logits)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters (logits).

        Returns:
            Natural parameters η.
        """
        return self.nat1

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support."""
        n_categories = self.nat1.shape[-1] + 1
        return (x >= 0) & (x < n_categories)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x).

        For categorical data, T(x) is a one-hot vector with the
        last category omitted (since probabilities sum to 1).

        Args:
            x: Category indices.

        Returns:
            One-hot encoded data (excluding last category).
        """
        n_categories = self.nat1.shape[-1] + 1
        one_hot = jnn.one_hot(x, n_categories)
        return one_hot[..., :-1]  # Omit last category

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = p_1,...,p_{K-1}.

        Returns:
            Expected probabilities for first K-1 categories.
        """
        probs = jnn.softmax(self.full_logits)
        return probs[..., :-1]  # Return first K-1 probabilities

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η).

        For categorical, A(η) = log(1 + sum(exp(η_k))).

        Returns:
            Log normalizer A(η).
        """
        return jnn.logsumexp(self.full_logits)

    def sample(self, key: PRNGKey, sample_shape: Optional[Shape] = None) -> Array:
        """Sample from Categorical distribution.

        Args:
            key: PRNG key.
            sample_shape: Shape of samples to draw.

        Returns:
            Category indices sampled from distribution.
        """
        return jr.categorical(key, self.full_logits, shape=sample_shape + self.shape)
