"""Base class for exponential family distributions."""

from typing import ClassVar

import jax.numpy as jnp

from ..types import Array, Shape
from .base import Distribution


class ExponentialFamily(Distribution):
    """Base class for exponential family distributions in natural parameterization.

    The exponential family has the form:
    p(x|η) = h(x)exp(η^T T(x) - A(η))
    where:
    - η: natural parameters
    - T(x): sufficient statistics
    - A(η): log normalizer
    - h(x): base measure

    Attributes:
        natural_param_shape: Shape of natural parameters (class variable).
    """

    natural_param_shape: ClassVar[Shape] = ()  # Override in subclasses

    def __init__(self, batch_shape: Shape = (), event_shape: Shape = ()):
        """Initialize exponential family distribution.

        Args:
            batch_shape: Shape of batch dimensions.
            event_shape: Shape of event dimensions.
        """
        super().__init__(batch_shape, event_shape)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters of the distribution.

        Returns:
            Natural parameters η with shape:
            batch_shape + natural_param_shape
        """
        raise NotImplementedError

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x).

        Args:
            x: Data to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics T(x) with shape:
            batch_shape + natural_param_shape
        """
        raise NotImplementedError

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute expected sufficient statistics E[T(x)].

        Returns:
            Expected sufficient statistics E[T(x)] with shape:
            batch_shape + natural_param_shape
        """
        raise NotImplementedError

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η).

        Returns:
            Log normalizer A(η) with shape: batch_shape
        """
        raise NotImplementedError

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            Log base measure log(h(x)) with shape: batch_shape
        """
        return jnp.zeros(())  # Default to h(x) = 1

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values with shape matching x's batch dimensions.
        """
        return jnp.ones(x.shape[: -len(self.event_shape)], dtype=bool)  # Default: all values valid

    def log_prob(self, x: Array) -> Array:
        """Compute log probability.

        Args:
            x: Data to compute log probability for.
               Shape: batch_shape + event_shape

        Returns:
            Log probability log p(x|η) with shape: batch_shape
            Returns -inf for values outside the support.
        """
        valid = self._check_support(x)
        eta = self.natural_parameters  # batch_shape + natural_param_shape
        T_x = self.sufficient_statistics(x)  # batch_shape + natural_param_shape

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(eta * T_x, axis=tuple(range(-len(self.natural_param_shape), 0)))
        log_prob = inner_product - self.log_normalizer + self.log_base_measure(x)
        print(inner_product.shape, self.log_normalizer.shape, self.log_base_measure(x).shape)

        return jnp.where(valid, log_prob, -jnp.inf)

    @property
    def expected_log_base_measure(self) -> Array:
        """Compute the expectation of the log base measure E_{p(x)}[log(h(x))]

        Returns:
            Expectation E_{p(x)}[log(h(x))] with shape: batch_shape
        """

        # We assume that by default base measure is not a function of x.
        # This has to be modified for distributions where this is not the case anymore.
        return self.log_base_measure()

    @property
    def entropy(self) -> Array:
        """Compute entropy of the distribution.

        Returns:
            Entropy with shape: batch_shape
        """

        eta = self.natural_parameters
        expected_T = self.expected_sufficient_statistics
        inner_product = jnp.sum(eta * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0)))

        return -self.expected_log_base_measure - inner_product + self.log_normalizer

    def kl_divergence(self, other: "ExponentialFamily") -> Array:
        """Compute KL divergence KL(self||other).

        Args:
            other: Other distribution to compute KL divergence with.

        Returns:
            KL divergence KL(self||other) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = other.natural_parameters
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other.log_normalizer + inner_product

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "ExponentialFamily":
        """Create distribution from natural parameters.

        Args:
            eta: Natural parameters with shape:
                batch_shape + natural_param_shape

        Returns:
            Distribution instance.
        """
        raise NotImplementedError
