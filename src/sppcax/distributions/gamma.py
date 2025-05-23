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

    nat1_0: Array  # prior value of the first naural parameter (α0 - 1)
    nat2_0: Array  # prior value of the second natural parameter (-β0)
    dnat1: Array  # Change in the first natural parameter (α-1)
    dnat2: Array  # Change in the second natural parameter (-β)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha0: Array = 1.0, beta0: Array = 1.0):
        """Initialize gamma distribution with alpha (shape) and beta (rate) parameters.

        Args:
            alpha: Shape parameter α (default: 1.0)
            beta: Rate parameter β (default: 1.0)
        """
        # Convert to arrays
        shape = jnp.asarray(alpha0)
        rate = jnp.asarray(beta0)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        self.nat1_0 = shape - 1  # α-1
        self.nat2_0 = jnp.broadcast_to(-rate, self.batch_shape)  # -β

        # Broadcast parameters
        self.dnat1 = jnp.zeros(())
        self.dnat2 = jnp.zeros(self.batch_shape)

    @property
    def nat1(self) -> Array:
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        return self.nat2_0 + self.dnat2

    @property
    def alpha(self) -> Array:
        """Get shape parameter α."""
        return self.nat1 + 1.0

    @property
    def beta(self) -> Array:
        """Get rate parameter β."""
        return -self.nat2

    @property
    def mean(self) -> Array:
        return self.alpha / self.beta

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

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values.
        """
        return x > 0

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

        return cls(alpha0=shape, beta0=rate)

    @property
    def kl_divergence_from_prior(self) -> Array:
        """Compute KL divergence KL(post||prior).

        Returns:
            KL divergence KL(post||prior) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = jnp.stack([self.nat1_0, self.nat2_0], axis=-1)
        alpha = eta_other[..., 0] + 1
        beta = -eta_other[..., 1]
        other_log_normalizer = jsp.gammaln(alpha) - alpha * jnp.log(beta)
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other_log_normalizer + inner_product


class InverseGamma(ExponentialFamily):
    """Inverse Gamma distribution in natural parameters.

    The gamma distribution has density:
    p(x|α,β) = β^α * x^(-α-1) * exp(-β/x) / Γ(α)

    In exponential family form:
    h(x) = 1
    η = [-α-1, -β]
    T(x) = [log(x), 1/x]
    A(η) = log(Γ(-η₁ - 1)) + (η₁ + 1)*log(-η₂)
    """

    nat1_0: Array  # prior value of the first naural parameter (-α0 - 1)
    nat2_0: Array  # prior value of the second natural parameter (-β0)
    dnat1: Array  # Change in the first natural parameter (-α-1)
    dnat2: Array  # Change in the second natural parameter (-β)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha0: Array = 1.0, beta0: Array = 1.0):
        """Initialize gamma distribution with alpha (shape) and beta (scale) parameters.

        Args:
            alpha: Shape parameter α (default: 1.0)
            beta: Scale parameter β (default: 1.0)
        """
        # Convert to arrays
        shape = jnp.asarray(alpha0)
        rate = jnp.asarray(beta0)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        self.nat1_0 = -shape - 1  # -α-1
        self.nat2_0 = jnp.broadcast_to(-rate, self.batch_shape)  # -β

        # Broadcast parameters
        self.dnat1 = jnp.zeros(())
        self.dnat2 = jnp.zeros(self.batch_shape)

    @property
    def nat1(self) -> Array:
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        return self.nat2_0 + self.dnat2

    @property
    def alpha(self) -> Array:
        """Get shape parameter α."""
        return -self.nat1 - 1.0

    @property
    def beta(self) -> Array:
        """Get rate parameter β."""
        return -self.nat2

    @property
    def mean(self) -> Array:
        return self.beta / (self.alpha - 1)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [-α-1, -β].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [log(x), 1/x].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [log(x), x] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([jnp.log(x), 1 / x], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [ψ(α) - log(β), α/β].

        Returns:
            Expected sufficient statistics [E[log(x)], E[1/x]] with shape:
            batch_shape + (2,)
        """
        alpha = self.alpha
        beta = self.beta
        return jnp.stack([jnp.log(beta) - jsp.digamma(alpha), alpha / beta], axis=-1)  # E[log(x)]  # E[x]

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(Γ(-η₁ - 1)) + (η₁ + 1)*log(-η₂)

        Returns:
            Log normalizer with shape: batch_shape
        """
        alpha = self.alpha
        beta = self.beta
        return jsp.gammaln(alpha) - alpha * jnp.log(beta)

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values.
        """
        return x > 0

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
        return self.beta / jr.gamma(key, self.alpha, shape=shape)

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Gamma":
        """Create gamma distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            Gamma distribution.
        """
        shape = -(eta[..., 0] + 1.0)  # α = - (η₁ + 1)
        rate = -eta[..., 1]  # β = -η₂

        return cls(alpha0=shape, beta0=rate)

    @property
    def kl_divergence_from_prior(self) -> Array:
        """Compute KL divergence KL(post||prior).

        Returns:
            KL divergence KL(post||prior) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = jnp.stack([self.nat1_0, self.nat2_0], axis=-1)
        alpha = -(eta_other[..., 0] + 1)
        beta = -eta_other[..., 1]
        other_log_normalizer = jsp.gammaln(alpha) - alpha * jnp.log(beta)
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other_log_normalizer + inner_product
