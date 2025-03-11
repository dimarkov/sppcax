"""Beta distribution implementation."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Beta(ExponentialFamily):
    """Beta distribution in natural parameters.

    The beta distribution has density:
    p(x|α,β) = x^(α-1) * (1-x)^(β-1) / B(α,β)  for x ∈ [0,1]

    In exponential family form:
    h(x) = 1
    η = [α-1, β-1]
    T(x) = [log(x), log(1-x)]
    A(η) = log(B(η₁+1, η₂+1))
    """

    nat1_0: Array  # prior value of the first natural parameter (α0 - 1)
    nat2_0: Array  # prior value of the second natural parameter (β0 - 1)
    dnat1: Array  # Change in the first natural parameter (α-1)
    dnat2: Array  # Change in the second natural parameter (β-1)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha0: Array = 1.0, beta0: Array = 1.0):
        """Initialize beta distribution with alpha and beta parameters.

        Args:
            alpha0: First shape parameter α (default: 1.0)
            beta0: Second shape parameter β (default: 1.0)
        """
        # Convert to arrays
        alpha = jnp.asarray(alpha0)
        beta = jnp.asarray(beta0)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(alpha), jnp.shape(beta))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        self.nat1_0 = jnp.broadcast_to(alpha - 1, self.batch_shape)  # α-1
        self.nat2_0 = jnp.broadcast_to(beta - 1, self.batch_shape)  # β-1

        # Initialize parameter changes
        self.dnat1 = jnp.zeros(self.batch_shape)
        self.dnat2 = jnp.zeros(self.batch_shape)

    @property
    def nat1(self) -> Array:
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        return self.nat2_0 + self.dnat2

    @property
    def alpha(self) -> Array:
        """Get first shape parameter α."""
        return self.nat1 + 1.0

    @property
    def beta(self) -> Array:
        """Get second shape parameter β."""
        return self.nat2 + 1.0

    @property
    def mean(self) -> Array:
        """Get mean of the distribution."""
        alpha = self.alpha
        beta = self.beta
        return alpha / (alpha + beta)

    @property
    def variance(self) -> Array:
        """Get variance of the distribution."""
        alpha = self.alpha
        beta = self.beta
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [α-1, β-1].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [log(x), log(1-x)].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [log(x), log(1-x)] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([jnp.log(x), jnp.log(1 - x)], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [ψ(α) - ψ(α+β), ψ(β) - ψ(α+β)].

        Returns:
            Expected sufficient statistics [E[log(x)], E[log(1-x)]] with shape:
            batch_shape + (2,)
        """
        alpha = self.alpha
        beta = self.beta
        digamma_sum = jsp.digamma(alpha + beta)

        # E[log(x)]
        expected_log_x = jsp.digamma(alpha) - digamma_sum

        # E[log(1-x)]
        expected_log_1_minus_x = jsp.digamma(beta) - digamma_sum

        return jnp.stack([expected_log_x, expected_log_1_minus_x], axis=-1)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(B(η₁+1, η₂+1)).

        Returns:
            Log normalizer with shape: batch_shape
        """
        alpha = self.alpha
        beta = self.beta
        return jsp.betaln(alpha, beta)

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values.
        """
        return (x > 0) & (x < 1)

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
        return jr.beta(key, self.alpha, self.beta, shape=shape)

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Beta":
        """Create beta distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            Beta distribution.
        """
        alpha = eta[..., 0] + 1.0  # α = η₁ + 1
        beta = eta[..., 1] + 1.0  # β = η₂ + 1

        return cls(alpha0=alpha, beta0=beta)

    @property
    def kl_divergence_from_prior(self) -> Array:
        """Compute KL divergence KL(post||prior).

        Returns:
            KL divergence KL(post||prior) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = jnp.stack([self.nat1_0, self.nat2_0], axis=-1)
        alpha = eta_other[..., 0] + 1
        beta = eta_other[..., 1] + 1
        other_log_normalizer = jsp.betaln(alpha, beta)
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other_log_normalizer + inner_product
