"""Multivariate normal distribution implementation."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

from ..types import Array, Matrix, PRNGKey, Shape, Vector
from .exponential_family import ExponentialFamily
from .utils import safe_cholesky, safe_cholesky_and_logdet


class MultivariateNormal(ExponentialFamily):
    """Multivariate normal distribution in natural parameters."""

    nat1: Vector  # First natural parameter (precision * mean)
    nat2: Matrix  # Second natural parameter (-0.5 * precision)

    natural_param_shape: ClassVar[Shape] = (1,)  # [nat1, nat2]

    def __init__(self, dim: int):
        """Initialize multivariate normal with standard parameters.

        Args:
            dim: Dimension of the distribution.
        """
        super().__init__(batch_shape=(), event_shape=(dim,))
        # Initialize with standard normal parameters (mean=0, cov=I)
        self.nat1 = jnp.zeros(dim)  # precision * mean = 0
        self.nat2 = -0.5 * jnp.eye(dim)  # -0.5 * precision = -0.5 * I

    @classmethod
    def from_canonical_parameters(cls, mean: Array, precision: Array) -> "MultivariateNormal":
        """Create MVN from canonical parameters.

        Args:
            mean: Mean vector.
            precision: Precision matrix.

        Returns:
            MultivariateNormal instance.
        """
        dim = mean.shape[-1]
        instance = cls(dim)
        instance.nat1 = precision @ mean
        instance.nat2 = -0.5 * precision
        return instance

    @classmethod
    def from_natural_parameters(cls, nat1: Array, nat2: Array) -> "MultivariateNormal":
        """Create MVN from natural parameters.

        Args:
            nat1: First natural parameter (precision * mean).
            nat2: Second natural parameter (-0.5 * precision).

        Returns:
            MultivariateNormal instance.
        """
        dim = nat1.shape[-1]
        instance = cls(dim)
        instance.nat1 = nat1
        instance.nat2 = nat2
        return instance

    @property
    def mean(self) -> Array:
        """Get mean parameter."""
        return jnp.linalg.solve(self.precision, self.nat1)

    @property
    def precision(self) -> Array:
        """Get precision parameter."""
        return -2.0 * self.nat2

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [x, xx^T].

        Args:
            x: Value to compute sufficient statistics for.

        Returns:
            Sufficient statistics [x, vec(xx^T)].
        """
        xx = x[..., None] * x[..., None, :]  # Outer product
        return jnp.concatenate([x, xx.reshape(*x.shape[:-1], -1)], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [μ, vec(μμ^T + Σ)].

        Returns:
            Expected sufficient statistics [E[x], vec(E[xx^T])].
        """
        precision = -2.0 * self.nat2
        mean = jnp.linalg.solve(precision, self.nat1)
        cov = jnp.linalg.inv(precision)

        d = mean.shape[-1]
        mean_outer = mean.reshape(-1, d, 1) @ mean.reshape(-1, 1, d)
        E_xx = mean_outer + cov

        return jnp.concatenate([mean, E_xx.reshape(*mean.shape[:-1], -1)], axis=-1)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [precision*mean, -0.5*precision].

        Returns:
            Natural parameters [η₁, vec(η₂)].
        """
        return jnp.concatenate([self.nat1, self.nat2.reshape(*self.batch_shape, -1)], axis=-1)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η).

        Returns:
            Log normalizer A(η) with shape: batch_shape
        """
        L, logdet = safe_cholesky_and_logdet(self.precision)
        m = solve_triangular(L, self.nat1, lower=True)

        return 0.25 * jnp.inner(m, m) - 0.5 * logdet

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            Log base measure log(h(x)) with shape: batch_shape
        """
        d = self.event_shape[-1]
        return self.broadcast_to_shape(-0.5 * d * jnp.log(2.0 * jnp.pi), ignore_event=True)

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples from the distribution.
        """
        precision = -2.0 * self.nat2
        mean = jnp.linalg.solve(precision, self.nat1)

        # Use Cholesky for sampling
        L = safe_cholesky(precision)
        L = jnp.broadcast_to(L, sample_shape + L.shape)

        # Generate standard normal samples and transform
        shape = sample_shape + self.shape
        z = jr.normal(key, shape)

        return mean + solve_triangular(L, z, trans=1, lower=True)
