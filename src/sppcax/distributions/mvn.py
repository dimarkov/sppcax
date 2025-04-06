"""Multivariate normal distribution implementation."""

from typing import ClassVar, Optional

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import qr, solve, solve_triangular

from ..types import Array, Matrix, PRNGKey, Shape, Vector
from .exponential_family import ExponentialFamily
from .utils import safe_cholesky, safe_cholesky_and_logdet


def qr_inv(matrix):
    q, r = qr(matrix)
    return solve_triangular(r, q.mT)


class MultivariateNormal(ExponentialFamily):
    """Multivariate normal distribution in natural parameters."""

    nat1: Vector  # First natural parameter (precision * mean)
    nat2: Matrix  # Second natural parameter (-0.5 * precision)
    mask: Vector  # Mask indicating active dimensions.

    natural_param_shape: ClassVar[Shape] = (1,)  # [nat1, nat2]

    def __init__(
        self,
        loc: Array,
        scale_tril: Optional[Array] = None,
        covariance: Optional[Array] = None,
        precision: Optional[Array] = None,
        mask: Optional[Array] = None,
    ):
        """Initialize multivariate normal with standard parameters.

        Args:
            loc: Mean vector with shape (..., d)
            scale_tril: Optional lower triangular scale matrix with shape (..., d, d)
            covariance: Optional covariance matrix with shape (..., d, d)
            precision: Optional precision matrix with shape (..., d, d)
            mask: Optional boolean mask with shape (..., d) where True indicates active dimensions

        Note:
            Only one of scale_tril, covariance, or precision should be provided.
            If none are provided, identity matrix is used as the scale.
        """
        # Get shapes from loc
        *batch_shape, dim = loc.shape
        super().__init__(batch_shape=tuple(batch_shape), event_shape=(dim,))

        # Validate inputs
        scale_params = sum(x is not None for x in [scale_tril, covariance, precision])
        if scale_params > 1:
            raise ValueError("Only one of scale_tril, covariance, or precision should be provided")

        # Process mask
        if mask is not None:
            if mask.shape != loc.shape:
                raise ValueError(f"Mask shape {mask.shape} must match loc shape {loc.shape}")
            self.mask = mask
        else:
            self.mask = jnp.ones_like(loc, dtype=bool)

        # Compute precision matrix
        if precision is not None:
            precision = jnp.broadcast_to(precision, (*batch_shape, dim, dim))
            if precision.shape != (*batch_shape, dim, dim):
                raise ValueError(f"Precision shape {precision.shape} must match loc batch shape")
            P = precision
        elif covariance is not None:
            covariance = jnp.broadcast_to(covariance, (*batch_shape, dim, dim))
            if covariance.shape != (*batch_shape, dim, dim):
                raise ValueError(f"Covariance shape {covariance.shape} must match loc batch shape")
            P = qr_inv(covariance)
        elif scale_tril is not None:
            scale_tril = jnp.broadcast_to(scale_tril, (*batch_shape, dim, dim))
            if scale_tril.shape != (*batch_shape, dim, dim):
                raise ValueError(f"Scale_tril shape {scale_tril.shape} must match loc batch shape")
            P = qr_inv(scale_tril @ scale_tril.mT)
        else:
            # Default to identity matrix with proper broadcasting
            P = jnp.broadcast_to(jnp.eye(dim), (*batch_shape, dim, dim))

        # Apply mask to precision and loc
        P = self._apply_mask_matrix(P)
        masked_loc = self._apply_mask_vector(loc)

        # Set natural parameters
        self.nat1 = P @ masked_loc[..., None]
        self.nat1 = self.nat1[..., 0]  # Remove singleton dimension
        self.nat2 = -0.5 * P

    def _apply_mask_vector(self, x: Array) -> Array:
        """Apply mask to a vector, zeroing out masked dimensions.

        Args:
            x: Vector with shape (..., d)

        Returns:
            Masked vector with same shape
        """
        return jnp.where(self.mask, x, 0.0)

    def _apply_mask_matrix(self, x: Array, zeromask: bool = False) -> Array:
        """Apply mask to a matrix, zeroing out masked rows and columns.

        Args:
            x: Matrix with shape (..., d, d)

        Returns:
            Masked matrix with same shape
        """
        mask_mat = self.mask[..., None] * self.mask[..., None, :]

        if zeromask:
            return jnp.where(mask_mat, x, 0.0)
        else:
            return jnp.where(mask_mat, x, jnp.eye(x.shape[-1]))

    @classmethod
    def from_natural_parameters(cls, nat1: Array, nat2: Array, mask: Optional[Array] = None) -> "MultivariateNormal":
        """Create MVN from natural parameters.

        Args:
            nat1: First natural parameter (precision * mean).
            nat2: Second natural parameter (-0.5 * precision).
            mask: Optional boolean mask with shape matching nat1

        Returns:
            MultivariateNormal instance.
        """
        precision = -2.0 * nat2
        loc = solve(precision, nat1, assume_a="pos")
        return cls(loc=loc, precision=precision, mask=mask)

    @property
    def mean(self) -> Array:
        """Get mean parameter."""
        mean = solve(self.precision, self.nat1, assume_a="pos")
        return self._apply_mask_vector(mean)

    @property
    def covariance(self) -> Array:
        return self._apply_mask_matrix(qr_inv(-2.0 * self.nat2), zeromask=True)

    @property
    def precision(self) -> Array:
        """Get precision parameter."""
        return self._apply_mask_matrix(-2.0 * self.nat2)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [x, xx^T].

        Args:
            x: Value to compute sufficient statistics for.

        Returns:
            Sufficient statistics [x, vec(xx^T)].
        """
        x = self._apply_mask_vector(x)
        xx = x[..., None] * x[..., None, :]  # Outer product
        return jnp.concatenate([x, xx.reshape(*x.shape[:-1], -1)], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [μ, vec(μμ^T + Σ)].

        Returns:
            Expected sufficient statistics [E[x], vec(E[xx^T])].
        """

        E_x = self.mean
        cov = self.covariance

        mean_outer = E_x[..., None] * E_x[..., None, :]
        E_xx = mean_outer + cov

        return jnp.concatenate([E_x, E_xx.reshape(*E_x.shape[:-1], -1)], axis=-1)

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [precision*mean, -0.5*precision].

        Returns:
            Natural parameters [η₁, vec(η₂)].
        """
        return jnp.concatenate([self.nat1, -0.5 * self.precision.reshape(*self.batch_shape, -1)], axis=-1)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η).

        Returns:
            Log normalizer A(η) with shape: batch_shape
        """
        precision = self.precision
        L, logdet = safe_cholesky_and_logdet(precision)
        m = self._apply_mask_vector(solve_triangular(L, self.nat1[..., None], lower=True)[..., 0])

        return 0.5 * jnp.sum(jnp.square(m), -1) - 0.5 * logdet

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            Log base measure log(h(x)) with shape: batch_shape
        """
        d = jnp.sum(self.mask, axis=-1)  # Count active dimensions
        return self.broadcast_to_shape(-0.5 * d * jnp.log(2.0 * jnp.pi), ignore_event=True)

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples from the distribution.
        """
        precision = self.precision
        mean = jnp.linalg.solve(precision, self.nat1[..., None])[..., 0]

        # Use Cholesky for sampling
        L = safe_cholesky(precision)
        L = jnp.broadcast_to(L, sample_shape + L.shape)

        # Generate standard normal samples and transform
        shape = sample_shape + self.shape
        z = jr.normal(key, shape)
        samples = mean + solve_triangular(L, z[..., None], trans=1, lower=True)[..., 0]

        return self._apply_mask_vector(samples)
