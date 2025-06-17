"""Utility functions for distributions."""

import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve, cho_factor, cho_solve
from jax import jit


from ..types import Array, Scalar


def symmetrize(matrix):
    """Symmetrize one or more matrices."""
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


@jit
def cho_inv(matrix, diagonal_boost=1e-9):
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    chol = cho_factor(symmetrize(matrix) + diagonal_boost * identity, lower=True)
    return cho_solve(chol, identity)


@jit
def safe_cholesky(X: Array, jitter: float = 1e-12) -> tuple[Array, Scalar]:
    """Compute Cholesky decomposition and log determinant with added diagonal jitter.

    Args:
        X: Symmetric positive definite matrix.
        jitter: Small positive value to add to diagonal for stability.

    Returns:
        Tuple of (L, logdet) where L is the lower triangular Cholesky factor
        and logdet is the log determinant of X.
    """
    n = X.shape[-1]
    X = X + jitter * jnp.eye(n)
    L = cholesky(X, lower=True)
    return L


@jit
def safe_cholesky_and_logdet(X: Array, jitter: float = 1e-12) -> tuple[Array, Scalar]:
    """Compute Cholesky decomposition and log determinant with added diagonal jitter.

    Args:
        X: Symmetric positive definite matrix.
        jitter: Small positive value to add to diagonal for stability.

    Returns:
        Tuple of (L, logdet) where L is the lower triangular Cholesky factor
        and logdet is the log determinant of X.
    """
    L = safe_cholesky(X, jitter=jitter)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return L, logdet


@jit
def natural_to_moment(nat1: Array, nat2: Array) -> tuple[Array, Array]:
    """Convert natural parameters to moment parameters for multivariate normal.

    Args:
        nat1: First natural parameter (precision * mean).
        nat2: Second natural parameter (-0.5 * precision).

    Returns:
        Tuple of (mean, covariance).
    """
    precision = -2.0 * nat2
    mean = solve(precision, nat1)
    covariance = solve(precision, jnp.eye(precision.shape[0]))
    return mean, covariance


@jit
def moment_to_natural(mean: Array, covariance: Array) -> tuple[Array, Array]:
    """Convert moment parameters to natural parameters for multivariate normal.

    Args:
        mean: Mean vector.
        covariance: Covariance matrix.

    Returns:
        Tuple of (nat1, nat2).
    """
    precision = solve(covariance, jnp.eye(covariance.shape[0]))
    nat1 = precision @ mean
    nat2 = -0.5 * precision
    return nat1, nat2
