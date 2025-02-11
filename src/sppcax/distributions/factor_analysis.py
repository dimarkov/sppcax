"""Factor analysis distributions module.

This module provides implementations of distributions specifically designed for
factor analysis models, including the Matrix Normal-Gamma distribution for
loading matrices.
"""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

from ..types import Array, Matrix, PRNGKey, Scalar, Shape
from .exponential_family import ExponentialFamily


class MatrixNormalGamma(ExponentialFamily):
    """Matrix Normal-Gamma distribution for factor analysis.

    This distribution combines:
    p(W|τ) = MN(W; 0, I, τ⁻¹I)  # Loading matrix
    p(τ) = Gamma(τ; α, β)        # Precision

    This is specifically designed as a prior for the loading matrix in
    factor analysis where:
    - W is the p×k loading matrix
    - τ is the precision parameter
    - Row covariance is identity (feature dimension)
    - Column covariance is scaled by τ⁻¹ (factor dimension)
    """

    # Natural parameters for W
    nat1: Matrix  # First natural parameter for W
    nat2: Matrix  # Second natural parameter for W (row precision)
    nat3: Matrix  # Third natural parameter for W (column precision base)

    # Natural parameters for τ
    gamma_nat1: Scalar  # First natural parameter for Gamma (α-1)
    gamma_nat2: Scalar  # Second natural parameter for Gamma (-β)

    def __init__(self, n_features: int, n_factors: int, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Matrix Normal-Gamma distribution.

        Args:
            n_features: Number of features (rows of W)
            n_factors: Number of factors (columns of W)
            alpha: Shape parameter for Gamma prior
            beta: Rate parameter for Gamma prior
        """
        # Initialize W's natural parameters
        # W ~ MN(0, I, τ⁻¹I)
        self.nat1 = jnp.zeros((n_features, n_factors))  # Mean parameter
        self.nat2 = -0.5 * jnp.eye(n_features)  # Row precision (fixed)
        self.nat3 = -0.5 * jnp.eye(n_factors)  # Base column precision

        # Initialize τ's natural parameters
        # τ ~ Gamma(α, β)
        self.gamma_nat1 = jnp.array(alpha - 1.0)  # α-1
        self.gamma_nat2 = jnp.array(-beta)  # -β

    def natural_parameters(self) -> Array:
        """Get natural parameters.

        Returns:
            Concatenated natural parameters [vec(W_params), τ_params]
        """
        W_params = jnp.concatenate([self.nat1.ravel(), self.nat2.ravel(), self.nat3.ravel()])
        tau_params = jnp.array([self.gamma_nat1, self.gamma_nat2])
        return jnp.concatenate([W_params, tau_params])

    def sufficient_statistics(self, x: Tuple[Array, Array]) -> Array:
        """Compute sufficient statistics.

        Args:
            x: Tuple of (W, τ) where:
                W: Loading matrix of shape (n_features, n_factors)
                τ: Precision scalar

        Returns:
            Sufficient statistics
        """
        W, tau = x
        W_stats = jnp.concatenate([W.ravel(), (W @ W.T).ravel(), (W.T @ W).ravel()])
        tau_stats = jnp.array([jnp.log(tau), tau])
        return jnp.concatenate([W_stats, tau_stats])

    def log_prob(self, x: Tuple[Array, Array]) -> Array:
        """Compute log probability.

        Args:
            x: Tuple of (W, τ) where:
                W: Loading matrix of shape (n_features, n_factors)
                τ: Precision scalar

        Returns:
            Log probability log p(W,τ)
        """
        W, tau = x
        p, k = W.shape

        # Matrix Normal term: p(W|τ)
        # log p(W|τ) = -0.5 * (pk*log(2π) + k*log|I| + p*log|τ⁻¹I| + τ*tr(W'W))
        matrix_normal_term = -0.5 * (p * k * jnp.log(2.0 * jnp.pi) + p * jnp.log(1.0 / tau) + tau * jnp.sum(W * W))

        # Gamma term: p(τ)
        # log p(τ) = α*log(β) + (α-1)*log(τ) - βτ - log(Γ(α))
        alpha = self.gamma_nat1 + 1.0
        beta = -self.gamma_nat2
        gamma_term = alpha * jnp.log(beta) + (alpha - 1.0) * jnp.log(tau) - beta * tau - jsp.gammaln(alpha)

        return matrix_normal_term + gamma_term

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Tuple[Array, Array]:
        """Sample from the distribution.

        Args:
            key: PRNG key
            sample_shape: Shape of samples to draw

        Returns:
            Tuple of (W, τ) samples
        """
        p, k = self.nat1.shape
        key_tau, key_W = jr.split(key)

        # Sample τ ~ Gamma(α, β)
        alpha = self.gamma_nat1 + 1.0
        beta = -self.gamma_nat2
        tau = jr.gamma(key_tau, alpha, shape=sample_shape) / beta

        # Sample W|τ ~ MN(0, I, τ⁻¹I)
        shape = sample_shape + (p, k)
        W = jr.normal(key_W, shape) / jnp.sqrt(tau)[..., None, None]

        return W, tau

    def expected_sufficient_statistics(self) -> Array:
        """Compute expected sufficient statistics.

        For factor analysis prior:
        E[W|τ] = 0
        E[WW'|τ] = (p/τ)I
        E[W'W|τ] = (k/τ)I
        E[log(τ)] = ψ(α) - log(β)
        E[τ] = α/β

        Returns:
            Expected sufficient statistics
        """
        p, k = self.nat1.shape
        alpha = self.gamma_nat1 + 1.0
        beta = -self.gamma_nat2

        # Compute E[τ] and E[log(τ)]
        E_tau = alpha / beta
        E_log_tau = jsp.digamma(alpha) - jnp.log(beta)

        # Compute expectations involving W
        E_W = jnp.zeros((p, k))
        E_WWT = jnp.eye(p) * (k / E_tau)
        E_WTW = jnp.eye(k) * (p / E_tau)

        W_stats = jnp.concatenate([E_W.ravel(), E_WWT.ravel(), E_WTW.ravel()])
        tau_stats = jnp.array([E_log_tau, E_tau])

        return jnp.concatenate([W_stats, tau_stats])

    def update_from_sufficient_statistics(
        self,
        n_samples: int,
        Ez: Array,  # Expected factors (n×k)
        Ezz: Array,  # Expected factor outer products (k×k)
        y: Array,  # Observations (n×p)
    ) -> "MatrixNormalGamma":
        """Update parameters using factor analysis sufficient statistics.

        Args:
            n_samples: Number of observations
            Ez: Expected factors E[z|y] of shape (n_samples, n_factors)
            Ezz: Expected factor outer products E[zz'|y] of shape (n_factors, n_factors)
            y: Observations of shape (n_samples, n_features)

        Returns:
            Updated distribution
        """
        p, k = self.nat1.shape

        # Update W's parameters
        W_precision = jnp.eye(p) + Ezz  # Row precision remains I
        W_mean = y.T @ Ez @ jnp.linalg.inv(W_precision)

        # Update τ's parameters
        alpha = self.gamma_nat1 + 1.0
        beta = -self.gamma_nat2

        new_alpha = alpha + (n_samples * p) / 2.0
        new_beta = beta + 0.5 * jnp.sum(y * y - 2.0 * (y @ W_mean) * Ez + jnp.trace(W_mean.T @ W_mean @ Ezz))

        # Create new instance with updated parameters
        new_dist = MatrixNormalGamma(p, k)
        new_dist.nat1 = W_mean
        new_dist.nat2 = -0.5 * jnp.eye(p)
        new_dist.nat3 = -0.5 * jnp.eye(k)
        new_dist.gamma_nat1 = new_alpha - 1.0
        new_dist.gamma_nat2 = -new_beta

        return new_dist
