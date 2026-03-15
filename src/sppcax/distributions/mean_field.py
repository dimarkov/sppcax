"""Mean-field composite distribution with independent components."""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jr

from ..types import Array, PRNGKey, Shape
from .base import Distribution
from .delta import Delta
from .gamma import InverseGamma


class MeanField(Distribution):
    """Mean-field composite distribution: q(W, Sigma) = q(W) x q(Sigma).

    Components are independent. Posterior updates use coordinate ascent
    (alternating updates of weights and noise components).

    Attributes:
        weights: Distribution over weight parameters (MVN or Delta if frozen).
        noise: Distribution over noise parameters (InverseGamma, Delta, etc.).
        n_iter: Number of coordinate ascent iterations for posterior updates.
    """

    weights: Distribution
    noise: Distribution

    def __init__(self, weights: Distribution, noise: Distribution):
        """Initialize MeanField composite distribution.

        Args:
            weights: Distribution over weight parameters.
            noise: Distribution over noise/covariance parameters.
            n_iter: Number of coordinate ascent iterations (default: 4).
        """
        super().__init__(
            batch_shape=weights.batch_shape,
            event_shape=weights.event_shape,
        )
        self.weights = weights
        self.noise = noise

    # --- Compatibility properties (drop-in for MVNIG/MVN/NIW) ---

    @property
    def mean(self) -> Array:
        """Mean of the weights component."""
        return self.weights.mean

    @property
    def precision(self) -> Array:
        """Precision of the weights component."""
        return self.weights.precision

    @property
    def covariance(self) -> Array:
        """Covariance of the weights component (base, not scaled by noise)."""
        return self.weights.covariance

    @property
    def col_covariance(self) -> Array:
        """Column covariance (base covariance, same as covariance)."""
        return self.weights.covariance

    @property
    def expected_psi(self) -> Array:
        """Expected noise precision E[1/sigma^2] from noise component."""
        return self.noise.expected_precision

    @property
    def expected_log_psi(self) -> Array:
        """Expected log noise precision from noise component."""
        return self.noise.expected_log_precision

    @property
    def expected_covariance(self) -> Array:
        """Expected covariance E[sigma^2] * base_covariance."""
        if isinstance(self.noise, Delta):
            exp_var = self.noise.mean
        elif isinstance(self.noise, InverseGamma):
            exp_var = jnp.broadcast_to(self.noise.mean, self.batch_shape)
        else:
            exp_var = jnp.broadcast_to(1.0 / self.noise.expected_precision, self.batch_shape)
        return self.weights.covariance * exp_var[..., None, None]

    @property
    def mvn(self) -> Distribution:
        """Weights component (for ARD compatibility)."""
        return self.weights

    @property
    def mask(self) -> Array:
        """Mask from weights component (if available)."""
        if hasattr(self.weights, "mask"):
            return self.weights.mask
        return None

    @property
    def inv_gamma(self):
        """InverseGamma component (for MVNIG compatibility)."""
        return self.noise

    @property
    def alpha(self) -> Array:
        """Shape parameter from InverseGamma noise (MVNIG compat)."""
        return self.noise.alpha

    @property
    def beta(self) -> Array:
        """Scale parameter from InverseGamma noise (MVNIG compat)."""
        return self.noise.beta

    @property
    def expected_sufficient_statistics_psi(self) -> Array:
        """Expected sufficient statistics of noise precision (MVNIG compat)."""
        if isinstance(self.noise, InverseGamma):
            from .gamma import Gamma

            gamma = Gamma(self.noise.alpha, self.noise.beta)
            suff_stats = gamma.expected_sufficient_statistics
            return jnp.broadcast_to(suff_stats, self.batch_shape + (2,))
        return None

    # --- Methods ---

    def mode(self) -> Tuple[Array, Array]:
        """Compute joint mode (noise_cov_matrix, weights_mean).

        Returns:
            Tuple of (noise covariance as matrix, weights mean).
        """
        mean = self.weights.mean

        if isinstance(self.noise, InverseGamma):
            dim = self.event_shape[-1]
            sigma_sqr_mode = self.noise.beta / (self.noise.alpha + (dim + 2) / 2)
            n = self.batch_shape[0] if self.batch_shape else 1
            cov = jnp.eye(n) * jnp.broadcast_to(sigma_sqr_mode, (n,))[:, None]
        elif isinstance(self.noise, Delta):
            cov = self.noise.mean
            # Ensure matrix form
            if cov.ndim == 1:
                cov = jnp.diag(cov)
            elif cov.ndim == 0:
                n = self.batch_shape[0] if self.batch_shape else 1
                cov = cov * jnp.eye(n)
        else:
            # Generic: use mean as mode approximation
            cov = self.noise.mean
            if hasattr(cov, "ndim") and cov.ndim < 2:
                n = self.batch_shape[0] if self.batch_shape else 1
                cov = jnp.diag(jnp.broadcast_to(cov, (n,)))

        return cov, mean

    def sample(self, seed: PRNGKey, sample_shape: Shape = ()) -> Tuple[Array, Array]:
        """Sample from both components independently.

        Args:
            seed: PRNG key.
            sample_shape: Additional sample dimensions.

        Returns:
            Tuple of (noise_sample, weights_sample).
        """
        key_w, key_n = jr.split(seed)

        weights_sample = self.weights.sample(key_w, sample_shape)

        if isinstance(self.noise, InverseGamma):
            noise_sample = self.noise.sample(key_n, sample_shape)
            n = self.batch_shape[0] if self.batch_shape else 1
            noise_cov = jnp.eye(n) * jnp.broadcast_to(noise_sample, (n,))[:, None]
        elif isinstance(self.noise, Delta):
            noise_cov = self.noise.mean
            if noise_cov.ndim == 1:
                noise_cov = jnp.diag(noise_cov)
            elif noise_cov.ndim == 0:
                n = self.batch_shape[0] if self.batch_shape else 1
                noise_cov = noise_cov * jnp.eye(n)
        else:
            noise_sample = self.noise.sample(key_n, sample_shape)
            noise_cov = noise_sample

        return noise_cov, weights_sample

    def log_prob(self, x) -> Array:
        """Log probability (sum of component log-probs)."""
        raise NotImplementedError("MeanField log_prob requires separate component values")

    def entropy(self) -> Array:
        """Entropy of the mean-field distribution (sum of component entropies)."""
        h_w = self.weights.entropy() if hasattr(self.weights, "entropy") else jnp.zeros(())
        h_n = self.noise.entropy() if hasattr(self.noise, "entropy") else jnp.zeros(())
        return h_w + h_n
