"""Mean-field composite distribution with independent components."""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jr

from ..types import Array, PRNGKey, Shape
from .base import Distribution
from .delta import Delta
from .gamma import InverseGamma
from .inverse_wishart import InverseWishart


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
        return self.noise.expected_psi

    @property
    def expected_covariance(self) -> Array:
        """Expected covariance E[sigma^2] * base_covariance.

        For InverseGamma noise: scalar E[sigma^2] per row.
        For InverseWishart noise: full matrix E[Sigma], not factored with weights cov.
        For Delta noise: fixed value.
        """

        if isinstance(self.noise, Delta):
            exp_var = self.noise.mean
        elif isinstance(self.noise, InverseGamma):
            exp_var = jnp.broadcast_to(self.noise.mean, self.batch_shape)
        elif isinstance(self.noise, InverseWishart):
            # IW noise: E[Sigma] is already a full (k,k) matrix
            exp_var = jnp.diag(self.noise.mean)
        else:
            raise NotImplementedError

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
        cov = self.noise.mode()
        if cov.ndim < 2:
            n = self.batch_shape[0]
            cov = jnp.broadcast_to(cov, (n,))

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
        noise_sample = self.noise.sample(key_n, sample_shape)
        if noise_sample.ndim < 1:
            n = self.batch_shape[0] if self.batch_shape else 1
            noise_cov = jnp.broadcast(noise_sample, (n,))
        else:
            noise_cov = noise_sample

        return noise_cov, weights_sample

    def log_prob(self, x: Tuple[Array, Array]) -> Array:
        """Compute log probability.

        Args:
            x: Tuple of (cov, w) where:
                w: Value of the sample state
                cov: Value of the sample covariance

        Returns:
            Log probability
        """
        return self.noise.log_prob(x[0]) + self.noise.log_prob(x[1])

    def entropy(self) -> Array:
        """Entropy of the mean-field distribution (sum of component entropies)."""
        h_w = self.weights.entropy() if hasattr(self.weights, "entropy") else jnp.zeros(())
        h_n = self.noise.entropy() if hasattr(self.noise, "entropy") else jnp.zeros(())
        return h_w + h_n
