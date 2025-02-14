"""Bayesian Factor Analysis implementation."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

from ..distributions.gamma import Gamma
from ..distributions.mvn import MultivariateNormal
from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array, PRNGKey
from .base import Model


class BayesianFactorAnalysis(Model):
    """Base class for Bayesian Factor Analysis models."""

    n_components: int
    n_features: int
    isotropic_noise: bool
    W_dist: MultivariateNormalGamma  # batched over features
    noise_precision: Gamma  # Single precision for PPCA or per-feature for FA
    mean_: Array  # Data mean for centering
    data_mask: Optional[Array] = None  # Mask for missing data (True for observed, False for missing)
    random_state: Optional[PRNGKey] = None

    def __init__(
        self,
        n_components: int,
        n_features: int,
        isotropic_noise: bool = False,
        data_mask: Optional[Array] = None,
        random_state: Optional[PRNGKey] = None,
    ):
        """Initialize BayesianFactorAnalysis model.

        Args:
            n_components: Number of components
            n_features: Number of features
            isotropic_noise: If True, use same noise precision for all features (PPCA)
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
            random_state: Random state for initialization
        """
        self.n_components = n_components
        self.n_features = n_features
        self.isotropic_noise = isotropic_noise
        self.random_state = random_state
        self.data_mask = data_mask

        # Initialize mean
        self.mean_ = jnp.zeros(n_features)

        # Initialize parameters
        self._init_params()

    def _init_params(self) -> None:
        """Initialize model parameters."""
        key = self.random_state
        if key is None:
            key = jr.PRNGKey(0)

        # Initialize loading matrix columns
        loc = jr.normal(key, (self.n_features, self.n_components)) * 0.01
        mask = mask = jnp.clip(jnp.arange(self.n_features), a_max=self.n_components)[..., None] >= jnp.arange(
            self.n_components
        )
        alpha = 2 + (self.n_features - jnp.arange(self.n_components)) / 2
        self.W_dist = MultivariateNormalGamma(loc=loc, mask=mask, alpha=alpha, beta=1.0)

        # Initialize noise precision
        if self.isotropic_noise:
            # Single precision for all features (PPCA)
            self.noise_precision = Gamma(alpha0=2.0, beta0=1.0)
        else:
            # Per-feature precision (FA)
            self.noise_precision = Gamma(alpha0=2 * jnp.ones(self.n_features), beta0=jnp.ones(self.n_features))

    def _validate_mask(self, X: Array) -> Array:
        """Validate and process the data mask.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Validated mask of shape (n_samples, n_features)
        """
        if self.data_mask is None:
            return jnp.ones_like(X, dtype=bool)

        if self.data_mask.shape != X.shape:
            raise ValueError(f"data_mask shape {self.data_mask.shape} does not match data shape {X.shape}")

        return self.data_mask

    def _e_step(self, X: Array) -> MultivariateNormal:
        """E-step: Compute expected latent variables.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Tuple of:
                Ez: Expected factors E[z|x] of shape (n_samples, n_components)
                Ezz: Expected factor outer products E[zz'|x] of shape (n_components, n_components)
        """
        mask = self._validate_mask(X)
        X_centered = X - self.mean_

        # Get current loading matrix and noise precision
        W = self.W_dist.mean.mT  # (n_components, n_features)
        sqrt_noise_precision = jnp.sqrt(self.noise_precision.mean)
        if self.isotropic_noise:
            sqrt_noise_precision = jnp.full(self.n_features, sqrt_noise_precision)

        # Scale noise precision by mask
        sqrt_noise_precision = jnp.where(mask, sqrt_noise_precision, 0.0)

        # Compute posterior parameters
        scaled_W = W * sqrt_noise_precision
        exp_cov = jnp.sum(self.W_dist.expected_covariance, 0)
        P = exp_cov + scaled_W @ scaled_W.mT + jnp.eye(self.n_components)
        q, r = jnp.linalg.qr(P)
        q_inv = q.mT

        # Compute expectations
        _r = jnp.broadcast_to(r, X_centered.shape[:-1] + r.shape)
        Ez = solve_triangular(
            _r,
            q_inv @ jnp.expand_dims((X_centered * sqrt_noise_precision) @ scaled_W.mT, -1),
        )

        Ez = Ez.squeeze(-1)
        qz = MultivariateNormal(loc=Ez, precision=P)

        return qz

    def _m_step(self, X: Array, qz: MultivariateNormal) -> "BayesianFactorAnalysis":
        mask = self._validate_mask(X)
        """M-step: Update parameters.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            qz: Posterior estimates of latent states obtained during the variational e-step (n_samples, n_components)

        Returns:
            Updated model instance
        """
        X_centered = jnp.where(mask, X - self.mean_, 0.0)

        exp_stats = qz.expected_sufficient_statistics
        Ez = exp_stats[..., : self.n_components]
        Ezz = jnp.sum(exp_stats[..., self.n_components :], 0).reshape(self.n_components, self.n_components)

        # Update loading matrix
        P = jnp.diag(self.W_dist.gamma.mean) + Ezz
        nat1 = jnp.sum(Ez[..., None, :] * X_centered[..., None], axis=0)
        mvn = eqx.tree_at(lambda x: (x.nat1, x.nat2), self.W_dist.mvn, (nat1, -0.5 * P))

        # update noise precision
        n_observed = jnp.sum(mask, axis=0)
        dnat1 = 0.5 * n_observed
        dnat2 = -0.5 * (jnp.sum(jnp.square(X_centered), axis=0) - jnp.sum(mvn.mean * nat1, -1))

        if self.isotropic_noise:
            # Single precision for all features
            dnat1 = jnp.sum(dnat1)
            dnat2 = jnp.sum(dnat2)

        gamma_np = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), self.noise_precision, (dnat1, dnat2))

        # update tau
        W = mvn.mean
        dnat2 = -0.5 * jnp.diag(mvn.covariance.sum(0) + (W.mT * gamma_np.mean) @ W)

        # Create new W distribution
        gamma_tau = eqx.tree_at(lambda x: x.dnat2, self.W_dist.gamma, dnat2)
        new_W_dist = eqx.tree_at(lambda x: (x.mvn, x.gamma), self.W_dist, (mvn, gamma_tau))

        # Update model with new W distribution
        model = eqx.tree_at(lambda x: (x.W_dist, x.noise_precision), self, (new_W_dist, gamma_np))

        return model

    def fit(self, X: Array, n_iter: int = 100, tol: float = 1e-6) -> "BayesianFactorAnalysis":
        """Fit the model using EM algorithm.

        Args:
            X: Training data of shape (n_samples, n_features)
            n_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Fitted model instance
        """
        # Create new instance with updated mean
        mask = self._validate_mask(X)
        model = eqx.tree_at(lambda x: x.mean_, self, jnp.sum(mask * X, axis=0) / jnp.sum(mask, axis=0))

        # EM algorithm
        old_ll = -jnp.inf
        lls = []
        for _ in range(n_iter):
            # E-step
            qz = model._e_step(X)

            # M-step (returns updated model)
            model = model._m_step(X, qz)

            # Check convergence
            ll = model.score(X, qz.mean)
            lls.append(ll)
            if jnp.abs(ll - old_ll) < tol:
                break
            old_ll = ll

        return model, lls

    def transform(self, X: Array) -> Array:
        """Apply dimensionality reduction to X.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            X_new: Transformed data of shape (n_samples, n_components)
        """
        qz = self._e_step(X)
        return qz

    def inverse_transform(self, Z: Array) -> Array:
        """Transform data back to its original space.

        Args:
            Z: Data in transformed space of shape (n_samples, n_components)

        Returns:
            X_original: Data in original space of shape (n_samples, n_features)
        """
        return Z @ self.W_dist.mean.T + self.mean_

    def score(self, X: Array, Ez: Array) -> Array:
        """Compute the log likelihood of X under the model.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            ll: Log likelihood
        """
        # Get current parameters
        W = self.W_dist.mean
        noise_precision = self.noise_precision.mean
        if self.isotropic_noise:
            noise_precision = jnp.full(self.n_features, noise_precision)

        # Get mask and compute number of observed values
        mask = self._validate_mask(X)
        n_observed = jnp.sum(mask)

        # Compute log likelihood for observed data only
        X_centered = X - self.mean_
        reconstruction = Ez @ W.mT

        # Data term
        ll = -0.5 * n_observed * jnp.log(2 * jnp.pi)
        ll += 0.5 * jnp.sum(mask * jnp.log(noise_precision))
        ll -= 0.5 * jnp.sum(noise_precision * mask * (X_centered - reconstruction) ** 2)

        return ll


class PPCA(BayesianFactorAnalysis):
    """Probabilistic Principal Component Analysis."""

    def __init__(self, n_components: int, n_features: int, random_state: Optional[PRNGKey] = None):
        """Initialize PPCA model.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
        """
        super().__init__(
            n_components=n_components, n_features=n_features, isotropic_noise=True, random_state=random_state
        )


class FactorAnalysis(BayesianFactorAnalysis):
    """Factor Analysis with per-feature noise."""

    def __init__(self, n_components: int, n_features: int, random_state: Optional[PRNGKey] = None):
        """Initialize Factor Analysis model.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
        """
        super().__init__(
            n_components=n_components, n_features=n_features, isotropic_noise=False, random_state=random_state
        )
