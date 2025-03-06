"""Bayesian Factor Analysis implementation."""

from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

from ..distributions.base import Distribution
from ..distributions.delta import Delta
from ..distributions.gamma import Gamma
from ..distributions.mvn import MultivariateNormal
from ..distributions.mvn_gamma import MultivariateNormalGamma
from ..types import Array, PRNGKey
from .base import Model


def _to_distribution(X: Union[Array, Distribution]) -> Distribution:
    """Convert input to a Distribution if it isn't already.

    Args:
        X: Input data, either an Array or Distribution

    Returns:
        Distribution instance
    """
    if isinstance(X, Distribution):
        return X
    return Delta(X)


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

    # Bayesian Model Reduction parameters
    use_bmr: bool = False
    bmr_threshold: float = 3.0
    bmr_frequency: int = 1  # Apply BMR every N M-steps
    _m_step_counter: int = 0
    _initial_W_prior: Optional[MultivariateNormalGamma] = None

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
        mask = mask = jnp.clip(jnp.arange(self.n_features), max=self.n_components)[..., None] >= jnp.arange(
            self.n_components
        )
        alpha = 2 + (self.n_features - jnp.arange(self.n_components)) / 2
        self.W_dist = MultivariateNormalGamma(loc=loc, mask=mask, alpha=alpha, beta=1.0)

        # Store initial W prior for BMR
        self._initial_W_prior = self.W_dist.copy()

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

    def _e_step(self, X: Union[Array, Distribution], use_data_mask: bool = True) -> MultivariateNormal:
        """E-step: Compute expected latent variables.

        Args:
            X: Data matrix of shape (n_samples, n_features) or Distribution instance
            use_data_mask: apply data mask (default True)

        Returns:
            qz: Posterior distribution over latents z in the form of a MultivariateNormal distribution
        """
        X_dist = _to_distribution(X)
        X_mean = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location
        mask = self._validate_mask(X_mean) if use_data_mask else jnp.ones_like(X_mean, dtype=bool)
        X_centered = X_mean - self.mean_

        # Get current loading matrix and noise precision
        W = self.W_dist.mean.mT  # (n_components, n_features)
        sqrt_noise_precision = jnp.sqrt(self.noise_precision.mean)
        if self.isotropic_noise:
            sqrt_noise_precision = jnp.full(self.n_features, sqrt_noise_precision)

        # Scale noise precision by mask
        sqrt_noise_precision = jnp.where(mask, sqrt_noise_precision, 0.0)

        # Compute posterior parameters
        scaled_W = W * sqrt_noise_precision[..., None, :]
        exp_cov = jnp.sum(mask[..., None, None] * self.W_dist.expected_covariance, -3)
        P = exp_cov + scaled_W @ scaled_W.mT + jnp.eye(self.n_components)
        q, r = jnp.linalg.qr(P)
        q_inv = q.mT

        # Compute expectations
        Ez = solve_triangular(
            r,
            q_inv @ ((X_centered * sqrt_noise_precision)[..., None, :] @ scaled_W.mT).mT,
        )

        Ez = Ez.squeeze(-1)
        qz = MultivariateNormal(loc=Ez, precision=P)

        return qz

    def _m_step(self, X: Union[Array, Distribution], qz: MultivariateNormal) -> "BayesianFactorAnalysis":
        """M-step: Update parameters.

        Args:
            X: Data matrix of shape (n_samples, n_features) or Distribution instance
            qz: Posterior estimates of latent states obtained during the variational e-step (n_samples, n_components)

        Returns:
            Updated model instance
        """
        X_dist = _to_distribution(X)
        exp_stats = X_dist.expected_sufficient_statistics

        dim = X_dist.event_shape[0]
        E_x = exp_stats[..., :dim]
        E_xx = exp_stats[..., dim :: dim + 1]

        mask = self._validate_mask(E_x)
        X_centered = jnp.where(mask, E_x - self.mean_, 0.0)

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
        dnat2 = -0.5 * (jnp.sum(E_xx, axis=0) - jnp.sum(mvn.mean * nat1, -1))

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
        model = eqx.tree_at(
            lambda x: (x.W_dist, x.noise_precision, x._m_step_counter),
            self,
            (new_W_dist, gamma_np, self._m_step_counter + 1),
        )

        # Apply Bayesian Model Reduction if enabled
        if self.use_bmr and (self._m_step_counter % self.bmr_frequency == 0):
            from ..bmr.model_reduction import reduce_model

            # Get current posteriors for loading matrix
            W_posterior = model.W_dist
            W_prior = self._initial_W_prior

            if W_prior is not None:
                # Apply model reduction to find sparse loading matrix
                updated_prior, _ = reduce_model(W_posterior, W_prior, threshold=self.bmr_threshold)

                # Update model with reduced prior
                model = eqx.tree_at(lambda x: x.W_dist, model, updated_prior)

        return model

    def fit(
        self,
        X: Union[Array, Distribution],
        n_iter: int = 100,
        tol: float = 1e-6,
        use_bmr: bool = False,
        bmr_threshold: float = 3.0,
        bmr_frequency: int = 1,
    ) -> "BayesianFactorAnalysis":
        """Fit the model using EM algorithm with optional Bayesian Model Reduction.

        Args:
            X: Training data of shape (n_samples, n_features) or Distribution instance
            n_iter: Maximum number of iterations
            tol: Convergence tolerance
            use_bmr: Whether to use Bayesian Model Reduction to create a sparse loading matrix
            bmr_threshold: Evidence threshold for pruning (higher values = more aggressive pruning)
            bmr_frequency: Apply BMR every N M-steps

        Returns:
            model: Fitted model instance
            elbos: a list of elbo values at each iteration step
        """
        # Convert input to distribution if needed
        X_dist = _to_distribution(X)
        X_mean = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location

        # Create new instance with updated mean and BMR settings
        mask = self._validate_mask(X_mean)
        model = eqx.tree_at(
            lambda x: (x.mean_, x.use_bmr, x.bmr_threshold, x.bmr_frequency, x._m_step_counter),
            self,
            (jnp.sum(mask * X_mean, axis=0) / jnp.sum(mask, axis=0), use_bmr, bmr_threshold, bmr_frequency, 0),
        )

        # EM algorithm
        old_elbo = -jnp.inf
        elbos = []
        for _ in range(n_iter):
            # E-step
            qz = model._e_step(X_dist)
            elbo_val = model.elbo(X_dist, qz)
            elbos.append(elbo_val)

            # M-step (returns updated model)
            model = model._m_step(X_dist, qz)

            # Check convergence
            if jnp.abs(elbo_val - old_elbo) < tol:
                break
            old_elbo = elbo_val

        return model, elbos

    def transform(self, X: Union[Array, Distribution], use_data_mask: bool = False) -> Array:
        """Apply dimensionality reduction to X.

        Args:
            X: Data matrix of shape (n_samples, n_features) or Distribution instance
            use_data_mask: apply data mask (default False)

        Returns:
            qz: Posterior estimate of the latents as MultivariateNormal distribution
        """
        qz = self._e_step(X, use_data_mask=use_data_mask)
        return qz

    def inverse_transform(self, Z: Union[Array, Distribution]) -> Array:
        """Transform latent states back to its original space.

        Args:
            Z: Data in transformed space of shape (n_samples, n_components) or a Distribution

        Returns:
            X_original: Prediction of the data as a MultivariateNormal distribution
        """
        Z_dist = _to_distribution(Z)
        W = self.W_dist.mean.mT
        loc = Z_dist.mean @ W + self.mean_
        covariance = (1 / self.noise_precision.mean)[..., None] * jnp.eye(
            self.n_features
        ) + W.mT @ Z_dist.covariance @ W
        return MultivariateNormal(loc=loc, covariance=covariance)

    def _expected_log_likelihood(self, X_dist: Distribution, qz: MultivariateNormal) -> Array:
        """Compute expected log likelihood E_q[log p(X|Z,W,τ)].

        Args:
            X_dist: Distribution over observations
            qz: Posterior distribution over latent variables

        Returns:
            Expected log likelihood
        """
        # Get parameters
        W = self.W_dist.mean
        exp_stats = self.noise_precision.expected_sufficient_statistics
        exp_log_prec = exp_stats[..., 0]
        exp_noise_precision = exp_stats[..., 1]
        if self.isotropic_noise:
            exp_noise_precision = jnp.full(self.n_features, exp_noise_precision)

        # Get expected sufficient statistics
        dim = X_dist.event_shape[0]
        exp_stats = X_dist.expected_sufficient_statistics
        E_x = exp_stats[..., :dim]
        E_xx = exp_stats[..., dim :: dim + 1]

        # Get mask
        mask = self._validate_mask(E_x)
        n_observed = jnp.sum(mask)

        # Get expectations for Z
        Ez = qz.mean
        Ezz = qz.covariance + Ez[..., None] * Ez[..., None, :]

        # Compute expected log likelihood
        exp_ll = -0.5 * n_observed * jnp.log(2 * jnp.pi)
        exp_ll += 0.5 * jnp.sum(mask * exp_log_prec)

        # E[(x - Wz)^T τ (x - Wz)]
        x_centered = E_x - self.mean_
        term1 = jnp.sum(exp_noise_precision * mask * E_xx)
        term2 = -2 * jnp.sum(exp_noise_precision * mask * (x_centered * (W @ Ez[..., None]).squeeze(-1)))
        term3 = jnp.trace((exp_noise_precision * mask)[..., None] * (W @ Ezz @ W.T), axis1=-1, axis2=-2).sum()

        exp_cov = self.W_dist.expected_covariance
        term4 = jnp.trace(mask[..., None, None] * (exp_cov @ jnp.expand_dims(Ezz, -3)), axis1=-1, axis2=-2).sum()

        exp_ll -= 0.5 * (term1 + term2 + term3 + term4)

        return exp_ll

    def _kl_latent(self, qz: MultivariateNormal) -> Array:
        """Compute KL(q(Z)||p(Z)).

        Args:
            qz: Posterior distribution over latent variables

        Returns:
            KL divergence
        """
        return qz.kl_divergence(
            MultivariateNormal(loc=jnp.zeros_like(qz.mean), precision=jnp.eye(self.n_components))
        ).sum()

    def _kl_loading(self) -> Array:
        """Compute KL(q(W)||p(W)).

        Returns:
            KL divergence
        """
        return self.W_dist.kl_divergence_from_prior.sum()

    def _kl_noise(self) -> Array:
        """Compute KL(q(τ)||p(τ)).

        Returns:
            KL divergence
        """
        return self.noise_precision.kl_divergence_from_prior.sum()

    def elbo(self, X: Union[Array, Distribution], qz: MultivariateNormal) -> Array:
        """Compute Evidence Lower Bound (ELBO).

        Args:
            X: Data matrix or distribution over observations
            qz: Posterior distribution over latent variables

        Returns:
            ELBO value
        """
        X_dist = _to_distribution(X)

        # Expected log likelihood
        exp_ll = self._expected_log_likelihood(X_dist, qz)

        # KL terms
        kl_z = self._kl_latent(qz)
        kl_w = self._kl_loading()
        kl_tau = self._kl_noise()

        return exp_ll - kl_z - kl_w - kl_tau


class PPCA(BayesianFactorAnalysis):
    """Probabilistic Principal Component Analysis."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        random_state: Optional[PRNGKey] = None,
        data_mask: Optional[Array] = None,
    ):
        """Initialize PPCA model.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=True,
            random_state=random_state,
            data_mask=data_mask,
        )


class FactorAnalysis(BayesianFactorAnalysis):
    """Factor Analysis with per-feature noise."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        random_state: Optional[PRNGKey] = None,
        data_mask: Optional[Array] = None,
    ):
        """Initialize Factor Analysis model.

        Args:
            n_components: Number of components
            n_features: Number of features
            random_state: Random state for initialization
            data_mask: Optional boolean array indicating which features are observed (True) or missing (False)
                      Shape should match input data (n_samples, n_features). If None, all features are observed.
        """
        super().__init__(
            n_components=n_components,
            n_features=n_features,
            isotropic_noise=False,
            random_state=random_state,
            data_mask=data_mask,
        )
