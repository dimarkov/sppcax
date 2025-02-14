"""Mixed Likelihood Variational Bayesian PCA."""

from typing import Dict, List, Optional

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from ..distributions import ExponentialFamily, MultivariateNormal
from ..types import Array, PRNGKey
from .base import Model
from .likelihood_utils import compute_expected_log_likelihood, parse_feature_groups


class MixedLikelihoodVBPCA(Model):
    """Variational Bayesian PCA with mixed likelihoods.

    Supports different likelihood functions for different features,
    enabling joint modeling of continuous, binary, and categorical data.
    """

    n_components: int
    feature_types: Dict[str, List[int]]
    likelihoods: Dict[int, ExponentialFamily]

    # Variational parameters
    W_: MultivariateNormal  # Loading matrix distribution
    z_: MultivariateNormal  # Latent variable distribution

    random_state: Optional[PRNGKey] = None

    def __init__(self, n_components: int, feature_types: Dict[str, List[int]], random_state: Optional[PRNGKey] = None):
        """Initialize mixed likelihood VBPCA.

        Args:
            n_components: Number of latent components.
            feature_types: Dictionary mapping likelihood type to feature indices.
                Example: {'normal': [0, 1], 'bernoulli': [2, 3]}
            random_state: Random state for initialization.
        """
        self.n_components = n_components
        self.feature_types = feature_types
        self.random_state = random_state

        # Create likelihood objects
        self.likelihoods = parse_feature_groups(feature_types)

        # Get total number of features
        n_features = max(max(indices) for indices in feature_types.values()) + 1

        self._init_variational_params(n_features)

    def _init_variational_params(self, n_features: int) -> None:
        """Initialize variational parameters.

        Args:
            n_features: Number of features in data.
        """
        key = self.random_state
        if key is None:
            key = jr.PRNGKey(0)

        # Initialize loading matrix distribution
        key, subkey = jr.split(key)
        W_mean = jr.normal(subkey, (n_features, self.n_components)) * 0.01
        W_cov = jnp.eye(self.n_components)
        self.W_ = MultivariateNormal(W_mean, W_cov)

        # Initialize latent variable distribution
        z_mean = jnp.zeros((1, self.n_components))  # Will be updated in fit
        z_cov = jnp.eye(self.n_components)
        self.z_ = MultivariateNormal(z_mean, z_cov)

    def _e_step(self, X: Array) -> None:
        """Update variational distribution q(Z).

        Args:
            X: Data matrix of shape (n_samples, n_features).
        """
        n_samples = X.shape[0]

        # initial q(z) set to prior
        qz = MultivariateNormal(loc=jnp.zeros((n_samples, self.n_components)))

        # Update for each feature's likelihood
        for j, likelihood in self.likelihoods.items():
            # Get feature data
            x_j = X[j]

            # Compute expected natural parameters
            params = self.expected_natural_parametes(j)

            # update qz
            qz = likelihood.update_latents(x_j, params, qz)

        # Update distribution
        return qz

    def _update_w_distribution(self, X: Array) -> None:
        """Update variational distribution q(W).

        Args:
            X: Data matrix of shape (n_samples, n_features).
        """
        n_samples = X.shape[0]

        # Get expected sufficient statistics
        z_mean = self.z_.mean
        z_cov = self.z_.covariance

        # Compute expected outer product
        S = jnp.dot(z_mean.T, z_mean) + n_samples * z_cov

        # Update for each feature
        for j, likelihood in self.likelihoods.items():
            # Get feature data
            x_j = X[:, j]

            # Compute expected sufficient statistics
            # nat_params = likelihood.natural_parameters()

            # Update covariance (same for all features)
            W_cov = jnp.linalg.inv(jnp.eye(self.n_components) + S)

            # Update mean
            W_mean_j = jnp.dot(jnp.dot(x_j, z_mean), W_cov)

            # Update distribution for this feature
            self.W_.mean = self.W_.mean.at[j].set(W_mean_j)
            self.W_.covariance = W_cov

    def _compute_elbo(self, X: Array) -> float:
        """Compute evidence lower bound (ELBO).

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            ELBO value.
        """
        # Expected log likelihood
        exp_ll = compute_expected_log_likelihood(
            X, self.likelihoods, self.W_.mean, self.W_.covariance, self.z_.mean, self.z_.covariance
        )

        # KL divergence terms
        kl_w = self.W_.kl_divergence(MultivariateNormal(jnp.zeros_like(self.W_.mean), jnp.eye(self.n_components)))
        kl_z = self.z_.kl_divergence(MultivariateNormal(jnp.zeros_like(self.z_.mean), jnp.eye(self.n_components)))

        return exp_ll - kl_w - kl_z

    def fit(self, X: Array, n_iter: int = 100, tol: float = 1e-6) -> "MixedLikelihoodVBPCA":
        """Fit the model using variational EM.

        Args:
            X: Training data of shape (n_samples, n_features).
            n_iter: Maximum number of iterations.
            tol: Convergence tolerance for ELBO.

        Returns:
            self: Fitted model.
        """
        # Initialize latent variables for actual data size
        n_samples = X.shape[0]
        self.z_ = MultivariateNormal(jnp.zeros((n_samples, self.n_components)), jnp.eye(self.n_components))

        # Run variational EM
        old_elbo = -jnp.inf
        for _ in range(n_iter):
            # E-step: Update q(Z)
            qz = self._e_step(X)

            # M-step: Update q(W)
            self._m_step(X, qz)

            # Check convergence
            elbo = self._compute_elbo(X, qz)
            if jnp.abs(elbo - old_elbo) < tol:
                break
            old_elbo = elbo

        return self

    def transform(self, X: Array) -> Array:
        """Transform data to latent space.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            Latent representations of shape (n_samples, n_components).
        """
        # Initialize latent distribution for new data
        n_samples = X.shape[0]
        self.z_ = MultivariateNormal(jnp.zeros((n_samples, self.n_components)), jnp.eye(self.n_components))

        # Update latent distribution
        self._update_z_distribution(X)

        return self.z_.mean

    def inverse_transform(self, Z: Array) -> Dict[str, Array]:
        """Transform latent representations back to feature space.

        Args:
            Z: Latent representations of shape (n_samples, n_components).

        Returns:
            Dictionary mapping feature type to reconstructed data.
        """
        reconstructed = {}

        # Reconstruct each feature type separately
        for likelihood_type, feature_indices in self.feature_types.items():
            # Get relevant loadings
            W = self.W_.mean[feature_indices]

            # Compute reconstructed values
            X_rec = jnp.dot(Z, W.T)

            # Apply appropriate transformation
            if likelihood_type == "normal":
                reconstructed[likelihood_type] = X_rec
            elif likelihood_type == "bernoulli":
                reconstructed[likelihood_type] = jnp.sigmoid(X_rec)
            elif likelihood_type == "categorical":
                reconstructed[likelihood_type] = jnn.softmax(X_rec)

        return reconstructed
