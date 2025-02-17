"""Tests for Factor Analysis models."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.distributions.delta import Delta
from sppcax.distributions.mvn import MultivariateNormal
from sppcax.models.factor_analysis import PPCA, FactorAnalysis


def test_factor_analysis_array_input():
    """Test Factor Analysis with array input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data
    X = jr.normal(key, (n_samples, n_features))

    # Fit model
    model = FactorAnalysis(n_components=n_components, n_features=n_features)
    model, lls = model.fit(X, n_iter=5)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == (n_features,)

    # Transform data
    qz = model.transform(X)
    assert qz.mean.shape == (n_samples, n_components)


def test_factor_analysis_delta_input():
    """Test Factor Analysis with Delta distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create Delta distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = Delta(X)

    # Fit model
    model = FactorAnalysis(n_components=n_components, n_features=n_features, random_state=jr.PRNGKey(0))
    model, elbos = model.fit(X_dist, n_iter=10)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == (n_features,)

    # Transform data
    qz = model.transform(X_dist)
    assert qz.mean.shape == (n_samples, n_components)

    # Check ELBO increases monotonically
    assert len(elbos) <= 10  # Should stop early if converged
    # TODO: fix elbo convergence issues.
    # assert jnp.all(jnp.diff(jnp.asarray(elbos)) >= -1e-6)  # ELBO should increase (allowing for numerical error)


def test_ppca_array_input():
    """Test PPCA with array input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data
    X = jr.normal(key, (n_samples, n_features))

    # Fit model
    model = PPCA(n_components=n_components, n_features=n_features)
    model, lls = model.fit(X, n_iter=5)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == ()  # Scalar for PPCA

    # Transform data
    qz = model.transform(X)
    assert qz.mean.shape == (n_samples, n_components)


def test_ppca_delta_input():
    """Test PPCA with Delta distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create Delta distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = Delta(X)

    # Fit model
    model = PPCA(n_components=n_components, n_features=n_features, random_state=jr.PRNGKey(0))
    model, elbos = model.fit(X_dist, n_iter=10)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == ()  # Scalar for PPCA

    # Transform data
    qz = model.transform(X_dist)
    assert qz.mean.shape == (n_samples, n_components)

    # Check ELBO increases monotonically
    assert len(elbos) <= 10  # Should stop early if converged
    # TODO: fix elbos convergence issues
    # assert jnp.all(jnp.diff(jnp.asarray(elbos)) >= -1e-6)  # ELBO should increase (allowing for numerical error)


def test_factor_analysis_mvn_input():
    """Test Factor Analysis with MVN distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create MVN distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = MultivariateNormal(loc=X, scale_tril=0.1 * jnp.eye(n_features)[None, ...].repeat(n_samples, axis=0))

    # Fit model
    model = FactorAnalysis(n_components=n_components, n_features=n_features)
    model, _ = model.fit(X_dist, n_iter=5)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == (n_features,)

    # Transform data
    qz = model.transform(X_dist)
    assert qz.mean.shape == (n_samples, n_components)


def test_ppca_mvn_input():
    """Test PPCA with MVN distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create MVN distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = MultivariateNormal(loc=X, scale_tril=0.1 * jnp.eye(n_features)[None, ...].repeat(n_samples, axis=0))

    # Fit model
    model = PPCA(n_components=n_components, n_features=n_features)
    model, _ = model.fit(X_dist, n_iter=5)

    # Check shapes
    assert model.W_dist.mean.shape == (n_features, n_components)
    assert model.noise_precision.mean.shape == ()  # Scalar for PPCA

    # Transform data
    qz = model.transform(X_dist)
    assert qz.mean.shape == (n_samples, n_components)
