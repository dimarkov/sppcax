"""Tests for Factor Analysis models using the new BayesianFactorAnalysis API."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.models.factor_analysis import BayesianFactorAnalysis, BayesianPCA


def test_factor_analysis_array_input():
    """Test Factor Analysis with array input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    X = jr.normal(key, (n_samples, n_features))

    model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
    params, props = model.initialize(key)
    params, elbos = model.fit_em(params, props, X, key=key, num_iters=5, verbose=False)

    assert params.emissions.weights.shape == (n_features, n_components)
    assert jnp.isfinite(elbos[-1])

    qz = model.transform(params, X)
    assert qz.mean.shape == (n_samples, n_components)


def test_factor_analysis_vbem():
    """Test Factor Analysis with VBEM."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    X = jr.normal(key, (n_samples, n_features))

    model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
    params, props = model.initialize(key, variational_bayes=True)
    params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=10, verbose=False)

    assert params.emissions.weights.shape == (n_features, n_components)
    assert jnp.isfinite(elbos[-1])

    qz = model.transform(params, X)
    assert qz.mean.shape == (n_samples, n_components)


def test_ppca_array_input():
    """Test PPCA with array input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    X = jr.normal(key, (n_samples, n_features))

    model = BayesianPCA(n_components=n_components, n_features=n_features, key=key)
    params, props = model.initialize(key)
    params, elbos = model.fit_em(params, props, X, key=key, num_iters=5, verbose=False)

    assert params.emissions.weights.shape == (n_features, n_components)
    assert jnp.isfinite(elbos[-1])

    # Check isotropic noise
    R = params.emissions.cov
    R_diag = jnp.diag(R)
    assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)

    qz = model.transform(params, X)
    assert qz.mean.shape == (n_samples, n_components)


def test_factor_analysis_transform_inverse():
    """Test Factor Analysis transform and inverse_transform."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    X = jr.normal(key, (n_samples, n_features))

    model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
    params, props = model.initialize(key)
    params, _ = model.fit_em(params, props, X, key=key, num_iters=10, verbose=False)

    qz = model.transform(params, X)
    assert qz.mean.shape == (n_samples, n_components)

    X_recon = model.inverse_transform(params, qz)
    assert X_recon.mean.shape == (n_samples, n_features)
