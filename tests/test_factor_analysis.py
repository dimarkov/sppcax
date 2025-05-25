"""Tests for Factor Analysis models."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.distributions.delta import Delta
from sppcax.distributions.mvn import MultivariateNormal
from sppcax.models.factor_analysis_params import PFA, PPCA  # Updated import
from sppcax.models.factor_analysis_algorithms import fit, transform  # Added import


def test_factor_analysis_array_input():  # Renamed test to reflect PFA
    """Test Factor Analysis with array input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data
    X = jr.normal(key, (n_samples, n_features))

    # Fit model
    model = PFA(n_components=n_components, n_features=n_features)  # Changed class name
    model, lls = fit(model, X, n_iter=5, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == (n_features,)  # Noise precision per feature
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X)  # Changed to function call
    assert qz.mean.shape == (n_samples, n_components)


def test_factor_analysis_delta_input():  # Renamed test to reflect PFA
    """Test Factor Analysis with Delta distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create Delta distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = Delta(X)

    # Fit model
    model = PFA(
        n_components=n_components, n_features=n_features, key=jr.PRNGKey(0)
    )  # Changed class name and random_state->key
    model, elbos = fit(model, X_dist, n_iter=10, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == (n_features,)  # Noise precision per feature
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X_dist)  # Changed to function call
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
    model, lls = fit(model, X, n_iter=5, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == ()  # Scalar noise precision for PPCA
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X)  # Changed to function call
    assert qz.mean.shape == (n_samples, n_components)


def test_ppca_delta_input():
    """Test PPCA with Delta distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create Delta distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = Delta(X)

    # Fit model
    model = PPCA(n_components=n_components, n_features=n_features, key=jr.PRNGKey(0))  # Changed random_state->key
    model, elbos = fit(model, X_dist, n_iter=10, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == ()  # Scalar noise precision for PPCA
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X_dist)  # Changed to function call
    assert qz.mean.shape == (n_samples, n_components)

    # Check ELBO increases monotonically
    assert len(elbos) <= 10  # Should stop early if converged
    # TODO: fix elbos convergence issues
    # assert jnp.all(jnp.diff(jnp.asarray(elbos)) >= -1e-6)  # ELBO should increase (allowing for numerical error)


def test_factor_analysis_mvn_input():  # Renamed test to reflect PFA
    """Test Factor Analysis with MVN distribution input."""
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 10, 3

    # Generate random data and create MVN distribution
    X = jr.normal(key, (n_samples, n_features))
    X_dist = MultivariateNormal(loc=X, scale_tril=0.1 * jnp.eye(n_features)[None, ...].repeat(n_samples, axis=0))

    # Fit model
    model = PFA(n_components=n_components, n_features=n_features)  # Changed class name
    model, _ = fit(model, X_dist, n_iter=5, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == (n_features,)  # Noise precision per feature
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X_dist)  # Changed to function call
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
    model, _ = fit(model, X_dist, n_iter=5, key=key)  # Changed to function call

    # Check shapes
    assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
    assert model.q_w_psi.inv_gamma.mean.shape == ()  # Scalar noise precision for PPCA
    assert model.q_tau.mean.shape == (n_components,)  # ARD prior precision per component

    # Transform data
    qz = transform(model, X_dist)  # Changed to function call
    assert qz.mean.shape == (n_samples, n_components)
