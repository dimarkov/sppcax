"""Tests for Dynamic Factor Analysis implementation."""

import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx  # Added import

from sppcax.models.dynamic_factor_analysis_params import DynamicFactorAnalysisParams

# Import necessary algorithm functions
from sppcax.models.factor_analysis_algorithms import fit, kalman_smoother_estep, dfa_mstep


@pytest.fixture
def dfa_setup():
    key = jr.PRNGKey(0)
    n_components = 3
    n_features = 10
    n_samples = 50
    model = DynamicFactorAnalysisParams(n_components=n_components, n_features=n_features, key=key)
    # Generate synthetic data (simple random walk for factors, linear observation)
    keys = jr.split(key, 5)
    A_true = jnp.eye(n_components) + jr.normal(keys[0], (n_components, n_components)) * 0.1
    Q_true_diag = jr.uniform(keys[1], (n_components,), minval=0.1, maxval=0.5)
    Q_true = jnp.diag(Q_true_diag)
    C_true = jr.normal(keys[2], (n_features, n_components))
    R_true_diag = jr.uniform(keys[3], (n_features,), minval=0.5, maxval=1.0)
    R_true = jnp.diag(R_true_diag)

    z = jnp.zeros((n_samples, n_components))
    y = jnp.zeros((n_samples, n_features))
    z_curr = jr.multivariate_normal(keys[4], jnp.zeros(n_components), jnp.eye(n_components))
    z = z.at[0].set(z_curr)
    y = y.at[0].set(jr.multivariate_normal(jr.PRNGKey(10), C_true @ z_curr, R_true))

    for t in range(1, n_samples):
        key_t = jr.PRNGKey(t)
        z_curr = jr.multivariate_normal(key_t, A_true @ z_curr, Q_true)
        z = z.at[t].set(z_curr)
        y = y.at[t].set(jr.multivariate_normal(jr.PRNGKey(t + 100), C_true @ z_curr, R_true))

    return model, y, key


def test_dfa_initialization(dfa_setup):
    """Test if DynamicFactorAnalysisParams initializes correctly."""
    model, _, _ = dfa_setup
    assert isinstance(model, DynamicFactorAnalysisParams)
    assert model.n_components == 3
    assert model.n_features == 10
    assert model.q_A.shape == (3, 3)  # k independent MVNs
    assert model.q_A.event_shape == (3,)  # Each MVN is k-dimensional
    assert model.q_Q.shape == (3,)
    assert model.q_c_r.mvn.shape == (10, 3)  # d independent MVNs
    assert model.q_c_r.mvn.event_shape == (3,)  # Each MVN is k-dimensional
    assert model.q_c_r.gamma.shape == (10,) or model.q_c_r.gamma.shape == ()  # Depending on isotropic
    assert model.q_initial_state.shape == (3,)  # Single MVN
    assert model.q_initial_state.event_shape == (3,)
    assert model.mean_.shape == (10,)


def test_dfa_fit(dfa_setup):
    """Test if the fit function runs for DFA."""
    model, Y, key = dfa_setup
    n_iter = 5
    tol = 1e-3

    fitted_model, log_likelihoods = fit(model, Y, key, n_iter=n_iter, tol=tol)

    assert isinstance(fitted_model, DynamicFactorAnalysisParams)
    assert len(log_likelihoods) <= n_iter
    # Check if log likelihood generally increases (allowing for small dips)
    # assert log_likelihoods[-1] > log_likelihoods[0] - 10.0 # Allow some initial fluctuation

    # Check parameter shapes remain consistent
    assert fitted_model.q_A.shape == model.q_A.shape
    assert fitted_model.q_Q.shape == model.q_Q.shape
    assert fitted_model.q_c_r.mvn.shape == model.q_c_r.mvn.shape
    assert fitted_model.q_initial_state.shape == model.q_initial_state.shape
    assert fitted_model.mean_.shape == model.mean_.shape


def test_dfa_kalman_smoother_estep(dfa_setup):
    """Test the Kalman smoother E-step function."""
    model, Y, _ = dfa_setup
    T, k = Y.shape[0], model.n_components

    Ez, Ezz, Ezz_lag, log_lik = kalman_smoother_estep(model, Y)

    assert Ez.shape == (T, k)
    assert Ezz.shape == (T, k, k)
    assert Ezz_lag.shape == (T - 1, k, k)
    assert jnp.isscalar(log_lik)
    assert jnp.isfinite(log_lik)


def test_dfa_mstep(dfa_setup):
    """Test the M-step function for DFA."""
    model, Y, _ = dfa_setup

    # Run E-step first to get necessary inputs for M-step
    Ez, Ezz, Ezz_lag, _ = kalman_smoother_estep(model, Y)

    # Run M-step
    updated_model = dfa_mstep(model, Y, Ez, Ezz, Ezz_lag)

    assert isinstance(updated_model, DynamicFactorAnalysisParams)
    # Check if parameters were updated (e.g., mean of A)
    # Note: Exact equality check might fail due to numerical precision
    assert not jnp.allclose(model.expected_A, updated_model.expected_A, atol=1e-7)
    assert updated_model.mean_.shape == model.mean_.shape


def test_dfa_em_step_jit(dfa_setup):
    """Test the JIT-compiled EM step inside the fit function."""
    model, Y, key = dfa_setup
    T, d = Y.shape

    # Initial centering
    current_mean = jnp.mean(Y, axis=0)
    initial_model = eqx.tree_at(lambda m: m.mean_, model, current_mean)
    old_log_lik = -jnp.inf

    # Define the inner em_step function (copied from fit for testing)
    @eqx.filter_jit
    def em_step(carry):
        model_k, key_k, old_log_lik_k = carry
        # E-step
        Ez, Ezz, Ezz_lag, log_lik = kalman_smoother_estep(model_k, Y)
        # M-step
        model_kp1 = dfa_mstep(model_k, Y, Ez, Ezz, Ezz_lag)
        return (model_kp1, key_k, log_lik), log_lik

    # Run one JITted step
    carry = (initial_model, key, old_log_lik)
    (final_model, _, final_log_lik), _ = em_step(carry)

    assert isinstance(final_model, DynamicFactorAnalysisParams)
    assert jnp.isscalar(final_log_lik)
    assert jnp.isfinite(final_log_lik)
    # Check parameters changed
    assert not jnp.allclose(initial_model.expected_A, final_model.expected_A, atol=1e-7)
