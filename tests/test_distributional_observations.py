"""Tests for distributional (uncertain) observations in FA and DFA models.

When Y is a Distribution (not a point array), the EM algorithm should:
- Use <Y> as observations in the E-step (Kalman smoother)
- Adjust sum_yyT to include observation covariance: sum_t <y_t y_t^T>
- Correct ELBO by -1/2 sum_t tr(R^{-1} Sigma_t)
"""

import jax.numpy as jnp
import jax.random as jr

from sppcax.distributions.mvn import MultivariateNormal
from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.factor_analysis import BayesianFactorAnalysis


def generate_iid_data(key, n_samples=200, n_features=6, n_components=2):
    """Generate synthetic iid FA data."""
    k1, k2, k3 = jr.split(key, 3)
    W_true = jr.normal(k1, (n_features, n_components))
    Z_true = jr.normal(k2, (n_samples, n_components))
    noise = 0.3 * jr.normal(k3, (n_samples, n_features))
    return Z_true @ W_true.T + noise


def generate_timeseries_data(key, n_timesteps=100, n_features=6, n_components=2):
    """Generate synthetic DFA time-series data."""
    k1, k2, k3 = jr.split(key, 3)
    H_true = jr.normal(k1, (n_features, n_components))
    F_true = 0.9 * jnp.eye(n_components)

    z = jnp.zeros((n_timesteps, n_components))
    z = z.at[0].set(jr.normal(k2, (n_components,)))
    keys = jr.split(k3, n_timesteps - 1)
    for t in range(1, n_timesteps):
        z = z.at[t].set(F_true @ z[t - 1] + jr.normal(keys[t - 1], (n_components,)))

    k1, _ = jr.split(k1)
    noise = 0.5 * jr.normal(k1, (n_timesteps, n_features))
    return z @ H_true.T + noise


class TestSmallCovarianceRegression:
    """Distributional Y with tiny covariance should closely match point Y."""

    def test_fa_em_small_cov_close_to_point(self):
        """FA EM with MVN(Y, eps*I) should give similar ELBO as point Y."""
        key = jr.PRNGKey(42)
        X = generate_iid_data(key)
        n_samples, n_features = X.shape

        eps = 1e-4
        cov = eps * jnp.broadcast_to(jnp.eye(n_features), (n_samples, n_features, n_features))
        X_dist = MultivariateNormal(loc=X, covariance=cov)

        model = BayesianFactorAnalysis(n_components=2, n_features=n_features)
        params, props = model.initialize(key)

        _, elbos_point = model.fit_em(params, props, X, key=key, num_iters=5)
        _, elbos_dist = model.fit_em(params, props, X_dist, key=key, num_iters=5)

        # With tiny covariance, ELBOs should be very close (within 1%)
        ratio = elbos_dist[-1] / elbos_point[-1]
        assert (
            0.99 < ratio < 1.01
        ), f"Small-cov ELBO ratio: {ratio:.6f} (dist={elbos_dist[-1]:.2f}, point={elbos_point[-1]:.2f})"

    def test_dfa_em_small_cov_close_to_point(self):
        """DFA EM with MVN(Y, eps*I) should give similar ELBO as point Y."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key)
        T, D = Y.shape

        eps = 1e-4
        cov = eps * jnp.broadcast_to(jnp.eye(D), (T, D, D))
        Y_dist = MultivariateNormal(loc=Y, covariance=cov)

        model = BayesianDynamicFactorAnalysis(state_dim=2, emission_dim=D)
        params, props = model.initialize(key)

        _, elbos_point = model.fit_em(params, props, Y, key=key, num_iters=5)
        _, elbos_dist = model.fit_em(params, props, Y_dist, key=key, num_iters=5)

        ratio = elbos_dist[-1] / elbos_point[-1]
        assert (
            0.99 < ratio < 1.01
        ), f"Small-cov ELBO ratio: {ratio:.6f} (dist={elbos_dist[-1]:.2f}, point={elbos_point[-1]:.2f})"

    def test_dfa_vbem_small_cov_close_to_point(self):
        """DFA VBEM with MVN(Y, eps*I) should give similar ELBO as point Y."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key)
        T, D = Y.shape

        eps = 1e-4
        cov = eps * jnp.broadcast_to(jnp.eye(D), (T, D, D))
        Y_dist = MultivariateNormal(loc=Y, covariance=cov)

        model = BayesianDynamicFactorAnalysis(state_dim=2, emission_dim=D)
        params, props = model.initialize(key, variational_bayes=True)

        _, elbos_point = model.fit_vbem(params, props, Y, key=key, num_iters=5)
        _, elbos_dist = model.fit_vbem(params, props, Y_dist, key=key, num_iters=5)

        ratio = elbos_dist[-1] / elbos_point[-1]
        assert (
            0.99 < ratio < 1.01
        ), f"Small-cov ELBO ratio: {ratio:.6f} (dist={elbos_dist[-1]:.2f}, point={elbos_point[-1]:.2f})"


class TestDistributionalObservations:
    """Distributional Y with non-zero covariance should increase learned R."""

    def test_fa_em_distributional_inflates_R(self):
        """FA EM with uncertain Y should learn larger R than with point Y."""
        key = jr.PRNGKey(42)
        X = generate_iid_data(key)
        n_samples, n_features = X.shape

        obs_var = 0.5
        cov = obs_var * jnp.broadcast_to(jnp.eye(n_features), (n_samples, n_features, n_features))
        X_dist = MultivariateNormal(loc=X, covariance=cov)

        model = BayesianFactorAnalysis(n_components=2, n_features=n_features)
        params, props = model.initialize(key)

        params_point, _ = model.fit_em(params, props, X, key=key, num_iters=20)
        params_dist, _ = model.fit_em(params, props, X_dist, key=key, num_iters=20)

        R_point = params_point.emissions.cov
        R_dist = params_dist.emissions.cov

        # Compare diagonals (R may be stored as (D, D) diagonal matrix or (D,) vector)
        R_point_diag = jnp.diag(R_point) if R_point.ndim == 2 else R_point
        R_dist_diag = jnp.diag(R_dist) if R_dist.ndim == 2 else R_dist

        # R_dist should be larger than R_point (absorbs observation uncertainty)
        assert jnp.all(
            R_dist_diag > R_point_diag
        ), f"R_dist should be larger: R_dist={R_dist_diag.mean():.4f}, R_point={R_point_diag.mean():.4f}"

    def test_dfa_em_distributional_inflates_R(self):
        """DFA EM with uncertain Y should learn larger R than with point Y."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key)
        T, D = Y.shape

        obs_var = 0.5
        cov = obs_var * jnp.broadcast_to(jnp.eye(D), (T, D, D))
        Y_dist = MultivariateNormal(loc=Y, covariance=cov)

        model = BayesianDynamicFactorAnalysis(state_dim=2, emission_dim=D)
        params, props = model.initialize(key)

        params_point, _ = model.fit_em(params, props, Y, key=key, num_iters=20)
        params_dist, _ = model.fit_em(params, props, Y_dist, key=key, num_iters=20)

        R_point = params_point.emissions.cov
        R_dist = params_dist.emissions.cov

        # Compare diagonals (R may be stored as (D, D) diagonal matrix or (D,) vector)
        R_point_diag = jnp.diag(R_point) if R_point.ndim == 2 else R_point
        R_dist_diag = jnp.diag(R_dist) if R_dist.ndim == 2 else R_dist

        # Mean R_dist should be larger than mean R_point (absorbs observation uncertainty)
        assert (
            R_dist_diag.mean() > R_point_diag.mean()
        ), f"R_dist should be larger on average: R_dist={R_dist_diag.mean():.4f}, R_point={R_point_diag.mean():.4f}"

    def test_distributional_elbo_lower_than_point(self):
        """ELBO with uncertain Y should be lower than with point Y (uncertainty penalty)."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key)
        T, D = Y.shape

        obs_var = 0.5
        cov = obs_var * jnp.broadcast_to(jnp.eye(D), (T, D, D))
        Y_dist = MultivariateNormal(loc=Y, covariance=cov)

        model = BayesianDynamicFactorAnalysis(state_dim=2, emission_dim=D)
        params, props = model.initialize(key)

        _, elbos_point = model.fit_em(params, props, Y, key=key, num_iters=10)
        _, elbos_dist = model.fit_em(params, props, Y_dist, key=key, num_iters=10)

        # First iteration ELBO (same params) should be lower with uncertainty
        assert (
            elbos_dist[0] < elbos_point[0]
        ), f"Distributional ELBO should be lower: {elbos_dist[0]:.4f} vs {elbos_point[0]:.4f}"
