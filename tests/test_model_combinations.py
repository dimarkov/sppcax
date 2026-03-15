"""Tests for FA/PCA/DFA model fitting across algorithm and feature combinations.

Covers EM, VBEM, and blocked Gibbs fitting for BayesianFactorAnalysis,
BayesianPCA, and BayesianDynamicFactorAnalysis with BMR on/off.
All tests exercise the input path by passing a zero input vector.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.factor_analysis import BayesianFactorAnalysis, BayesianPCA


# ============================================================================
# Synthetic data generation
# ============================================================================

INPUT_DIM = 2


def generate_iid_data(key, n_samples=100, n_features=10, n_components=3):
    """Generate synthetic iid FA data with known ground truth."""
    k1, k2, k3 = jr.split(key, 3)
    W_true = jnp.zeros((n_features, n_components))
    W_true = W_true.at[:4, 0].set(jr.normal(k1, (4,)))
    W_true = W_true.at[3:7, 1].set(jr.normal(k2, (4,)))
    W_true = W_true.at[6:10, 2].set(jr.normal(k3, (4,)))

    k1, k2 = jr.split(k1)
    Z_true = jr.normal(k1, (n_samples, n_components))
    noise = 0.3 * jr.normal(k2, (n_samples, n_features))
    X = Z_true @ W_true.T + noise
    # FA reshapes (N, D) to (N, 1, D); input shape is (N, 1, input_dim)
    U = jnp.zeros((n_samples, 1, INPUT_DIM))
    return X, U, W_true, Z_true


def generate_sparse_iid_data(key, n_samples=100, n_features=10, n_true_components=2, n_model_components=5):
    """Generate sparse iid data where only a few components are active.

    Useful for testing BMR and ARD, which should prune excess components.
    """
    k1, k2, k3 = jr.split(key, 3)
    W_true = jnp.zeros((n_features, n_true_components))
    W_true = W_true.at[:5, 0].set(jr.normal(k1, (5,)) * 2.0)
    W_true = W_true.at[5:10, 1].set(jr.normal(k2, (5,)) * 2.0)

    k1, k2 = jr.split(k1)
    Z_true = jr.normal(k1, (n_samples, n_true_components))
    noise = 0.3 * jr.normal(k2, (n_samples, n_features))
    X = Z_true @ W_true.T + noise
    U = jnp.zeros((n_samples, 1, INPUT_DIM))
    return X, U, W_true, Z_true, n_model_components


def generate_timeseries_data(key, n_timesteps=100, n_features=10, n_components=3):
    """Generate synthetic time-series DFA data with known ground truth."""
    k1, k2, k3 = jr.split(key, 3)
    H_true = jnp.zeros((n_features, n_components))
    H_true = H_true.at[:4, 0].set(jr.normal(k1, (4,)))
    H_true = H_true.at[3:7, 1].set(jr.normal(k2, (4,)))
    H_true = H_true.at[6:10, 2].set(jr.normal(k3, (4,)))

    F_true = 0.95 * jnp.eye(n_components)

    k1, k2 = jr.split(k1)
    z = jnp.zeros((n_timesteps, n_components))
    z = z.at[0].set(jr.normal(k1, (n_components,)))
    keys = jr.split(k2, n_timesteps - 1)
    for t in range(1, n_timesteps):
        z = z.at[t].set(F_true @ z[t - 1] + 0.1 * jr.normal(keys[t - 1], (n_components,)))

    k1, _ = jr.split(k1)
    noise = 0.3 * jr.normal(k1, (n_timesteps, n_features))
    Y = z @ H_true.T + noise
    U = jnp.zeros((n_timesteps, INPUT_DIM))
    return Y, U, H_true, F_true, z


# ============================================================================
# FA/PCA Subclass Tests
# ============================================================================


class TestFASubclasses:
    """Test BayesianFactorAnalysis and BayesianPCA subclasses."""

    def test_fa_hierarchy(self):
        """Test that BayesianPCA is a subclass of BayesianFactorAnalysis is a subclass of DFA."""
        model_pca = BayesianPCA(n_components=3, n_features=10)
        model_fa = BayesianFactorAnalysis(n_components=3, n_features=10)
        model_dfa = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10)

        assert isinstance(model_pca, BayesianFactorAnalysis)
        assert isinstance(model_pca, BayesianDynamicFactorAnalysis)
        assert isinstance(model_fa, BayesianDynamicFactorAnalysis)
        assert not isinstance(model_dfa, BayesianFactorAnalysis)

        assert model_pca.is_static
        assert model_fa.is_static
        assert not model_dfa.is_static

    def test_bayesian_fa_basic(self):
        """Test BayesianFactorAnalysis creates and fits correctly."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, input_dim=INPUT_DIM, has_ard=True, key=key)

        assert model.is_static
        assert model.has_ard
        assert not model.isotropic_noise

        params, props = model.initialize(key)
        assert params.emissions.weights.shape == (10, 3)
        assert not props.dynamics.weights.trainable  # dynamics frozen

        params, elbos = model.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_bayesian_pca_basic(self):
        """Test BayesianPCA creates and fits correctly."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianPCA(n_components=3, n_features=10, input_dim=INPUT_DIM, has_ard=True, key=key)

        assert model.is_static
        assert model.has_ard
        assert model.isotropic_noise

        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

        # Check isotropic noise: cov should be scalar * I
        R_diag = jnp.diag(params.emissions.cov)
        assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)

    def test_bayesian_fa_transform(self):
        """Test BayesianFactorAnalysis transform and inverse_transform."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, input_dim=INPUT_DIM, key=key)
        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        qz = model.transform(params, X)
        assert qz.mean.shape == (100, 3)

        X_recon = model.inverse_transform(params, qz)
        assert X_recon.mean.shape == (100, 10)

    def test_bayesian_fa_with_ard_updates(self):
        """Test that ARD prior gets updated during training."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, input_dim=INPUT_DIM, has_ard=True, key=key)
        ard_alpha_before = model.ard_prior.emission.alpha.copy()

        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        ard_alpha_after = model.ard_prior.emission.alpha
        assert not jnp.allclose(ard_alpha_before, ard_alpha_after)


# ============================================================================
# FA/PCA Fitting Tests (EM + VBEM, with/without BMR)
# ============================================================================


class TestFAFitting:
    """Test FA and PCA fitting with EM and VBEM, parametrized by BMR."""

    @pytest.mark.parametrize("method", ["em", "vbem"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_fa_fit(self, method, use_bmr):
        """FA should converge with finite, improving ELBO."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_px=True,
            use_bmr=use_bmr,
            key=key,
        )

        if method == "em":
            params, props = model.initialize(key)
            params, elbos = model.fit_em(params, props, X, U=U, key=key, num_iters=20, verbose=False)
        else:
            params, props = model.initialize(key, variational_bayes=True)
            params, elbos = model.fit_vbem(params, props, X, U=U, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0], f"ELBO did not improve: first={elbos[0]:.4f}, last={elbos[-1]:.4f}"
        assert params.emissions.weights.shape == (10, 3)

        if not use_bmr:
            diffs = jnp.diff(elbos)
            tol = -2.0 if method == "em" else -6.0
            assert jnp.all(diffs >= tol), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

    @pytest.mark.parametrize("method", ["em", "vbem"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_pca_fit(self, method, use_bmr):
        """PCA should converge with finite, improving ELBO and isotropic noise."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianPCA(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_px=True,
            use_bmr=use_bmr,
            key=key,
        )

        if method == "em":
            params, props = model.initialize(key)
            params, elbos = model.fit_em(params, props, X, U=U, key=key, num_iters=20, verbose=False)
        else:
            params, props = model.initialize(key, variational_bayes=True)
            params, elbos = model.fit_vbem(params, props, X, U=U, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]

        if not use_bmr:
            diffs = jnp.diff(elbos)
            tol = -2.0 if method == "em" else -6.0
            assert jnp.all(diffs >= tol), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

        # Check isotropic noise
        R_diag = jnp.diag(params.emissions.cov)
        assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)

    def test_fa_no_bias(self):
        """Test FA without emission bias."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            has_emissions_bias=False,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, U=U, key=key, num_iters=20, verbose=False)

        assert jnp.isfinite(elbos[-1])
        assert jnp.allclose(params.emissions.bias, jnp.zeros(10))


# ============================================================================
# DFA Fitting Tests (EM + VBEM + Gibbs, with/without BMR)
# ============================================================================


class TestDFAFitting:
    """Test DFA fitting with EM, VBEM, and blocked Gibbs, parametrized by BMR."""

    @pytest.mark.parametrize("method", ["em", "vbem"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_dfa_fit(self, method, use_bmr):
        """DFA should converge with finite, improving ELBO."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            use_px=True,
            use_bmr=use_bmr,
        )

        if method == "em":
            params, props = model.initialize(key)
            params, elbos = model.fit_em(params, props, Y, U=U, key=key, num_iters=20, verbose=False)
        else:
            params, props = model.initialize(key, variational_bayes=True)
            params, elbos = model.fit_vbem(params, props, Y, U=U, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0], f"ELBO did not improve: first={elbos[0]:.4f}, last={elbos[-1]:.4f}"
        assert params.emissions.weights.shape == (10, 3)
        assert params.dynamics.weights.shape == (3, 3)

        if not use_bmr:
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -0.2), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

    def test_dfa_gibbs(self):
        """DFA blocked Gibbs should produce valid samples with finite ELBO."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=True,
            has_emissions_bias=True,
        )
        params, props = model.initialize(key)
        sample_of_params, elbos = model.fit_blocked_gibbs(
            key=key,
            initial_params=params,
            props=props,
            sample_size=10,
            emissions=Y[None, ...],
            inputs=U[None, ...],
            verbose=False,
        )

        assert sample_of_params.emissions.weights.shape[0] == 10  # 10 samples
        assert sample_of_params.emissions.weights.shape[1:] == (10, 3)
        assert jnp.all(jnp.isfinite(elbos))

    def test_dfa_no_bias(self):
        """Test DFA without dynamics or emissions bias."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=False,
            has_emissions_bias=False,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, U=U, key=key, num_iters=20, verbose=False)

        assert jnp.isfinite(elbos[-1])
        assert jnp.allclose(params.emissions.bias, jnp.zeros(10))
        assert jnp.allclose(params.dynamics.bias, jnp.zeros(3))

    def test_dfa_ard_updates(self):
        """Test that ARD prior gets updated during DFA training."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        from sppcax.models.utils import _make_mvnig_prior

        emission_prior = _make_mvnig_prior(n_features=10, n_components=5, input_dim=INPUT_DIM, has_bias=True)

        model = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            emission_prior=emission_prior,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, U=U, key=key, num_iters=30, verbose=False)

        assert jnp.isfinite(elbos[-1])
        assert not jnp.allclose(model.ard_prior.emission.dnat1, jnp.zeros(5))


# ============================================================================
# Parameter Expansion (PX) Tests
# ============================================================================


class TestParameterExpansion:
    """Test that parameter expansion improves convergence."""

    def test_px_improves_fa_convergence(self):
        """FA with PX should converge faster (higher ELBO at same iteration count)."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model_no_px = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            use_px=False,
            key=key,
        )
        params, props = model_no_px.initialize(key)
        _, elbos_no_px = model_no_px.fit_em(params, props, X, U=U, key=key, num_iters=20, verbose=False)

        model_px = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            use_px=True,
            key=key,
        )
        params, props = model_px.initialize(key)
        _, elbos_px = model_px.fit_em(params, props, X, U=U, key=key, num_iters=20, verbose=False)

        assert (
            elbos_px[-1] >= elbos_no_px[-1]
        ), f"PX ELBO ({elbos_px[-1]:.4f}) much worse than no-PX ({elbos_no_px[-1]:.4f})"

    def test_px_improves_dfa_convergence(self):
        """DFA with PX should converge faster."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model_no_px = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            use_px=False,
        )
        params, props = model_no_px.initialize(key)
        _, elbos_no_px = model_no_px.fit_em(params, props, Y, U=U, key=key, num_iters=20, verbose=False)

        model_px = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            input_dim=INPUT_DIM,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            use_px=True,
        )
        params, props = model_px.initialize(key)
        _, elbos_px = model_px.fit_em(params, props, Y, U=U, key=key, num_iters=20, verbose=False)

        assert (
            elbos_px[-1] >= elbos_no_px[-1]
        ), f"PX ELBO ({elbos_px[-1]:.4f}) much worse than no-PX ({elbos_no_px[-1]:.4f})"

    def test_px_custom_params(self):
        """Test PX with custom step count and learning rate."""
        key = jr.PRNGKey(42)
        X, U, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            input_dim=INPUT_DIM,
            use_px=True,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(
            params,
            props,
            X,
            U=U,
            key=key,
            num_iters=15,
            verbose=False,
            px_n_steps=64,
            px_lr=5e-4,
        )

        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]


# ============================================================================
# BMR Comparison Tests
# ============================================================================


class TestBMRComparison:
    """Test that BMR achieves better ELBO on sparse data."""

    def test_bmr_improves_elbo_sparse_fa(self):
        """BMR should achieve higher ELBO than no-BMR on sparse FA data."""
        key = jr.PRNGKey(42)
        X, U, _, _, n_model = generate_sparse_iid_data(key, n_samples=100)

        model_no_bmr = BayesianFactorAnalysis(
            n_components=n_model,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_bmr=False,
            key=key,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianFactorAnalysis(
            n_components=n_model,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_bmr=True,
            key=key,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        assert (
            elbos_bmr[-1] > elbos_no_bmr[-1]
        ), f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) on sparse data"

    def test_bmr_improves_elbo_sparse_pca(self):
        """BMR should achieve higher ELBO than no-BMR on sparse PCA data."""
        key = jr.PRNGKey(42)
        X, U, _, _, n_model = generate_sparse_iid_data(key, n_samples=100)

        model_no_bmr = BayesianPCA(
            n_components=n_model,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_bmr=False,
            key=key,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianPCA(
            n_components=n_model,
            n_features=10,
            input_dim=INPUT_DIM,
            has_ard=True,
            use_bmr=True,
            key=key,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, X, U=U, key=key, num_iters=30, verbose=False)

        assert (
            elbos_bmr[-1] > elbos_no_bmr[-1]
        ), f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) on sparse data"

    def test_bmr_improves_elbo_sparse_dfa(self):
        """BMR should achieve higher ELBO than no-BMR on sparse DFA data."""
        key = jr.PRNGKey(42)
        Y, U, _, _, _ = generate_timeseries_data(key, n_timesteps=100, n_components=2)

        model_no_bmr = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            input_dim=INPUT_DIM,
            use_bmr=False,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, Y, U=U, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            input_dim=INPUT_DIM,
            use_bmr=True,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, Y, U=U, key=key, num_iters=30, verbose=False)

        assert (
            elbos_bmr[-1] > elbos_no_bmr[-1]
        ), f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) on sparse DFA data"
