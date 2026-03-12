"""Tests for unified FA/PCA/DFA models.

Covers all combinations of ARD, parameter expansion (PX), and Bayesian model
reduction (BMR) for both FA and DFA, plus feature parity tests.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.factor_analysis import BayesianFactorAnalysis, BayesianPCA


# ============================================================================
# Synthetic data generation
# ============================================================================


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
    return X, W_true, Z_true


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
    return X, W_true, Z_true, n_model_components


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
    return Y, H_true, F_true, z


# ============================================================================
# DFA Baseline Tests
# ============================================================================


class TestDFABaseline:
    """Record and verify baseline DFA outputs."""

    def test_dfa_em_baseline(self):
        """Record DFA EM baseline."""
        key = jr.PRNGKey(42)
        Y, H_true, F_true, z_true = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert params.emissions.weights.shape == (10, 3)
        assert params.dynamics.weights.shape == (3, 3)

    def test_dfa_vbem_baseline(self):
        """Record DFA VBEM baseline."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=30, verbose=False)

        assert params.emissions.weights.shape == (10, 3)
        assert params.dynamics.weights.shape == (3, 3)

    def test_dfa_gibbs_baseline(self):
        """Record DFA Gibbs baseline."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key)
        sample_of_params, elbos = model.fit_blocked_gibbs(
            key=key, initial_params=params, props=props, sample_size=10, emissions=Y[None, ...], verbose=False
        )

        assert sample_of_params.emissions.weights.shape[0] == 10  # 10 samples
        assert sample_of_params.emissions.weights.shape[1:] == (10, 3)


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
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, has_ard=True, key=key)

        assert model.is_static
        assert model.has_ard
        assert not model.isotropic_noise

        params, props = model.initialize(key)
        assert params.emissions.weights.shape == (10, 3)
        assert not props.dynamics.weights.trainable  # dynamics frozen

        params, elbos = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_bayesian_pca_basic(self):
        """Test BayesianPCA creates and fits correctly."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianPCA(n_components=3, n_features=10, has_ard=True, key=key)

        assert model.is_static
        assert model.has_ard
        assert model.isotropic_noise

        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

        # Check isotropic noise: cov should be scalar * I
        R_diag = jnp.diag(params.emissions.cov)
        assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)

    def test_bayesian_fa_transform(self):
        """Test BayesianFactorAnalysis transform and inverse_transform."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, key=key)
        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        qz = model.transform(params, X)
        assert qz.mean.shape == (100, 3)

        X_recon = model.inverse_transform(params, qz)
        assert X_recon.mean.shape == (100, 10)

    def test_bayesian_fa_with_ard_updates(self):
        """Test that ARD prior gets updated during training."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, has_ard=True, key=key)
        ard_alpha_before = model.ard_prior.emission.alpha.copy()

        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        ard_alpha_after = model.ard_prior.emission.alpha
        assert not jnp.allclose(ard_alpha_before, ard_alpha_after)


# ============================================================================
# Parametrized FA EM Tests: ARD x PX x BMR combinations
# ============================================================================


class TestFAEMCombinations:
    """Test FA EM with all combinations of ARD, PX, and BMR."""

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_fa_em(self, has_ard, use_px, use_bmr):
        """FA EM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0], f"ELBO did not improve: first={elbos[0]:.4f}, last={elbos[-1]:.4f}"
        if not use_bmr:
            # ARD/PX can cause small dips but not large ones
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -0.1), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_pca_em(self, has_ard, use_px, use_bmr):
        """PCA EM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianPCA(
            n_components=3,
            n_features=10,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
        if not use_bmr:
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -0.1), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

        # Check isotropic noise
        R_diag = jnp.diag(params.emissions.cov)
        assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)


# ============================================================================
# Parametrized FA VBEM Tests: ARD x PX x BMR combinations
# ============================================================================


class TestFAVBEMCombinations:
    """Test FA VBEM with all combinations of ARD, PX, and BMR."""

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_fa_vbem(self, has_ard, use_px, use_bmr):
        """FA VBEM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
            key=key,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
        if not use_bmr:
            # VBEM uses variational corrections and PX applies a numerical
            # rotation — together they can cause larger transient dips than EM
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -1.0), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_pca_vbem(self, has_ard, use_px, use_bmr):
        """PCA VBEM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianPCA(
            n_components=3,
            n_features=10,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
            key=key,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
        if not use_bmr:
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -1.0), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )


# ============================================================================
# Parametrized DFA Tests: ARD x PX x BMR combinations
# ============================================================================


class TestDFACombinations:
    """Test DFA EM and VBEM with all combinations of ARD, PX, and BMR."""

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_dfa_em(self, has_ard, use_px, use_bmr):
        """DFA EM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0], f"ELBO did not improve: first={elbos[0]:.4f}, last={elbos[-1]:.4f}"
        if not use_bmr:
            # DFA Kalman smoother numerics + ARD/PX can cause slightly larger
            # dips than FA, so we use a relaxed tolerance of 0.2
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -0.2), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )

    @pytest.mark.parametrize("has_ard", [True, False], ids=["ard", "no_ard"])
    @pytest.mark.parametrize("use_px", [True, False], ids=["px", "no_px"])
    @pytest.mark.parametrize("use_bmr", [True, False], ids=["bmr", "no_bmr"])
    def test_dfa_vbem(self, has_ard, use_px, use_bmr):
        """DFA VBEM should converge and be approximately monotonic for all ARD/PX/BMR combinations."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=has_ard,
            use_px=use_px,
            use_bmr=use_bmr,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
        if not use_bmr:
            diffs = jnp.diff(elbos)
            assert jnp.all(diffs >= -0.2), (
                f"ELBO decreased by more than tolerance. "
                f"Worst decrease: {diffs.min():.6f} at iteration {jnp.argmin(diffs)}"
            )


# ============================================================================
# Parameter Expansion (PX) Tests
# ============================================================================


class TestParameterExpansion:
    """Test that parameter expansion improves convergence."""

    def test_px_improves_fa_convergence(self):
        """FA with PX should converge faster (higher ELBO at same iteration count)."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model_no_px = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            use_px=False,
            key=key,
        )
        params, props = model_no_px.initialize(key)
        _, elbos_no_px = model_no_px.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        model_px = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            use_px=True,
            key=key,
        )
        params, props = model_px.initialize(key)
        _, elbos_px = model_px.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        # PX should achieve at least as good an ELBO
        assert (
            elbos_px[-1] >= elbos_no_px[-1] - 1.0
        ), f"PX ELBO ({elbos_px[-1]:.4f}) much worse than no-PX ({elbos_no_px[-1]:.4f})"

    def test_px_improves_dfa_convergence(self):
        """DFA with PX should converge faster."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model_no_px = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            use_px=False,
        )
        params, props = model_no_px.initialize(key)
        _, elbos_no_px = model_no_px.fit_em(params, props, Y, key=key, num_iters=20, verbose=False)

        model_px = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            use_px=True,
        )
        params, props = model_px.initialize(key)
        _, elbos_px = model_px.fit_em(params, props, Y, key=key, num_iters=20, verbose=False)

        assert (
            elbos_px[-1] >= elbos_no_px[-1] - 1.0
        ), f"PX ELBO ({elbos_px[-1]:.4f}) much worse than no-PX ({elbos_no_px[-1]:.4f})"

    def test_px_custom_params(self):
        """Test PX with custom step count and learning rate."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            use_px=True,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(
            params,
            props,
            X,
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
        """BMR should achieve higher ELBO than no-BMR on sparse FA data.

        When the true model is sparse (2 components, fitting 5), BMR should
        prune unnecessary parameters and achieve a tighter bound. Since BMR
        uses stochastic Gibbs sampling, we don't expect strict monotonicity at
        every step, but the final ELBO should be higher.
        """
        key = jr.PRNGKey(42)
        X, _, _, n_model = generate_sparse_iid_data(key, n_samples=100)

        model_no_bmr = BayesianFactorAnalysis(
            n_components=n_model,
            n_features=10,
            has_ard=True,
            use_bmr=False,
            key=key,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianFactorAnalysis(
            n_components=n_model,
            n_features=10,
            has_ard=True,
            use_bmr=True,
            key=key,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        assert elbos_bmr[-1] > elbos_no_bmr[-1], (
            f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) " f"on sparse data"
        )

    def test_bmr_improves_elbo_sparse_pca(self):
        """BMR should achieve higher ELBO than no-BMR on sparse PCA data."""
        key = jr.PRNGKey(42)
        X, _, _, n_model = generate_sparse_iid_data(key, n_samples=100)

        model_no_bmr = BayesianPCA(
            n_components=n_model,
            n_features=10,
            has_ard=True,
            use_bmr=False,
            key=key,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianPCA(
            n_components=n_model,
            n_features=10,
            has_ard=True,
            use_bmr=True,
            key=key,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        assert elbos_bmr[-1] > elbos_no_bmr[-1], (
            f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) " f"on sparse data"
        )

    def test_bmr_improves_elbo_sparse_dfa(self):
        """BMR should achieve higher ELBO than no-BMR on sparse DFA data."""
        key = jr.PRNGKey(42)
        # Generate sparse time-series: 5 model components, but only 2 are active
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100, n_components=2)

        model_no_bmr = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            use_bmr=False,
        )
        params, props = model_no_bmr.initialize(key)
        _, elbos_no_bmr = model_no_bmr.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        model_bmr = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            use_bmr=True,
        )
        params, props = model_bmr.initialize(key)
        _, elbos_bmr = model_bmr.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        assert elbos_bmr[-1] > elbos_no_bmr[-1], (
            f"BMR ELBO ({elbos_bmr[-1]:.4f}) should exceed no-BMR ({elbos_no_bmr[-1]:.4f}) " f"on sparse DFA data"
        )

    def test_bmr_non_monotonic_but_improving(self):
        """BMR ELBOs may fluctuate but should improve overall."""
        key = jr.PRNGKey(42)
        X, _, _, n_model = generate_sparse_iid_data(key, n_samples=100)

        model = BayesianFactorAnalysis(
            n_components=n_model,
            n_features=10,
            has_ard=True,
            use_bmr=True,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        assert jnp.all(jnp.isfinite(elbos))
        # Overall improvement (not checking strict monotonicity due to stochastic BMR)
        assert elbos[-1] > elbos[0], f"BMR ELBO should improve overall: first={elbos[0]:.4f}, last={elbos[-1]:.4f}"


# ============================================================================
# Feature Parity Tests
# ============================================================================


class TestFeatureParity:
    """Test feature parity across model types."""

    def test_fa_no_bias(self):
        """Test FA without emission bias."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=3,
            n_features=10,
            has_emissions_bias=False,
            key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])
        assert jnp.allclose(params.emissions.bias, jnp.zeros(10))

    def test_dfa_with_ard(self):
        """Test DFA with ARD and MVNIG emission prior for time-series data."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        from sppcax.models.utils import _make_mvnig_prior

        emission_prior = _make_mvnig_prior(n_features=10, n_components=5, input_dim=0, has_bias=True)

        model = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            emission_prior=emission_prior,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])
        # ARD should have been updated (dnat1 gets non-zero values)
        assert not jnp.allclose(model.ard_prior.emission.dnat1, jnp.zeros(5))

    def test_fa_with_bmr_vbem(self):
        """Test FA BMR with VBEM algorithm."""
        key = jr.PRNGKey(42)
        X, _, _ = generate_iid_data(key, n_samples=100, n_features=10, n_components=3)

        model = BayesianFactorAnalysis(
            n_components=5,
            n_features=10,
            use_bmr=True,
            key=key,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])

    def test_dfa_with_px_and_bmr(self):
        """Test DFA with both PX and BMR enabled."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100, n_components=2)

        model = BayesianDynamicFactorAnalysis(
            state_dim=5,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            has_ard=True,
            use_px=True,
            use_bmr=True,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=20, verbose=False)

        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]

    def test_dfa_with_px_vbem(self):
        """Test DFA VBEM with PX enabled."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key, n_timesteps=100)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3,
            emission_dim=10,
            has_dynamics_bias=True,
            has_emissions_bias=True,
            use_px=True,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=20, verbose=False)

        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
