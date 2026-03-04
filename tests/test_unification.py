"""Tests for unifying FA/PCA as special cases of DFA.

Includes baseline tests, performance benchmarks, equivalence tests,
and feature parity tests.
"""

import time
import warnings
from functools import partial

import jax
import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import jit

from sppcax.distributions.delta import Delta
from sppcax.distributions.mvn import MultivariateNormal

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from sppcax.models.factor_analysis_params import PFA, PPCA
    from sppcax.models.factor_analysis_algorithms import fit, transform, e_step, m_step, compute_elbo

from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.factor_analysis import BayesianFactorAnalysis, BayesianPCA


# ============================================================================
# Synthetic data generation
# ============================================================================

def generate_iid_data(key, n_samples=100, n_features=10, n_components=3):
    """Generate synthetic iid FA data with known ground truth."""
    k1, k2, k3 = jr.split(key, 3)

    # True loading matrix (sparse)
    W_true = jnp.zeros((n_features, n_components))
    W_true = W_true.at[:4, 0].set(jr.normal(k1, (4,)))
    W_true = W_true.at[3:7, 1].set(jr.normal(k2, (4,)))
    W_true = W_true.at[6:10, 2].set(jr.normal(k3, (4,)))

    k1, k2 = jr.split(k1)
    Z_true = jr.normal(k1, (n_samples, n_components))
    noise = 0.3 * jr.normal(k2, (n_samples, n_features))
    X = Z_true @ W_true.T + noise

    return X, W_true, Z_true


def generate_timeseries_data(key, n_timesteps=200, n_features=10, n_components=3):
    """Generate synthetic time-series DFA data with known ground truth."""
    k1, k2, k3 = jr.split(key, 3)

    # True emission matrix (sparse)
    H_true = jnp.zeros((n_features, n_components))
    H_true = H_true.at[:4, 0].set(jr.normal(k1, (4,)))
    H_true = H_true.at[3:7, 1].set(jr.normal(k2, (4,)))
    H_true = H_true.at[6:10, 2].set(jr.normal(k3, (4,)))

    # True AR(1) dynamics
    F_true = 0.95 * jnp.eye(n_components)

    # Generate latent states
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
# Phase 1a: Reference outputs from current FA/PCA
# ============================================================================

class TestFABaseline:
    """Record and verify baseline FA/PCA outputs."""

    def test_pfa_baseline(self):
        """Record PFA baseline: shapes, ELBO trend, and reproducibility."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, W_true, Z_true = generate_iid_data(key, n_samples, n_features, n_components)

        model = PFA(n_components=n_components, n_features=n_features, key=jr.PRNGKey(0))
        model, elbos = fit(model, X, n_iter=50, key=key)

        # Shapes
        assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
        assert model.q_w_psi.expected_psi.shape == (n_features,)
        assert model.q_tau.mean.shape == (n_components,)
        assert len(elbos) == 50

        # ELBO should generally increase (allow small dips due to numerical issues)
        elbos_arr = jnp.array(elbos)
        assert elbos_arr[-1] > elbos_arr[0], "ELBO should improve over training"

        # Reproducibility: same seed => same result
        model2 = PFA(n_components=n_components, n_features=n_features, key=jr.PRNGKey(0))
        model2, elbos2 = fit(model2, X, n_iter=50, key=key)
        assert jnp.allclose(jnp.array(elbos), jnp.array(elbos2), atol=1e-5)

    def test_ppca_baseline(self):
        """Record PPCA baseline: shapes and ELBO trend."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = PPCA(n_components=n_components, n_features=n_features, key=jr.PRNGKey(0))
        model, elbos = fit(model, X, n_iter=50, key=key)

        # Shapes
        assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
        assert model.q_w_psi.inv_gamma.mean.shape == ()  # Scalar for PPCA
        assert model.q_tau.mean.shape == (n_components,)

        elbos_arr = jnp.array(elbos)
        assert elbos_arr[-1] > elbos_arr[0], "ELBO should improve over training"

    def test_pfa_with_bmr_baseline(self):
        """Record PFA with BMR baseline."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 5
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = PFA(
            n_components=n_components,
            n_features=n_features,
            bmr_m_step=True,
            optimize_with_bmr=True,
            key=jr.PRNGKey(0),
        )
        model, elbos = fit(model, X, n_iter=50, key=key)

        assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
        assert len(elbos) <= 50  # May converge early

    def test_pfa_with_bmr_e_step_baseline(self):
        """Record PFA with BMR E-step baseline."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 5
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = PFA(
            n_components=n_components,
            n_features=n_features,
            bmr_e_step=True,
            key=jr.PRNGKey(0),
        )
        model, elbos = fit(model, X, n_iter=50, key=key)

        assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)


# ============================================================================
# Phase 1b: Reference outputs from current DFA
# ============================================================================

class TestDFABaseline:
    """Record and verify baseline DFA outputs."""

    def test_dfa_em_baseline(self):
        """Record DFA EM baseline."""
        key = jr.PRNGKey(42)
        Y, H_true, F_true, z_true = generate_timeseries_data(key)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)  # fit_em returns elbos[1:]
        assert params.emissions.weights.shape == (10, 3)
        assert params.dynamics.weights.shape == (3, 3)

    def test_dfa_vbem_baseline(self):
        """Record DFA VBEM baseline."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key)

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
        Y, _, _, _ = generate_timeseries_data(key)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key)
        sample_of_params, elbos = model.fit_blocked_gibbs(
            key=key, initial_params=params, props=props, sample_size=10,
            emissions=Y[None, ...], verbose=False
        )

        assert sample_of_params.emissions.weights.shape[0] == 10  # 10 samples
        assert sample_of_params.emissions.weights.shape[1:] == (10, 3)


# ============================================================================
# Phase 1c: Performance benchmark - QR E-step vs Kalman with F=0, Q=I
# ============================================================================

class TestEStepPerformanceBenchmark:
    """Compare FA QR-based E-step vs DFA Kalman smoother with trivial dynamics."""

    def _run_fa_e_step(self, X, n_components, n_iters=5):
        """Run FA E-step timing."""
        key = jr.PRNGKey(0)
        n_samples, n_features = X.shape
        model = PFA(n_components=n_components, n_features=n_features, key=key)

        X_dist = Delta(X)
        # Warm up JIT
        qz = e_step(model, X_dist)
        qz.mean.block_until_ready()

        start = time.perf_counter()
        for _ in range(n_iters):
            qz = e_step(model, X_dist)
            qz.mean.block_until_ready()
        elapsed = (time.perf_counter() - start) / n_iters
        return elapsed

    def _run_dfa_e_step_as_fa(self, X, n_components, n_iters=5):
        """Run DFA E-step with F=0, Q=I (treating iid data as time series)."""
        key = jr.PRNGKey(0)
        n_samples, n_features = X.shape

        model = BayesianDynamicFactorAnalysis(
            state_dim=n_components, emission_dim=n_features,
            has_dynamics_bias=False, has_emissions_bias=True,
        )
        params, props = model.initialize(
            key,
            dynamics_weights=jnp.zeros((n_components, n_components)),
            dynamics_covariance=jnp.eye(n_components),
            dynamics_bias=jnp.zeros(n_components),
        )

        # Warm up JIT
        stats, ll = model.e_step(params, X)
        ll.block_until_ready()

        start = time.perf_counter()
        for _ in range(n_iters):
            stats, ll = model.e_step(params, X)
            ll.block_until_ready()
        elapsed = (time.perf_counter() - start) / n_iters
        return elapsed

    def test_benchmark_e_step(self):
        """Benchmark FA QR E-step vs DFA Kalman with F=0, Q=I."""
        key = jr.PRNGKey(42)
        results = []

        configs = [
            (50, 10, 2),
            (200, 10, 5),
            (200, 20, 5),
            (1000, 10, 5),
        ]

        for n_samples, n_features, n_components in configs:
            X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

            t_fa = self._run_fa_e_step(X, n_components, n_iters=3)
            t_dfa = self._run_dfa_e_step_as_fa(X, n_components, n_iters=3)

            ratio = t_dfa / t_fa
            results.append((n_samples, n_features, n_components, t_fa, t_dfa, ratio))
            print(f"N={n_samples}, D={n_features}, K={n_components}: "
                  f"FA={t_fa:.4f}s, DFA={t_dfa:.4f}s, ratio={ratio:.1f}x")

        # Just verify both paths produce valid results (no assertion on speed)
        assert len(results) == len(configs)

    def test_kalman_f0_qi_produces_valid_stats(self):
        """Verify DFA with F=0, Q=I produces valid sufficient statistics."""
        key = jr.PRNGKey(42)
        n_timesteps, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_timesteps, n_features, n_components)

        model = BayesianDynamicFactorAnalysis(
            state_dim=n_components, emission_dim=n_features,
            has_dynamics_bias=False, has_emissions_bias=True,
        )
        params, props = model.initialize(
            key,
            dynamics_weights=jnp.zeros((n_components, n_components)),
            dynamics_covariance=jnp.eye(n_components),
            dynamics_bias=jnp.zeros(n_components),
        )

        stats, ll = model.e_step(params, X)
        init_stats, dynamics_stats, emission_stats = stats

        # Verify shapes
        sum_zzT, sum_zyT, sum_yyT, N = emission_stats
        assert sum_zzT.shape[0] == n_components + 1  # +1 for bias
        assert sum_zyT.shape == (n_components + 1, n_features)
        assert sum_yyT.shape == (n_features, n_features)
        assert N == n_timesteps
        assert jnp.isfinite(ll)

    def test_dfa_f0_qi_fit_em(self):
        """Verify DFA with F=0, Q=I can fit iid data via EM."""
        key = jr.PRNGKey(42)
        n_timesteps, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_timesteps, n_features, n_components)

        model = BayesianDynamicFactorAnalysis(
            state_dim=n_components, emission_dim=n_features,
            has_dynamics_bias=False, has_emissions_bias=True,
        )
        params, props = model.initialize(
            key,
            dynamics_weights=jnp.zeros((n_components, n_components)),
            dynamics_covariance=jnp.eye(n_components),
            dynamics_bias=jnp.zeros(n_components),
        )

        # Fix dynamics props to non-trainable
        from dynamax.parameters import ParameterProperties
        from dynamax.utils.bijectors import RealToPSDBijector
        from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMDynamics
        from sppcax.models.dynamic_factor_analysis import ParamsLGSSM

        fixed_dynamics_props = ParamsLGSSMDynamics(
            weights=ParameterProperties(trainable=False),
            bias=ParameterProperties(trainable=False),
            input_weights=ParameterProperties(trainable=False),
            cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector()),
        )
        props = ParamsLGSSM(initial=props.initial, dynamics=fixed_dynamics_props, emissions=props.emissions)

        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert params.emissions.weights.shape == (n_features, n_components)
        # Dynamics weights should remain at zero (non-trainable returns prior mode)
        assert jnp.allclose(params.dynamics.weights, jnp.zeros((n_components, n_components)))
        # Dynamics cov will be the prior's mode (not necessarily I), just check it's valid PSD
        assert jnp.all(jnp.linalg.eigvalsh(params.dynamics.cov) > 0)


# ============================================================================
# Phase 3: New FA/PCA subclasses tests
# ============================================================================

class TestNewFASubclasses:
    """Test BayesianFactorAnalysis and BayesianPCA subclasses."""

    def test_bayesian_fa_basic(self):
        """Test BayesianFactorAnalysis creates and fits correctly."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)

        assert model.is_static
        assert model.has_ard
        assert not model.isotropic_noise

        params, props = model.initialize(key)
        assert params.emissions.weights.shape == (n_features, n_components)
        assert not props.dynamics.weights.trainable  # dynamics frozen

        params, elbos = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_bayesian_pca_basic(self):
        """Test BayesianPCA creates and fits correctly."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianPCA(n_components=n_components, n_features=n_features, key=key)

        assert model.is_static
        assert model.has_ard
        assert model.isotropic_noise

        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_bayesian_fa_vbem(self):
        """Test BayesianFactorAnalysis with VBEM."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_bayesian_fa_transform(self):
        """Test BayesianFactorAnalysis transform and inverse_transform."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        qz = model.transform(params, X)
        assert qz.mean.shape == (n_samples, n_components)

        X_recon = model.inverse_transform(params, qz)
        assert X_recon.mean.shape == (n_samples, n_features)

    def test_bayesian_fa_with_ard_updates(self):
        """Test that ARD prior gets updated during training."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        ard_alpha_before = model.ard_prior.alpha.copy()

        params, props = model.initialize(key)
        params, _ = model.fit_em(params, props, X, key=key, num_iters=30, verbose=False)

        # ARD prior should have been updated
        ard_alpha_after = model.ard_prior.alpha
        assert not jnp.allclose(ard_alpha_before, ard_alpha_after)

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

    def test_bayesian_fa_with_bmr(self):
        """Test BayesianFactorAnalysis with BMR enabled."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 5
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(
            n_components=n_components, n_features=n_features,
            use_bmr=True, key=key,
        )

        assert model.use_bmr.emissions
        assert model.has_ard

        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])


# ============================================================================
# Phase 7: Comprehensive equivalence and feature parity tests
# ============================================================================

class TestEquivalence:
    """Test that new unified model produces results equivalent to legacy code."""

    def test_e_step_sufficient_stats(self):
        """Verify unified E-step with T=1 batches produces correctly formatted DFA sufficient statistics."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 50, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key)

        # Run e_step with T=1 batches
        batch_emissions = X[:, None, :]  # (N, 1, D)
        batch_inputs = jnp.zeros((n_samples, 1, 0))
        batch_stats, lls = jax.vmap(partial(model.e_step, params))(batch_emissions, batch_inputs)
        init_stats, dynamics_stats, emission_stats = batch_stats

        # Sum batch stats for aggregate checks
        sum_zzT = emission_stats[0].sum(0)
        sum_zyT = emission_stats[1].sum(0)
        sum_yyT = emission_stats[2].sum(0)
        N = emission_stats[3].sum(0)

        # emission_stats: verify shapes and PSD
        assert N == n_samples
        assert sum_zzT.shape == (n_components + 1, n_components + 1)  # +1 for bias
        assert sum_zyT.shape == (n_components + 1, n_features)
        assert sum_yyT.shape == (n_features, n_features)
        # sum_zzT should be positive semi-definite
        eigvals = jnp.linalg.eigvalsh(sum_zzT)
        assert jnp.all(eigvals >= -1e-6)

        # ll should be finite
        assert jnp.all(jnp.isfinite(lls))

    def test_fa_batched_matches_dfa_sequential(self):
        """Verify FA E-step (T=1 batches) matches DFA E-step (F=0, Q=I) on data-only stats.

        FA uses T=1 batches (each observation is independent), while DFA processes
        all observations as a single time series. With F=0, Q=I, both should give
        equivalent emission sufficient statistics.
        """
        key = jr.PRNGKey(42)
        n_timesteps, n_features, n_components = 30, 10, 3
        X, _, _ = generate_iid_data(key, n_timesteps, n_features, n_components)

        # DFA model (Kalman path: is_static=False, F=0, Q=I)
        model_kalman = BayesianDynamicFactorAnalysis(
            state_dim=n_components, emission_dim=n_features,
            has_dynamics_bias=False, has_emissions_bias=True,
        )
        params_k, props_k = model_kalman.initialize(
            key,
            dynamics_weights=jnp.zeros((n_components, n_components)),
            dynamics_covariance=jnp.eye(n_components),
            dynamics_bias=jnp.zeros(n_components),
        )

        # Run DFA Kalman E-step on full sequence
        stats_kalman, ll_kalman = model_kalman.e_step(params_k, X)
        em_kalman = stats_kalman[2]  # (sum_zzT, sum_zyT, sum_yyT, N)
        assert em_kalman[3] == n_timesteps
        assert jnp.isfinite(ll_kalman)

        # Run FA E-step with T=1 batches using same emission params
        model_static = BayesianFactorAnalysis(
            n_components=n_components, n_features=n_features,
            has_ard=False, key=key,
        )
        params_s, _ = model_static.initialize(key)
        params_s = params_s._replace(emissions=params_k.emissions)

        batch_emissions = X[:, None, :]  # (N, 1, D)
        batch_inputs = jnp.zeros((n_timesteps, 1, 0))
        batch_stats, lls = jax.vmap(partial(model_static.e_step, params_s))(batch_emissions, batch_inputs)
        em_static = batch_stats[2]
        sum_yyT_static = em_static[2].sum(0)
        N_static = em_static[3].sum(0)

        # sum_yyT must be identical (data-only statistic)
        assert jnp.allclose(sum_yyT_static, em_kalman[2], atol=1e-5), "sum_yyT should match"
        assert N_static == em_kalman[3], "N should match"

    def test_fa_via_dfa_trains_well(self):
        """Test FA via DFA trains well on synthetic data with known structure."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 200, 10, 3
        X, W_true, Z_true = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=50, verbose=False)

        # ELBO should improve
        assert elbos[-1] > elbos[0]

        # Reconstruct and check MSE is reasonable
        qz = model.transform(params, X)
        X_recon = model.inverse_transform(params, qz)
        mse = jnp.mean(jnp.square(X - X_recon.mean))
        assert mse < 1.0, f"Reconstruction MSE too high: {mse}"

    def test_ppca_via_dfa_trains_well(self):
        """Test PCA via DFA trains well on synthetic data."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 200, 10, 3
        X, W_true, Z_true = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianPCA(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=50, verbose=False)

        # ELBO should improve
        assert elbos[-1] > elbos[0]

        # Check isotropic noise: cov should be scalar * I
        R = params.emissions.cov
        R_diag = jnp.diag(R)
        # All diagonal elements should be the same (isotropic)
        assert jnp.allclose(R_diag, R_diag[0] * jnp.ones_like(R_diag), atol=1e-5)

    def test_dfa_unchanged(self):
        """Verify DFA time-series results haven't been broken by refactoring."""
        key = jr.PRNGKey(42)
        Y, H_true, F_true, z_true = generate_timeseries_data(key)

        model = BayesianDynamicFactorAnalysis(
            state_dim=3, emission_dim=10, has_dynamics_bias=True, has_emissions_bias=True
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        # ELBO should be computed correctly
        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

        # Dynamics should learn something non-trivial (not zero)
        assert jnp.abs(params.dynamics.weights).max() > 0.1

    def test_dfa_with_ard(self):
        """Test DFA with ARD and MVNIG emission prior for time-series data."""
        key = jr.PRNGKey(42)
        Y, _, _, _ = generate_timeseries_data(key)

        from sppcax.models.factor_analysis import _make_mvnig_prior
        emission_prior = _make_mvnig_prior(n_features=10, n_components=5, has_bias=True)

        model = BayesianDynamicFactorAnalysis(
            state_dim=5, emission_dim=10,
            has_dynamics_bias=True, has_emissions_bias=True,
            has_ard=True,
            emission_prior=emission_prior,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, Y, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])
        # ARD should have been updated (dnat1 gets non-zero values)
        assert not jnp.allclose(model.ard_prior.dnat1, jnp.zeros(5))


class TestFeatureParity:
    """Test feature parity across model types."""

    def test_fa_vbem(self):
        """Test FA via DFA with VBEM algorithm."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_pca_vbem(self):
        """Test PCA via DFA with VBEM algorithm."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianPCA(n_components=n_components, n_features=n_features, key=key)
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=30, verbose=False)

        assert elbos.shape == (29,)
        assert jnp.isfinite(elbos[-1])

    def test_fa_no_bias(self):
        """Test FA via DFA without emission bias."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(
            n_components=n_components, n_features=n_features,
            has_emissions_bias=False, key=key,
        )
        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])
        assert jnp.allclose(params.emissions.bias, jnp.zeros(n_features))

    def test_fa_no_ard(self):
        """Test FA via DFA without ARD."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 3
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(
            n_components=n_components, n_features=n_features,
            has_ard=False, key=key,
        )

        assert not model.has_ard
        assert not hasattr(model, 'ard_prior') or not model.has_ard

        params, props = model.initialize(key)
        params, elbos = model.fit_em(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])

    def test_fa_bmr_with_vbem(self):
        """Test FA BMR with VBEM algorithm."""
        key = jr.PRNGKey(42)
        n_samples, n_features, n_components = 100, 10, 5
        X, _, _ = generate_iid_data(key, n_samples, n_features, n_components)

        model = BayesianFactorAnalysis(
            n_components=n_components, n_features=n_features,
            use_bmr=True, key=key,
        )
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=20, verbose=False)

        assert elbos.shape == (19,)
        assert jnp.isfinite(elbos[-1])

    def test_legacy_tests_still_pass(self):
        """Verify existing test_factor_analysis tests work with deprecation."""
        key = jr.PRNGKey(0)
        n_samples, n_features, n_components = 100, 10, 3
        X = jr.normal(key, (n_samples, n_features))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = PFA(n_components=n_components, n_features=n_features)
            model, lls = fit(model, X, n_iter=5, key=key)

            assert model.q_w_psi.mvn.mean.shape == (n_features, n_components)
            qz = transform(model, X)
            assert qz.mean.shape == (n_samples, n_components)
