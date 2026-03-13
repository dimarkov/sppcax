"""Tests for masked observation support in FA and DFA models.

Covers realistic masking scenarios: random per-element missing data for FA,
random temporal gaps and time-varying partial observations for DFA.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.factor_analysis import BayesianFactorAnalysis


# ============================================================================
# Helpers
# ============================================================================


def generate_iid_data(key, n_samples=200, n_features=10, n_components=3):
    """Generate synthetic iid FA data."""
    k1, k2, k3 = jr.split(key, 3)
    W_true = jnp.zeros((n_features, n_components))
    W_true = W_true.at[:4, 0].set(jr.normal(k1, (4,)))
    W_true = W_true.at[3:7, 1].set(jr.normal(k2, (4,)))
    W_true = W_true.at[6:10, 2].set(jr.normal(k3, (4,)))

    k1, k2 = jr.split(k1)
    Z_true = jr.normal(k1, (n_samples, n_components))
    noise = 0.3 * jr.normal(k2, (n_samples, n_features))
    return Z_true @ W_true.T + noise


def generate_timeseries_data(key, n_timesteps=200, n_features=10, n_components=3):
    """Generate synthetic time-series DFA data with Q=I dynamics noise."""
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
        z = z.at[t].set(F_true @ z[t - 1] + jr.normal(keys[t - 1], (n_components,)))

    k1, _ = jr.split(k1)
    noise = 0.5 * jr.normal(k1, (n_timesteps, n_features))
    return z @ H_true.T + noise


# ============================================================================
# All-True mask regression: mask=ones should match mask=None
# ============================================================================


class TestAllTrueMaskRegression:
    """An all-True mask must produce a similar ELBO as no mask."""

    def test_fa_vbem_all_true_mask(self):
        key = jr.PRNGKey(42)
        X = generate_iid_data(key)
        mask = jnp.ones(X.shape, dtype=bool)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, has_ard=False)
        params, props = model.initialize(key, variational_bayes=True)

        _, elbos_none = model.fit_vbem(params, props, X, key=key, num_iters=10)
        _, elbos_mask = model.fit_vbem(params, props, X, key=key, num_iters=10, mask=mask)

        assert jnp.allclose(
            elbos_none[-1], elbos_mask[-1], atol=0.5
        ), f"All-True mask ELBO differs from None: {elbos_mask[-1]:.4f} vs {elbos_none[-1]:.4f}"

    def test_dfa_vbem_all_true_mask(self):
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key)
        mask = jnp.ones(Y.shape, dtype=bool)

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10, has_ard=False)
        params, props = model.initialize(key, variational_bayes=True)

        _, elbos_none = model.fit_vbem(params, props, Y, key=key, num_iters=10)
        _, elbos_mask = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask)

        assert jnp.allclose(
            elbos_none[-1], elbos_mask[-1], atol=0.5
        ), f"All-True mask ELBO differs from None: {elbos_mask[-1]:.4f} vs {elbos_none[-1]:.4f}"


# ============================================================================
# FA: random per-element masking (the realistic FA scenario)
# ============================================================================


class TestFARandomMasking:
    """FA with random per-element missing data should converge."""

    @pytest.mark.parametrize("method", ["em", "vbem"], ids=["em", "vbem"])
    def test_fa_random_mask(self, method):
        """FA with 30% random missing entries should converge with non-zero weights."""
        key = jr.PRNGKey(42)
        X = generate_iid_data(key)

        k1, _ = jr.split(key)
        mask = jr.bernoulli(k1, p=0.7, shape=X.shape)

        model = BayesianFactorAnalysis(n_components=3, n_features=10, key=key)

        if method == "em":
            params, props = model.initialize(key)
            params, elbos = model.fit_em(params, props, X, key=key, num_iters=10, mask=mask)
        else:
            params, props = model.initialize(key, variational_bayes=True)
            params, elbos = model.fit_vbem(params, props, X, key=key, num_iters=10, mask=mask)

        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] > elbos[0]
        W = params.emissions.weights
        assert jnp.any(jnp.abs(W) > 0.1), "Emission weights collapsed to zero"


# ============================================================================
# DFA: random per-element masking
# ============================================================================


class TestDFARandomElementMasking:
    """DFA with random per-element missing data should converge."""

    def test_dfa_vbem_random_mask(self):
        """DFA VBEM with 30% random missing entries should converge."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key, n_features=10)

        k1, _ = jr.split(key)
        mask = jr.bernoulli(k1, p=0.7, shape=Y.shape)

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10)
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask)

        assert jnp.all(jnp.isfinite(elbos))
        W = params.emissions.weights
        assert jnp.any(jnp.abs(W) > 0.1), "Emission weights collapsed to zero"


# ============================================================================
# DFA: random temporal gaps (irregular sampling)
# ============================================================================


class TestDFARandomTemporalMask:
    """DFA with random temporal gaps (T,) mask — irregular sampling."""

    def test_dfa_vbem_random_temporal_mask(self):
        """DFA VBEM with random (T,) temporal gaps should converge."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key, n_timesteps=200, n_features=10)

        k1, _ = jr.split(key)
        mask_1d = jr.bernoulli(k1, p=0.7, shape=(200,))

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10)
        params, props = model.initialize(key, variational_bayes=True)
        _, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask_1d)

        assert jnp.all(jnp.isfinite(elbos))

    def test_temporal_mask_matches_broadcast(self):
        """A random (T,) mask should produce the same ELBO as its (T, D) broadcast."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key, n_timesteps=200, n_features=10)

        k1, _ = jr.split(key)
        mask_1d = jr.bernoulli(k1, p=0.7, shape=(200,))
        mask_2d = jnp.broadcast_to(mask_1d[:, None], (200, 10))

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10, has_ard=False)
        params, props = model.initialize(key, variational_bayes=True)

        _, elbos_1d = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask_1d)
        _, elbos_2d = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask_2d)

        assert jnp.allclose(
            elbos_1d[-1], elbos_2d[-1], atol=0.1
        ), f"(T,) mask ELBO differs from (T,D) broadcast: {elbos_1d[-1]:.4f} vs {elbos_2d[-1]:.4f}"


# ============================================================================
# DFA: time-varying partial observation
# ============================================================================


class TestDFATimeVaryingPartialMask:
    """DFA with time-varying partial observation (different dims at different times)."""

    def test_dfa_vbem_switching_dims(self):
        """First half dims 0-4, second half dims 5-9: should converge with non-zero weights."""
        key = jr.PRNGKey(42)
        T, D = 200, 10
        Y = generate_timeseries_data(key, n_timesteps=T, n_features=D)

        mask = jnp.zeros((T, D), dtype=bool)
        mask = mask.at[: T // 2, :5].set(True)
        mask = mask.at[T // 2 :, 5:].set(True)

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=D)
        params, props = model.initialize(key, variational_bayes=True)
        params, elbos = model.fit_vbem(params, props, Y, key=key, num_iters=10, mask=mask)

        assert jnp.all(jnp.isfinite(elbos))
        W = params.emissions.weights
        assert jnp.any(jnp.abs(W) > 0.1), "Emission weights collapsed to zero"


# ============================================================================
# Gibbs sampler masking
# ============================================================================


class TestGibbsMasking:
    """Test masked observations with blocked Gibbs sampler."""

    def test_gibbs_random_element_mask(self):
        """Gibbs sampler should run with random per-element mask."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key, n_timesteps=100, n_features=10)

        k1, _ = jr.split(key)
        mask = jr.bernoulli(k1, p=0.7, shape=Y.shape)

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10)
        params, props = model.initialize(key)
        sample_of_params, elbos = model.fit_blocked_gibbs(
            key=key,
            initial_params=params,
            props=props,
            sample_size=5,
            emissions=Y[None, ...],
            mask=mask[None, ...],
            verbose=False,
        )

        assert jnp.all(jnp.isfinite(elbos))
        assert sample_of_params.emissions.weights.shape[1:] == (10, 3)

    def test_gibbs_random_temporal_mask(self):
        """Gibbs sampler should run with random (T,) temporal mask."""
        key = jr.PRNGKey(42)
        Y = generate_timeseries_data(key, n_timesteps=100, n_features=10)

        k1, _ = jr.split(key)
        mask_1d = jr.bernoulli(k1, p=0.7, shape=(100,))

        model = BayesianDynamicFactorAnalysis(state_dim=3, emission_dim=10)
        params, props = model.initialize(key)
        sample_of_params, elbos = model.fit_blocked_gibbs(
            key=key,
            initial_params=params,
            props=props,
            sample_size=5,
            emissions=Y[None, ...],
            mask=mask_1d[None, ...],
            verbose=False,
        )

        assert jnp.all(jnp.isfinite(elbos))
