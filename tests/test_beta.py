"""Tests for the Beta distribution."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.distributions import Beta


def test_beta_init():
    """Test Beta distribution initialization."""
    # Test scalar parameters
    dist = Beta(alpha0=2.0, beta0=3.0)
    assert dist.alpha == 2.0
    assert dist.beta == 3.0
    assert dist.batch_shape == ()
    assert dist.event_shape == ()

    # Test array parameters
    alpha = jnp.array([1.0, 2.0])
    beta = jnp.array([3.0, 4.0])
    dist = Beta(alpha0=alpha, beta0=beta)
    assert jnp.allclose(dist.alpha, alpha)
    assert jnp.allclose(dist.beta, beta)
    assert dist.batch_shape == (2,)
    assert dist.event_shape == ()


def test_beta_properties():
    """Test Beta distribution properties."""
    dist = Beta(alpha0=2.0, beta0=5.0)

    # Test mean
    expected_mean = 2.0 / (2.0 + 5.0)
    assert jnp.allclose(dist.mean, expected_mean)

    # Test variance
    expected_var = (2.0 * 5.0) / ((2.0 + 5.0) ** 2 * (2.0 + 5.0 + 1))
    assert jnp.allclose(dist.variance, expected_var)


def test_beta_sampling():
    """Test Beta distribution sampling."""
    key = jr.PRNGKey(0)
    dist = Beta(alpha0=2.0, beta0=3.0)

    # Test single sample
    sample = dist.sample(key)
    assert sample.shape == ()
    assert 0 < sample < 1

    # Test multiple samples
    samples = dist.sample(key, sample_shape=(10,))
    assert samples.shape == (10,)
    assert jnp.all((samples > 0) & (samples < 1))


def test_beta_log_prob():
    """Test Beta distribution log probability."""
    dist = Beta(alpha0=2.0, beta0=3.0)

    # Test valid values
    x = jnp.array([0.2, 0.5, 0.8])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == (3,)

    # Manual calculation for comparison
    alpha, beta = 2.0, 3.0
    expected_log_prob = (
        (alpha - 1) * jnp.log(x)
        + (beta - 1) * jnp.log(1 - x)
        - jnp.log(jnp.exp(jnp.math.lgamma(alpha) + jnp.math.lgamma(beta) - jnp.math.lgamma(alpha + beta)))
    )
    assert jnp.allclose(log_prob, expected_log_prob)

    # Test invalid values
    invalid_x = jnp.array([-0.1, 1.1])
    invalid_log_prob = dist.log_prob(invalid_x)
    assert jnp.all(jnp.isinf(invalid_log_prob))
    assert jnp.all(invalid_log_prob < 0)


def test_beta_natural_parameters():
    """Test Beta distribution natural parameters."""
    dist = Beta(alpha0=2.0, beta0=3.0)

    # Test natural parameters
    nat_params = dist.natural_parameters
    assert nat_params.shape == (2,)
    assert jnp.allclose(nat_params, jnp.array([1.0, 2.0]))  # [α-1, β-1]

    # Test from_natural_parameters
    new_dist = Beta.from_natural_parameters(nat_params)
    assert jnp.allclose(new_dist.alpha, 2.0)
    assert jnp.allclose(new_dist.beta, 3.0)


def test_beta_sufficient_statistics():
    """Test Beta distribution sufficient statistics."""
    dist = Beta(alpha0=2.0, beta0=3.0)
    x = jnp.array([0.2, 0.5, 0.8])

    # Test sufficient statistics
    suff_stats = dist.sufficient_statistics(x)
    assert suff_stats.shape == (3, 2)
    expected_suff_stats = jnp.stack([jnp.log(x), jnp.log(1 - x)], axis=-1)
    assert jnp.allclose(suff_stats, expected_suff_stats)

    # Test expected sufficient statistics
    expected_suff_stats = dist.expected_sufficient_statistics
    assert expected_suff_stats.shape == (2,)
