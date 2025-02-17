"""Tests for Delta distribution."""

import jax.numpy as jnp
from sppcax.distributions.delta import Delta, default_ss


def test_delta_log_prob():
    """Test log probability computation."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)

    # Test at location
    assert jnp.all(dist.log_prob(loc) == 0.0)

    # Test away from location
    x = jnp.array([1.0, 2.1])
    assert jnp.all(jnp.isinf(dist.log_prob(x)))
    assert jnp.all(dist.log_prob(x) < 0)


def test_delta_sample():
    """Test sampling (should always return location)."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)

    # Test single sample
    sample = dist.sample(jnp.array([0]))
    assert jnp.all(sample == loc)

    # Test multiple samples
    samples = dist.sample(jnp.array([0]), sample_shape=(3,))
    assert samples.shape == (3, 2)
    assert jnp.all(samples == loc)


def test_delta_entropy():
    """Test entropy (should always be 0)."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)
    assert jnp.all(dist.entropy() == 0.0)


def test_delta_sufficient_statistics():
    """Test sufficient statistics computation."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)

    # Test default sufficient statistics
    ss = dist.sufficient_statistics(loc)
    expected_ss = default_ss(loc)
    assert jnp.allclose(ss, expected_ss)

    # Test custom sufficient statistics
    dist_custom = Delta(loc, sufficient_statistics_fn=lambda x: x**2)
    assert jnp.allclose(dist_custom.sufficient_statistics(loc), loc**2)


def test_delta_expected_sufficient_statistics():
    """Test expected sufficient statistics."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)

    # Expected sufficient statistics should equal sufficient statistics at location
    assert jnp.allclose(dist.expected_sufficient_statistics, dist.sufficient_statistics(loc))


def test_delta_batch_shape():
    """Test batch shape handling."""
    # Single batch dimension
    loc = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    dist = Delta(loc)
    assert dist.batch_shape == (2,)
    assert dist.event_shape == (2,)

    # Multiple batch dimensions
    loc = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    dist = Delta(loc)
    assert dist.batch_shape == (2, 2)
    assert dist.event_shape == (2,)


def test_delta_broadcasting():
    """Test broadcasting behavior."""
    loc = jnp.array([1.0, 2.0])
    dist = Delta(loc)

    # Test broadcasting in log_prob
    x = jnp.array([[1.0, 2.0], [1.0, 2.0]])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == (2,)
    assert jnp.all(log_prob == 0.0)

    # Test broadcasting in sampling
    samples = dist.sample(jnp.array([0]), sample_shape=(3, 2))
    assert samples.shape == (3, 2, 2)
    assert jnp.all(samples == loc)
