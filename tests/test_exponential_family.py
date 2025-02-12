"""Tests for exponential family distributions."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.distributions import Categorical, Gamma, MultivariateNormal, Normal, Poisson


def test_normal_distribution():
    """Test Normal distribution implementation."""
    # Test initialization with loc/scale
    loc = jnp.array([0.0, 1.0])
    scale = jnp.array([1.0, 2.0])
    dist = Normal(loc=loc, scale=scale)

    # Test shapes
    assert dist.batch_shape == (2,)
    assert dist.event_shape == ()

    # Test parameter properties
    assert jnp.allclose(dist.loc, loc)
    assert jnp.allclose(dist.scale, scale)
    assert jnp.allclose(dist.precision, 1.0 / (scale * scale))

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == dist.batch_shape + (2,)

    # Test parameter conversion
    dist2 = Normal.from_natural_parameters(eta)
    assert jnp.allclose(dist2.loc, loc)
    assert jnp.allclose(dist2.scale, scale)

    # Test log probability
    x = jnp.array([0.0, 0.0])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == dist.batch_shape

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100,) + dist.batch_shape + dist.event_shape

    # Test log probability with samples
    test_samples = samples[:5]  # Use first 5 samples for testing
    log_probs = dist.log_prob(test_samples)
    assert log_probs.shape == (5,) + dist.batch_shape
    assert jnp.all(jnp.isfinite(log_probs))

    # Verify symmetry around mean for equidistant points
    x1 = loc + scale
    x2 = loc - scale
    assert jnp.allclose(dist.log_prob(x1), dist.log_prob(x2))

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == dist.batch_shape
    assert jnp.all(entropy >= 0)


def test_gamma_distribution():
    """Test Gamma distribution implementation."""
    # Test initialization with alpha/beta
    alpha = jnp.array([2.0, 3.0])
    beta = jnp.array([1.0, 2.0])
    dist = Gamma(alpha=alpha, beta=beta)

    # Test shapes
    assert dist.batch_shape == (2,)
    assert dist.event_shape == ()

    # Test parameter properties
    assert jnp.allclose(dist.alpha, alpha)
    assert jnp.allclose(dist.beta, beta)

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == dist.batch_shape + (2,)
    assert jnp.allclose(eta[..., 0], alpha - 1.0)  # α-1
    assert jnp.allclose(eta[..., 1], -beta)  # -β

    # Test parameter conversion
    dist2 = Gamma.from_natural_parameters(eta)
    assert jnp.allclose(dist2.alpha, alpha)
    assert jnp.allclose(dist2.beta, beta)

    # Test log probability
    x = jnp.array([1.0, 2.0])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == dist.batch_shape

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100,) + dist.batch_shape + dist.event_shape
    assert jnp.all(samples > 0)

    # Test log probability with samples
    test_samples = samples[:5]  # Use first 5 samples for testing
    log_probs = dist.log_prob(test_samples)
    assert log_probs.shape == (5,) + dist.batch_shape
    assert jnp.all(jnp.isfinite(log_probs))

    # Test invalid values (out of range)
    invalid_samples = jnp.array([-1, 0])  # negative
    invalid_log_probs = dist.log_prob(invalid_samples)
    assert jnp.all(jnp.isneginf(invalid_log_probs))

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == dist.batch_shape
    assert jnp.all(entropy >= 0)


def test_kl_divergence():
    """Test KL divergence computation."""
    # Test Normal KL divergence
    loc1, scale1 = jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0])
    loc2, scale2 = jnp.array([1.0, 0.0]), jnp.array([2.0, 1.0])
    dist1 = Normal(loc=loc1, scale=scale1)
    dist2 = Normal(loc=loc2, scale=scale2)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == dist1.batch_shape
    assert jnp.all(kl >= 0)

    # Test Gamma KL divergence
    alpha1, beta1 = jnp.array([2.0, 3.0]), jnp.array([1.0, 2.0])
    alpha2, beta2 = jnp.array([1.0, 2.0]), jnp.array([2.0, 1.0])
    dist1 = Gamma(alpha=alpha1, beta=beta1)
    dist2 = Gamma(alpha=alpha2, beta=beta2)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == dist1.batch_shape
    assert jnp.all(kl >= 0)

    # Test MultivariateNormal KL divergence
    loc1 = jnp.array([0.0, 1.0])
    loc2 = jnp.array([1.0, 0.0])
    scale_tril1 = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    scale_tril2 = jnp.array([[2.0, 0.0], [0.3, 0.5]])
    mask = jnp.array([True, False])  # Test with mask

    dist1 = MultivariateNormal(loc=loc1, scale_tril=scale_tril1, mask=mask)
    dist2 = MultivariateNormal(loc=loc2, scale_tril=scale_tril2, mask=mask)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == ()
    assert kl >= 0

    # Test Categorical KL divergence
    logits1 = jnp.array([1.0, -1.0])  # 3 categories
    logits2 = jnp.array([0.0, 0.0])
    dist1 = Categorical(logits=logits1)
    dist2 = Categorical(logits=logits2)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == ()
    assert kl >= 0

    # Test Poisson KL divergence
    log_rate1 = jnp.array([0.0, 1.0])
    log_rate2 = jnp.array([1.0, 0.0])
    dist1 = Poisson(log_rate=log_rate1)
    dist2 = Poisson(log_rate=log_rate2)

    kl = dist1.kl_divergence(dist2)
    assert kl.shape == (2,)
    assert jnp.all(kl >= 0)


def test_multivariate_normal_distribution():
    """Test MultivariateNormal distribution implementation."""
    # Test initialization with loc only (identity scale)
    loc = jnp.array([1.0, 2.0])
    dist = MultivariateNormal(loc=loc)

    # Test shapes
    assert dist.batch_shape == ()
    assert dist.event_shape == (2,)
    assert dist.nat1.shape == (2,)
    assert dist.nat2.shape == (2, 2)

    # Test parameter properties
    assert jnp.allclose(dist.mean, loc)
    assert jnp.allclose(dist.precision, jnp.eye(2))

    # Test initialization with scale_tril
    scale_tril = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    dist = MultivariateNormal(loc=loc, scale_tril=scale_tril)
    assert jnp.allclose(dist.mean, loc)

    # Test initialization with covariance
    covariance = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    dist = MultivariateNormal(loc=loc, covariance=covariance)
    assert jnp.allclose(dist.mean, loc)

    # Test initialization with precision
    precision = jnp.array([[2.0, -0.5], [-0.5, 1.0]])
    dist = MultivariateNormal(loc=loc, precision=precision)
    assert jnp.allclose(dist.mean, loc)

    # Test error on multiple scale parameters
    try:
        dist = MultivariateNormal(loc=loc, scale_tril=scale_tril, covariance=covariance)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == (6,)  # [nat1(2), vec(nat2)(4)]

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100, 2)

    # Test log probability with samples
    test_samples = samples[:5]
    log_probs = dist.log_prob(test_samples)
    assert log_probs.shape == (5,)
    assert jnp.all(jnp.isfinite(log_probs))

    # Test vectorized inputs
    batch_log_probs = dist.log_prob(samples)
    assert batch_log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(batch_log_probs))

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == ()
    assert entropy >= 0


def test_multivariate_normal_masking():
    """Test MultivariateNormal masking functionality."""
    # Test basic masking
    loc = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([True, False, True])
    dist = MultivariateNormal(loc=loc, mask=mask)

    # Test masked mean
    assert jnp.allclose(dist.mean, jnp.array([1.0, 0.0, 3.0]))

    # Test masked precision
    expected_precision = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(dist.precision, expected_precision)

    # Test sampling with mask
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert jnp.allclose(samples[:, 1], 0.0)  # Masked dimension should be zero

    # Test log_prob with mask
    x = jnp.array([1.0, 0.0, 1.0])
    log_prob = dist.log_prob(x)
    assert jnp.isfinite(log_prob)

    # Test with scale_tril and mask
    scale_tril = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.3, 0.2, 1.0]])
    dist = MultivariateNormal(loc=loc, scale_tril=scale_tril, mask=mask)
    samples = dist.sample(key, sample_shape=(100,))
    assert jnp.allclose(samples[:, 1], 0.0)  # Masked dimension should be zero


def test_multivariate_normal_batch_masking():
    """Test MultivariateNormal with batched masking."""
    # Test batched masking
    loc = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([[True, False], [False, True]])
    dist = MultivariateNormal(loc=loc, mask=mask)

    # Test batch shapes
    assert dist.batch_shape == (2,)
    assert dist.event_shape == (2,)

    # Test masked means
    expected_means = jnp.array([[1.0, 0.0], [0.0, 4.0]])
    assert jnp.allclose(dist.mean, expected_means)

    # Test sampling with batched mask
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100, 2, 2)
    assert jnp.allclose(samples[:, 0, 1], 0.0)  # Second dim of first batch
    assert jnp.allclose(samples[:, 1, 0], 0.0)  # First dim of second batch

    # Test log_prob with batched mask
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == (2,)
    assert jnp.all(jnp.isfinite(log_prob))


def test_categorical_distribution():
    """Test Categorical distribution implementation."""
    # Test initialization with logits
    logits = jnp.array([1.0, -1.0, 1.0])  # 3 categories
    dist = Categorical(logits=logits)

    # Test shapes
    assert dist.batch_shape == ()
    assert dist.event_shape == ()

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == (2,)
    assert jnp.allclose(eta, logits[:-1] - 1.0)

    # Test sufficient statistics
    x = jnp.array(0)  # First category
    stats = dist.sufficient_statistics(x)
    assert stats.shape == (2,)
    assert jnp.allclose(stats, jnp.array([1.0, 0.0]))

    # Test expected sufficient statistics
    exp_stats = dist.expected_sufficient_statistics
    assert exp_stats.shape == (2,)
    assert jnp.all((exp_stats >= 0) & (exp_stats <= 1))

    # Test log probability
    log_prob = dist.log_prob(x)
    assert log_prob.shape == ()

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100,)
    assert jnp.all((samples >= 0) & (samples <= 2))

    # Test log probability with samples
    test_samples = samples[:5]  # Use first 5 samples for testing
    log_probs = dist.log_prob(test_samples)
    assert log_probs.shape == (5,) + dist.batch_shape
    assert jnp.all(jnp.isfinite(log_probs))

    # Test invalid values (negative and to large)
    invalid_samples = jnp.array([-1, 4])  # negative and to large
    invalid_log_probs = dist.log_prob(invalid_samples)
    assert jnp.all(jnp.isneginf(invalid_log_probs))

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == ()
    assert entropy >= 0


def test_poisson_distribution():
    """Test Poisson distribution implementation."""
    # Test initialization with log rate
    log_rate = jnp.array([0.0, 1.0])  # rates = [1.0, 2.718...]
    dist = Poisson(log_rate=log_rate)

    # Test shapes
    assert dist.batch_shape == (2,)
    assert dist.event_shape == ()
    assert dist.log_rate.shape == (2,)

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == (2,)
    assert jnp.allclose(eta, log_rate)

    # Test sufficient statistics
    x = jnp.array([1, 2])
    stats = dist.sufficient_statistics(x)
    assert stats.shape == (2,)
    assert jnp.allclose(stats, x)

    # Test expected sufficient statistics
    exp_stats = dist.expected_sufficient_statistics
    assert exp_stats.shape == (2,)
    assert jnp.all(exp_stats > 0)

    # Test log probability
    log_prob = dist.log_prob(x)
    assert log_prob.shape == (2,)

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100, 2)
    assert jnp.all(samples >= 0)

    # Test log probability with samples
    test_samples = samples[:5]  # Use first 5 samples for testing
    log_probs = dist.log_prob(test_samples)
    assert log_probs.shape == (5, 2)
    assert jnp.all(jnp.isfinite(log_probs))

    # Test invalid values (negative)
    invalid_samples = jnp.array([-1, -2])  # negative values
    invalid_log_probs = dist.log_prob(invalid_samples)
    assert jnp.all(jnp.isneginf(invalid_log_probs))

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == (2,)
    assert jnp.all(entropy >= 0)


def test_broadcasting():
    """Test parameter broadcasting."""
    # Test Normal broadcasting
    loc = jnp.array([0.0, 1.0])
    scale = 1.0
    dist = Normal(loc=loc, scale=scale)
    assert dist.batch_shape == (2,)
    assert jnp.allclose(dist.scale, jnp.ones(2))

    # Test Gamma broadcasting
    alpha = 2.0
    beta = jnp.array([1.0, 2.0])
    dist = Gamma(alpha=alpha, beta=beta)
    assert dist.batch_shape == (2,)
    assert jnp.allclose(dist.alpha, jnp.full(2, alpha))

    # Test MultivariateNormal broadcasting
    loc = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    scale_tril = jnp.array([1.0, 0.0, 0.5, 1.0]).reshape(2, 2)  # shape (2, 2)
    mask = jnp.array([[True, False], [True, True]])  # shape (2, 2)

    # Broadcasting mask to match loc's batch shape
    dist = MultivariateNormal(loc=loc, scale_tril=scale_tril, mask=mask)
    assert dist.batch_shape == (2,)
    assert dist.event_shape == (2,)

    # Test that mask is properly broadcast
    assert jnp.all(dist.mean[0, 1] == 0.0)  # Second dimension of the first batch is masked

    # Test Categorical broadcasting
    logits = jnp.array([1.0, -1.0])  # No broadcasting
    dist = Categorical(logits=logits)
    assert dist.batch_shape == ()

    # Test Poisson broadcasting
    log_rate = 0.0
    dist = Poisson(log_rate=log_rate)  # Scalar broadcasting
    assert dist.batch_shape == ()
    assert jnp.allclose(dist.log_rate, log_rate)
