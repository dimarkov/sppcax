"""Tests for exponential family distributions."""

import jax.numpy as jnp
import jax.random as jr
from sppcax.distributions import Bernoulli, Categorical, Gamma, MultivariateNormal, Normal, Poisson


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

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == dist.batch_shape
    assert jnp.all(entropy >= 0)


def test_bernoulli_distribution():
    """Test Bernoulli distribution implementation."""
    # Test initialization with logits
    logits = jnp.array([1.0, -1.0])
    dist = Bernoulli(logits=logits)

    # Test shapes
    assert dist.batch_shape == (2,)
    assert dist.event_shape == ()

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == dist.batch_shape + (1,)
    assert jnp.allclose(eta[..., 0], logits)

    # Test parameter conversion
    dist2 = Bernoulli.from_natural_parameters(eta)
    assert jnp.allclose(dist2.logits, dist.logits)

    # Test probabilities
    probs = dist.probs
    assert jnp.all((probs >= 0) & (probs <= 1))

    # Test log probability
    x = jnp.array([1, 0])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == dist.batch_shape

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100,) + dist.batch_shape + dist.event_shape
    assert jnp.all((samples == 0) | (samples == 1))

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

    # Test Bernoulli KL divergence
    logits1 = jnp.array([1.0, -1.0])
    logits2 = jnp.array([0.0, 0.0])
    dist1 = Bernoulli(logits=logits1)
    dist2 = Bernoulli(logits=logits2)

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
    dim = 2
    dist1 = MultivariateNormal(dim)
    dist2 = MultivariateNormal(dim)

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
    # Test initialization with natural parameters
    dim = 2
    dist = MultivariateNormal(dim)

    # Test shapes
    assert dist.batch_shape == ()
    assert dist.event_shape == (dim,)
    assert dist.nat1.shape == (dim,)
    assert dist.nat2.shape == (dim, dim)

    # Test parameter properties
    precision = -2.0 * dist.nat2
    mean = jnp.linalg.solve(precision, dist.nat1)
    assert jnp.allclose(mean, jnp.zeros(dim))  # Standard normal mean
    assert jnp.allclose(precision, jnp.eye(dim))  # Standard normal precision

    # Test natural parameters
    eta = dist.natural_parameters
    assert eta.shape == (dim + dim * dim,)  # [nat1, vec(nat2)]

    # Test log probability
    x = jnp.array([0.0, 0.0])
    log_prob = dist.log_prob(x)
    assert log_prob.shape == ()

    # Test sampling
    key = jr.PRNGKey(0)
    samples = dist.sample(key, sample_shape=(100,))
    assert samples.shape == (100, dim)

    # Test entropy
    entropy = dist.entropy
    assert entropy.shape == ()
    assert entropy >= 0


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

    # Test Bernoulli broadcasting
    logits = 0.0
    dist = Bernoulli(logits=logits)
    assert dist.batch_shape == ()

    # Test Gamma broadcasting
    alpha = 2.0
    beta = jnp.array([1.0, 2.0])
    dist = Gamma(alpha=alpha, beta=beta)
    assert dist.batch_shape == (2,)
    assert jnp.allclose(dist.alpha, jnp.full(2, alpha))

    # Test MultivariateNormal broadcasting
    dim = 2
    dist1 = MultivariateNormal(dim)  # No broadcasting
    assert dist1.batch_shape == ()

    # Test Categorical broadcasting
    logits = jnp.array([1.0, -1.0])  # No broadcasting
    dist = Categorical(logits=logits)
    assert dist.batch_shape == ()

    # Test Poisson broadcasting
    log_rate = 0.0
    dist = Poisson(log_rate=log_rate)  # Scalar broadcasting
    assert dist.batch_shape == ()
    assert jnp.allclose(dist.log_rate, log_rate)
