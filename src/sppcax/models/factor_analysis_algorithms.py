"""Bayesian Factor Analysis algorithm implementations."""

from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.linalg import solve_triangular

from sppcax.distributions.utils import safe_cholesky_and_logdet

from ..bmr import reduce_model
from ..distributions import Delta, Distribution, MultivariateNormal
from ..types import Array, Matrix, PRNGKey
from .factor_analysis_params import BayesianFactorAnalysisParams


def _to_distribution(X: Union[Array, Distribution]) -> Distribution:
    """Convert input to a Distribution if it isn't already.

    Args:
        X: Input data, either an Array or Distribution

    Returns:
        Distribution instance
    """
    if isinstance(X, Distribution):
        return X
    return Delta(X)


def e_step(
    model: BayesianFactorAnalysisParams,
    X: Union[Matrix, Distribution],
    use_data_mask: bool = True,
    use_bmr: bool = False,
    key: PRNGKey = None,
) -> MultivariateNormal:
    """E-step: Compute expected latent variables.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        use_data_mask: apply data mask (default True)
        use_bmr: apply Bayesian Model Reduction (default False)
        key: random number generator key

    Returns:
        qz: Posterior distribution over latents z in the form of a MultivariateNormal distribution
    """
    X_dist = _to_distribution(X)
    X_mean = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location
    mask = model._validate_mask(X_mean) if use_data_mask else jnp.ones_like(X_mean, dtype=bool)
    X_centered = X_mean - model.mean_

    # Get current loading matrix and noise precision
    W = model.W_dist.mean.mT  # (n_components, n_features)
    sqrt_noise_precision = jnp.sqrt(model.noise_precision.mean)
    if model.isotropic_noise:
        sqrt_noise_precision = jnp.full(model.n_features, sqrt_noise_precision)

    # Scale noise precision by mask
    sqrt_noise_precision = jnp.where(mask, sqrt_noise_precision, 0.0)

    # Compute posterior parameters
    scaled_W = W * sqrt_noise_precision[..., None, :]
    masked_cov = jnp.sum(mask[..., None, None] * model.W_dist.mvn.covariance, -3)
    P = masked_cov + scaled_W @ scaled_W.mT + jnp.eye(model.n_components)
    q, r = jnp.linalg.qr(P)
    q_inv = q.mT

    # Compute expectations
    Ez = solve_triangular(
        r,
        q_inv @ ((X_centered * sqrt_noise_precision)[..., None, :] @ scaled_W.mT).mT,
    )

    Ez = Ez.squeeze(-1)
    qz = MultivariateNormal(loc=Ez, precision=P)

    if use_bmr:
        qz = reduce_model(qz, key=key, **model.bmr_e_step.opts)

    return qz


def m_step(
    model: BayesianFactorAnalysisParams, X: Union[Array, Distribution], qz: MultivariateNormal, use_bmr: bool = False
) -> BayesianFactorAnalysisParams:
    """VBM-step: Update parameters.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        qz: Posterior estimates of latent states obtained during the variational e-step (n_samples, n_components)
        use_bmr: Whether to apply Bayesian Model Reduction

    Returns:
        Updated model instance
    """
    X_dist = _to_distribution(X)

    E_x = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location

    mask = model._validate_mask(E_x)
    X_centered = jnp.where(mask, E_x - model.mean_, 0.0)
    sigma_sqr_x = jnp.where(mask, jnp.diagonal(X_dist.covariance, axis1=-1, axis2=-2), 0.0).sum(0)

    exp_stats = qz.expected_sufficient_statistics
    Ez = exp_stats[..., : model.n_components]
    Ezz = jnp.sum(exp_stats[..., model.n_components :], 0).reshape(model.n_components, model.n_components)
    cov_z = Ezz - jnp.sum(Ez[..., None] * Ez[..., None, :], 0)

    # Update loading matrix
    tau = model.W_dist.gamma.mean
    P = jnp.diag(tau) + Ezz
    nat1 = jnp.sum(Ez[..., None, :] * X_centered[..., None], axis=0)
    mvn = eqx.tree_at(lambda x: (x.nat1, x.nat2), model.W_dist.mvn, (nat1, -0.5 * P))
    W = mvn.mean
    sigma_sqr_w = jnp.diagonal(mvn.covariance, axis1=-1, axis2=-2)

    # update noise precision
    n_observed = jnp.sum(mask, axis=0)
    dnat1 = 0.5 * (n_observed + mvn.mask.sum(-1))

    dnat2 = -0.5 * (jnp.square(X_centered - Ez @ W.mT).sum(0) + jnp.sum((W @ cov_z) * W, -1))
    dnat2 -= 0.5 * (sigma_sqr_w + jnp.square(W)) @ tau
    dnat2 -= 0.5 * sigma_sqr_x

    nat2_0 = dnat2 * (model.noise_precision.nat1_0 + 1) / dnat1 if use_bmr else None

    if model.isotropic_noise:
        # Single precision for all features
        dnat1 = jnp.sum(dnat1)
        dnat2 = jnp.sum(dnat2)
        nat2_0 = jnp.sum(nat2_0) if use_bmr else None

    if use_bmr:
        gamma_np = eqx.tree_at(lambda x: (x.dnat1, x.dnat2, x.nat2_0), model.noise_precision, (dnat1, dnat2, nat2_0))
    else:
        gamma_np = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.noise_precision, (dnat1, dnat2))

    # update tau
    dnat1 = mvn.mask.sum(0) / 2
    dnat2 = -0.5 * jnp.sum(sigma_sqr_w + gamma_np.mean[..., None] * jnp.square(W), 0)
    nat2_0 = dnat2 * (model.W_dist.gamma.nat1_0 + 1)
    nat2_0 = jnp.where(dnat1 > 0, nat2_0 / dnat1, model.W_dist.gamma.nat2_0)

    # Create new W distribution
    gamma_tau = eqx.tree_at(lambda x: (x.dnat1, x.dnat2, x.nat2_0), model.W_dist.gamma, (dnat1, dnat2, nat2_0))
    new_W_dist = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.W_dist, (mvn, gamma_tau))

    # Update model with posterior distributions
    updated_model = eqx.tree_at(
        lambda x: (x.W_dist, x.noise_precision),
        model,
        (new_W_dist, gamma_np),
    )

    return updated_model


def fit(
    model: BayesianFactorAnalysisParams,
    X: Union[Matrix, Distribution],
    n_iter: int = 100,
    tol: float = 1e-6,
    bmr_frequency: int = 1,
    key: PRNGKey = None,
) -> Tuple[BayesianFactorAnalysisParams, List[float]]:
    """Fit the model using EM algorithm with optional Bayesian Model Reduction.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Training data of shape (n_samples, n_features) or Distribution instance
        n_iter: Maximum number of iterations
        tol: Convergence tolerance
        bmr_frequency: Apply BMR every N M-steps
        key: Random number generator key

    Returns:
        model: Fitted model instance
        elbos: a list of elbo values at each iteration step
    """
    # Convert input to distribution if needed
    X_dist = _to_distribution(X)
    X_mean = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location

    # Create new instance with updated mean and BMR settings
    mask = model._validate_mask(X_mean)
    updated_model = eqx.tree_at(lambda x: x.mean_, model, jnp.sum(mask * X_mean, axis=0) / jnp.sum(mask, axis=0))

    # EM algorithm
    old_elbo = -jnp.inf
    elbos = []
    for n in range(n_iter):
        # E-step
        key, _key = jr.split(key)
        use_bmr = updated_model.bmr_e_step.use and (n > 32)
        qz = eqx.filter_jit(e_step)(updated_model, X_dist, use_bmr=use_bmr, key=_key)
        elbo_val = eqx.filter_jit(compute_elbo)(updated_model, X_dist, qz)
        elbos.append(elbo_val)

        # M-step (returns updated model)
        updated_model = eqx.filter_jit(m_step)(updated_model, X_dist, qz, use_bmr=updated_model.optimize_with_bmr)

        # Apply Bayesian Model Reduction if enabled
        if updated_model.bmr_m_step.use and ((n + 1) % bmr_frequency == 0):
            key, _key = jr.split(key)
            updated_model = eqx.filter_jit(reduce_model)(updated_model, key=_key, **updated_model.bmr_m_step.opts)

        # Check convergence
        if jnp.abs(elbo_val - old_elbo) < tol:
            break
        old_elbo = elbo_val

    return updated_model, elbos


def transform(
    model: BayesianFactorAnalysisParams,
    X: Union[Matrix, Distribution],
    use_data_mask: bool = False,
    use_bmr: bool = False,
    key: PRNGKey = None,
) -> MultivariateNormal:
    """Apply dimensionality reduction to X.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        use_data_mask: apply data mask (default False)
        use_bmr: apply Bayesian Model Reduction (default False)
        key: Random number generator key

    Returns:
        qz: Posterior estimate of the latents as MultivariateNormal distribution
    """
    return e_step(model, X, use_data_mask=use_data_mask, use_bmr=use_bmr, key=key)


def inverse_transform(model: BayesianFactorAnalysisParams, Z: Union[Array, Distribution]) -> MultivariateNormal:
    """Transform latent states back to its original space.

    Args:
        model: BayesianFactorAnalysis model parameters
        Z: Data in transformed space of shape (n_samples, n_components) or a Distribution

    Returns:
        X_original: Prediction of the data as a MultivariateNormal distribution
    """
    Z_dist = _to_distribution(Z)
    W = model.W_dist.mean.mT
    loc = Z_dist.mean @ W + model.mean_
    covariance = (1 / model.noise_precision.mean)[..., None] * jnp.eye(model.n_features) + W.mT @ Z_dist.covariance @ W
    return MultivariateNormal(loc=loc, covariance=covariance)


def _expected_log_likelihood(
    model: BayesianFactorAnalysisParams, X_dist: Distribution, qz: MultivariateNormal
) -> float:
    """Compute expected log likelihood E_q[log p(X|Z,W, psi)].

    Args:
        model: BayesianFactorAnalysis model parameters
        X_dist: Distribution over observations
        qz: Posterior distribution over latent variables

    Returns:
        Expected log likelihood
    """
    # Get parameters
    W = model.W_dist.mean
    exp_stats = model.noise_precision.expected_sufficient_statistics
    exp_log_prec = exp_stats[..., 0]
    exp_noise_precision = exp_stats[..., 1]
    if model.isotropic_noise:
        exp_noise_precision = jnp.full(model.n_features, exp_noise_precision)

    # Get expected sufficient statistics
    dim = X_dist.event_shape[0]
    exp_stats = X_dist.expected_sufficient_statistics
    m = model.mean_
    E_x = exp_stats[..., :dim]
    E_x_centered = exp_stats[..., :dim] - m
    E_xx_centered = exp_stats[..., dim :: dim + 1] + m**2 - 2 * m * E_x

    # Get mask
    mask = model._validate_mask(E_x)
    n_observed = jnp.sum(mask)

    # Get expectations for Z
    Ez = qz.mean
    Ezz = qz.covariance + Ez[..., None] * Ez[..., None, :]

    # Compute expected log likelihood
    exp_ll = -0.5 * n_observed * jnp.log(2 * jnp.pi)
    exp_ll += 0.5 * jnp.sum(mask * exp_log_prec)

    # E[(x - Wz - m)^T \psi (x - Wz- m)]
    term1 = jnp.sum(exp_noise_precision * mask * E_xx_centered)
    term2 = -2 * jnp.sum(exp_noise_precision * mask * (E_x_centered * (W @ Ez[..., None]).squeeze(-1)))
    term3 = jnp.trace((exp_noise_precision * mask)[..., None] * (W @ Ezz @ W.T), axis1=-1, axis2=-2).sum()

    cov = model.W_dist.mvn.covariance
    term4 = jnp.trace(mask[..., None, None] * (cov @ jnp.expand_dims(Ezz, -3)), axis1=-1, axis2=-2).sum()

    exp_ll -= 0.5 * (term1 + term2 + term3 + term4)

    return exp_ll


def _kl_latent(qz: MultivariateNormal) -> float:
    """Compute KL(q(Z)||p(Z)).

    Args:
        qz: Posterior distribution over latent variables

    Returns:
        KL divergence
    """
    return qz.kl_divergence(MultivariateNormal(loc=jnp.zeros_like(qz.mean), precision=jnp.eye(qz.shape[-1]))).sum()


def _kl_loading(model: BayesianFactorAnalysisParams) -> float:
    """Compute KL(q(W)||p(W)).

    Args:
        model: BayesianFactorAnalysis model parameters

    Returns:
        KL divergence
    """
    exp_tau = model.W_dist.gamma.mean
    exp_phi = model.noise_precision.mean * jnp.ones(model.n_features)
    W = model.W_dist.mvn.mean
    cov = model.W_dist.mvn.covariance
    sigma_sqr_w = jnp.diagonal(cov, axis1=-1, axis2=-2)

    exp_ln_tau = model.W_dist.gamma.expected_sufficient_statistics[..., 0]
    kl = -jnp.sum((exp_ln_tau - jnp.log(2 * jnp.pi)) * (model.n_features - jnp.arange(model.n_components))) / 2
    kl -= model.W_dist.mvn.mask.sum() * jnp.log(2 * jnp.pi) / 2
    kl += 0.5 * jnp.inner(exp_phi, (jnp.square(W) + sigma_sqr_w) @ exp_tau)
    _, logdet = safe_cholesky_and_logdet(model.W_dist.mvn.precision)
    kl += 0.5 * logdet.sum()
    return kl


def _kl_noise(model: BayesianFactorAnalysisParams) -> float:
    """Compute KL(q(τ)||p(τ)).

    Args:
        model: BayesianFactorAnalysis model parameters

    Returns:
        KL divergence
    """
    return model.noise_precision.kl_divergence_from_prior.sum()


def compute_elbo(model: BayesianFactorAnalysisParams, X: Union[Matrix, Distribution], qz: MultivariateNormal) -> float:
    """Compute Evidence Lower Bound (ELBO).

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix or distribution over observations
        qz: Posterior distribution over latent variables

    Returns:
        ELBO value
    """
    X_dist = _to_distribution(X)

    # Expected log likelihood
    exp_ll = _expected_log_likelihood(model, X_dist, qz)

    # KL terms
    kl_z = _kl_latent(qz)
    kl_w = _kl_loading(model)
    kl_tau = model.W_dist.gamma.kl_divergence_from_prior.sum(-1)
    kl_phi = _kl_noise(model)

    return exp_ll - kl_z - kl_w - kl_tau - kl_phi
