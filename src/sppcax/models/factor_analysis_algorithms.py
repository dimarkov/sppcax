"""Bayesian Factor Analysis algorithm implementations."""

from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from sppcax.distributions.utils import safe_cholesky_and_logdet

from ..bmr import reduce_model
from ..distributions.base import Distribution
from ..distributions.mvn import MultivariateNormal
from ..types import Array
from .factor_analysis_params import BayesianFactorAnalysisParams, _to_distribution


def e_step(
    model: BayesianFactorAnalysisParams, X: Union[Array, Distribution], use_data_mask: bool = True
) -> MultivariateNormal:
    """E-step: Compute expected latent variables.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        use_data_mask: apply data mask (default True)

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

    exp_stats = qz.expected_sufficient_statistics
    Ez = exp_stats[..., : model.n_components]
    Ezz = jnp.sum(exp_stats[..., model.n_components :], 0).reshape(model.n_components, model.n_components)

    # Update loading matrix
    tau = model.W_dist.gamma.mean
    P = jnp.diag(tau) + Ezz
    nat1 = jnp.sum(Ez[..., None, :] * X_centered[..., None], axis=0)
    mvn = eqx.tree_at(lambda x: (x.nat1, x.nat2), model.W_dist.mvn, (nat1, -0.5 * P))
    W = mvn.mean
    sigma_sqr_w = jnp.diagonal(mvn.covariance, axis1=-1, axis2=-2)

    # update noise precision
    n_observed = jnp.sum(mask, axis=0)
    dnat1 = 0.5 * (n_observed + jnp.minimum(jnp.arange(model.n_features) + 1, model.n_components))

    dnat2 = -0.5 * (
        jnp.square(X_centered).sum(0) - 2 * jnp.sum((Ez @ W.mT) * X_centered, 0) + jnp.sum((W @ Ezz) * W, -1)
    )
    dnat2 -= 0.5 * (sigma_sqr_w + jnp.square(W)) @ tau

    if model.isotropic_noise:
        # Single precision for all features
        dnat1 = jnp.sum(dnat1)
        dnat2 = jnp.sum(dnat2)

    gamma_np = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.noise_precision, (dnat1, dnat2))

    # update tau
    dnat2 = -0.5 * jnp.sum(sigma_sqr_w + gamma_np.mean[..., None] * jnp.square(W), 0)

    # Create new W distribution
    gamma_tau = eqx.tree_at(lambda x: x.dnat2, model.W_dist.gamma, dnat2)
    new_W_dist = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.W_dist, (mvn, gamma_tau))

    # Update model with posterior distributions
    updated_model = eqx.tree_at(
        lambda x: (x.W_dist, x.noise_precision),
        model,
        (new_W_dist, gamma_np),
    )

    # Apply Bayesian Model Reduction if enabled
    if use_bmr:
        updated_model = reduce_model(updated_model)

    return updated_model


def fit(
    model: BayesianFactorAnalysisParams,
    X: Union[Array, Distribution],
    n_iter: int = 100,
    tol: float = 1e-6,
    use_bmr: bool = False,
    bmr_frequency: int = 1,
) -> Tuple[BayesianFactorAnalysisParams, List[float]]:
    """Fit the model using EM algorithm with optional Bayesian Model Reduction.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Training data of shape (n_samples, n_features) or Distribution instance
        n_iter: Maximum number of iterations
        tol: Convergence tolerance
        use_bmr: Whether to use Bayesian Model Reduction to create a sparse loading matrix
        bmr_frequency: Apply BMR every N M-steps

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
        qz = e_step(updated_model, X_dist)
        elbo_val = compute_elbo(updated_model, X_dist, qz)
        elbos.append(elbo_val)

        # M-step (returns updated model)
        use_bmr_now = use_bmr and ((n + 1) % bmr_frequency == 0)
        updated_model = m_step(updated_model, X_dist, qz, use_bmr=use_bmr_now)

        # Check convergence
        if jnp.abs(elbo_val - old_elbo) < tol:
            break
        old_elbo = elbo_val

    return updated_model, elbos


def transform(
    model: BayesianFactorAnalysisParams, X: Union[Array, Distribution], use_data_mask: bool = False
) -> MultivariateNormal:
    """Apply dimensionality reduction to X.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        use_data_mask: apply data mask (default False)

    Returns:
        qz: Posterior estimate of the latents as MultivariateNormal distribution
    """
    return e_step(model, X, use_data_mask=use_data_mask)


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
) -> Array:
    """Compute expected log likelihood E_q[log p(X|Z,W,τ)].

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
    E_x = exp_stats[..., :dim]
    E_xx = exp_stats[..., dim :: dim + 1]

    # Get mask
    mask = model._validate_mask(E_x)
    n_observed = jnp.sum(mask)

    # Get expectations for Z
    Ez = qz.mean
    Ezz = qz.covariance + Ez[..., None] * Ez[..., None, :]

    # Compute expected log likelihood
    exp_ll = -0.5 * n_observed * jnp.log(2 * jnp.pi)
    exp_ll += 0.5 * jnp.sum(mask * exp_log_prec)

    # E[(x - Wz)^T τ (x - Wz)]
    x_centered = E_x - model.mean_
    term1 = jnp.sum(exp_noise_precision * mask * E_xx)
    term2 = -2 * jnp.sum(exp_noise_precision * mask * (x_centered * (W @ Ez[..., None]).squeeze(-1)))
    term3 = jnp.trace((exp_noise_precision * mask)[..., None] * (W @ Ezz @ W.T), axis1=-1, axis2=-2).sum()

    exp_cov = model.W_dist.expected_covariance
    term4 = jnp.trace(mask[..., None, None] * (exp_cov @ jnp.expand_dims(Ezz, -3)), axis1=-1, axis2=-2).sum()

    exp_ll -= 0.5 * (term1 + term2 + term3 + term4)

    return exp_ll


def _kl_latent(qz: MultivariateNormal) -> Array:
    """Compute KL(q(Z)||p(Z)).

    Args:
        qz: Posterior distribution over latent variables

    Returns:
        KL divergence
    """
    return qz.kl_divergence(MultivariateNormal(loc=jnp.zeros_like(qz.mean), precision=jnp.eye(qz.shape[-1]))).sum()


def _kl_loading(model: BayesianFactorAnalysisParams) -> Array:
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


def _kl_noise(model: BayesianFactorAnalysisParams) -> Array:
    """Compute KL(q(τ)||p(τ)).

    Args:
        model: BayesianFactorAnalysis model parameters

    Returns:
        KL divergence
    """
    return model.noise_precision.kl_divergence_from_prior.sum()


def compute_elbo(model: BayesianFactorAnalysisParams, X: Union[Array, Distribution], qz: MultivariateNormal) -> Array:
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
