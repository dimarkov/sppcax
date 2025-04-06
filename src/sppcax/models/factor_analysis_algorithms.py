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
        key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation.

    Returns:
        qz: Posterior distribution over latents z in the form of a MultivariateNormal distribution
    """
    X_dist = _to_distribution(X)
    X_mean = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location
    mask = model._validate_mask(X_mean) if use_data_mask else jnp.ones_like(X_mean, dtype=bool)
    X_centered = X_mean - model.mean_

    # Get current loading matrix and noise precision expectations
    W = model.q_w_psi.mvn.mean.mT  # (n_components, n_features)
    sqrt_noise_precision = jnp.sqrt(model.q_w_psi.expected_psi)

    # Scale noise precision by mask
    sqrt_noise_precision = jnp.where(mask, sqrt_noise_precision, 0.0)

    # Compute posterior parameters
    scaled_W = W * sqrt_noise_precision[..., None, :]
    masked_cov = jnp.sum(mask[..., None, None] * model.q_w_psi.mvn.covariance, -3)
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
        use_bmr: Whether to apply Bayesian Model Reduction based optimisation for q(tau) and q(psi)

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

    # --- Update q(W, psi) ---
    # Update MVN part (q(W | psi))
    exp_tau = model.q_tau.mean  # Expected precision from ARD prior
    P_w = jnp.diag(exp_tau) + Ezz
    nat1_w = jnp.sum(Ez[..., None, :] * X_centered[..., None], axis=0)
    updated_mvn = eqx.tree_at(lambda x: (x.nat1, x.nat2), model.q_w_psi.mvn, (nat1_w, -0.5 * P_w))
    W = updated_mvn.mean
    sigma_sqr_w = jnp.diagonal(updated_mvn.covariance, axis1=-1, axis2=-2)

    # Update Gamma part (q(psi)) - noise precision
    n_observed = jnp.sum(mask, axis=0)
    # Natural parameters for Gamma distribution q(psi)
    # dnat1 is shape parameter update (alpha - 1)
    # dnat2 is rate parameter update (-beta)
    dnat1_psi = 0.5 * (
        n_observed + updated_mvn.mask.sum(-1)
    )  # Shape parameter update depends on observed data and effective features per component
    # Rate parameter update depends on reconstruction error and uncertainty
    term1_psi = jnp.square(X_centered - Ez @ W.mT).sum(0)  # Squared error term
    term2_psi = jnp.sum((W @ cov_z) * W, -1)  # Uncertainty from latent covariance
    term3_psi = (sigma_sqr_w + jnp.square(W)) @ exp_tau  # Uncertainty from W posterior, scaled by ARD precision
    term4_psi = sigma_sqr_x  # Uncertainty from input data X
    dnat2_psi = -0.5 * (term1_psi + term2_psi + term3_psi + term4_psi)

    # Handle isotropic noise: sum parameters if noise is shared
    if model.isotropic_noise:
        dnat1_psi = jnp.sum(dnat1_psi)
        dnat2_psi = jnp.sum(dnat2_psi)

    # Update the Gamma distribution q(psi) within q_w_psi
    updated_q_psi = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.q_w_psi.gamma, (dnat1_psi, dnat2_psi))

    # --- Update q(tau) --- ARD prior precision
    dnat1_tau = updated_mvn.mask.sum(0) / 2  # Shape parameter update depends on effective features per component
    # Rate parameter update depends on expected squared loadings scaled by noise precision
    exp_psi = updated_q_psi.mean  # Expected noise precision

    dnat2_tau = -0.5 * jnp.sum((sigma_sqr_w + jnp.square(W)) * exp_psi[..., None], 0)
    updated_q_tau = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.q_tau, (dnat1_tau, dnat2_tau))

    if use_bmr:
        # BMR optimisation of gamma prior hyper-parameters
        # Update q_tau prior rate (beta0)
        nat2_0_tau = updated_q_tau.dnat2 * (updated_q_tau.nat1_0 + 1)
        nat2_0_tau = jnp.where(updated_q_tau.dnat1 > 0, nat2_0_tau / updated_q_tau.dnat1, updated_q_tau.nat2_0)
        updated_q_tau = eqx.tree_at(lambda x: x.nat2_0, updated_q_tau, nat2_0_tau)

        # Update q_psi prior rate (beta0)
        nat2_0_psi = updated_q_psi.dnat2 * (updated_q_psi.nat1_0 + 1) / (updated_q_psi.dnat1 + 1e-8)
        updated_q_psi = eqx.tree_at(lambda x: x.nat2_0, updated_q_psi, nat2_0_psi)

    # Update model with new posterior distributions
    # Combine updated MVN and Gamma parts back into q_w_psi
    updated_q_w_psi = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.q_w_psi, (updated_mvn, updated_q_psi))

    updated_model = eqx.tree_at(
        lambda x: (x.q_w_psi, x.q_tau),
        model,
        (updated_q_w_psi, updated_q_tau),
    )

    return updated_model


def fit(
    model: BayesianFactorAnalysisParams,
    X: Union[Matrix, Distribution],
    *,
    key: PRNGKey,
    n_iter: int = 100,
    tol: float = 1e-6,
    bmr_frequency: int = 1,
) -> Tuple[BayesianFactorAnalysisParams, List[float]]:
    """Fit the model using EM algorithm with optional Bayesian Model Reduction.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Training data of shape (n_samples, n_features) or Distribution instance
        key: A `jax.random.PRNGKey` used to provide randomness to bmr-gibbs sampling.
        n_iter: Maximum number of iterations
        tol: Convergence tolerance
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
        key, _key = jr.split(key)
        use_bmr = updated_model.bmr_e_step.use and (n > 32)
        qz = eqx.filter_jit(e_step)(updated_model, X_dist, use_bmr=use_bmr, key=_key)
        elbo_val = eqx.filter_jit(compute_elbo)(updated_model, X_dist, qz)
        elbos.append(elbo_val)

        # M-step (returns updated model)
        # updated_model = eqx.filter_jit(m_step)(updated_model, X_dist, qz, use_bmr=updated_model.optimize_with_bmr)
        updated_model = m_step(updated_model, X_dist, qz, use_bmr=updated_model.optimize_with_bmr)

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
    *,
    key: PRNGKey = None,
) -> MultivariateNormal:
    """Apply dimensionality reduction to X.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        use_data_mask: apply data mask (default False)
        use_bmr: apply Bayesian Model Reduction (default False)
        key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

    Returns:
        qz: Posterior estimate of the latents as MultivariateNormal distribution
    """
    return e_step(model, X, use_data_mask=use_data_mask, use_bmr=model.bmr_m_step.use, key=key)


def inverse_transform(model: BayesianFactorAnalysisParams, Z: Union[Array, Distribution]) -> MultivariateNormal:
    """Transform latent states back to its original space.

    Args:
        model: BayesianFactorAnalysis model parameters
        Z: Data in transformed space of shape (n_samples, n_components) or a Distribution

    Returns:
        X_original: Prediction of the data as a MultivariateNormal distribution
    """
    Z_dist = _to_distribution(Z)
    W = model.q_w_psi.mvn.mean.mT
    loc = Z_dist.mean @ W + model.mean_
    # Use expected noise precision E[psi] = alpha/beta
    exp_noise_precision = model.q_w_psi.expected_psi
    covariance = jnp.diag(1 / exp_noise_precision) + W.mT @ Z_dist.covariance @ W
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
    # Get parameters and expectations
    W = model.q_w_psi.mvn.mean
    exp_stats_psi = model.q_w_psi.expected_sufficient_statistics_psi
    exp_log_psi = exp_stats_psi[..., 0]  # E[log psi]
    exp_psi = exp_stats_psi[..., 1]  # E[psi]

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

    # Compute expected log likelihood E[log p(X | Z, W, psi)]
    exp_ll = -0.5 * n_observed * jnp.log(2 * jnp.pi)
    exp_ll += 0.5 * jnp.sum(mask * exp_log_psi)  # Use E[log psi]

    # Expectation of the quadratic term E[(x - Wz - m)^T psi (x - Wz - m)]
    term1 = jnp.sum(exp_psi * mask * E_xx_centered)  # E[x^T psi x] related terms
    term2 = -2 * jnp.sum(
        exp_psi * mask * (E_x_centered * (W @ Ez[..., None]).squeeze(-1))
    )  # E[x^T psi W z] related terms
    # E[z^T W^T psi W z] related terms, including uncertainty in W
    term3 = jnp.trace((exp_psi * mask)[..., None] * (W @ Ezz @ W.T), axis1=-1, axis2=-2).sum()  # Part from E[W]
    cov_w = model.q_w_psi.mvn.covariance
    # Term accounting for covariance of W, Tr(E[psi] * E[Z Z^T] * Cov[W])
    term4 = jnp.trace(jnp.expand_dims(mask, (-1, -2)) * (cov_w @ jnp.expand_dims(Ezz, -3)), axis1=-1, axis2=-2).sum()

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


def _kl_loading_and_psi(model: BayesianFactorAnalysisParams) -> float:
    """Compute KL(q(W, psi) || bar{p}(W | tau, psi) p(psi)).

    This combines KL divergence for W and psi, as they are coupled in q_w_psi.
    KL(q(W, psi) || bar{p}(W|tau, psi)p(psi)) = E_q[log q(W, psi) - <log p(W|tau, psi)>_q(tau) - log p(psi)].

    Args:
        model: BayesianFactorAnalysis model parameters

    Returns:
        Combined KL divergence for W and psi.
    """
    # KL divergence for q(W | psi) relative to prior <p(W | tau, psi)>_q(tau)
    exp_tau = model.q_tau.mean  # E[tau]
    exp_log_tau = model.q_tau.expected_sufficient_statistics[..., 0]  # E[log tau]
    exp_psi = model.q_w_psi.expected_psi  # E[psi]
    W = model.q_w_psi.mvn.mean
    cov_w = model.q_w_psi.mvn.covariance  # Shape (Features, K, K)
    sigma_sqr_w = jnp.diagonal(cov_w, axis1=-1, axis2=-2)  # Shape (Features, K)
    count = model.q_w_psi.mvn.mask.sum(0)  # number of nonzero elements in each column of the loading matrix

    # Note we ignore <ln psi>_q(psi) - ln(2 pi) as this terms cancel out
    e_log_p_w_tau_psi = -0.5 * jnp.inner(exp_psi, (jnp.square(W) + sigma_sqr_w) @ exp_tau)
    e_log_p_w_tau_psi += 0.5 * jnp.sum(count * exp_log_tau)

    # E_q[log q(W|psi)] - negative entropy term - H(q(W|psi))
    # Let's approximate using standard MVN entropy: - 0.5 * log |2 pi e Cov(W)_i/psi_i|
    _, logdet_q_w_prec = safe_cholesky_and_logdet(model.q_w_psi.mvn.precision)
    e_log_q_w = -0.5 * count.sum() + 0.5 * logdet_q_w_prec.sum()

    kl_w = e_log_q_w - e_log_p_w_tau_psi  # <KL(q(W|psi) || p(W|tau, psi))>_q(psi)q(tau)

    # KL divergence for q(psi) relative to its prior p(psi)
    kl_psi = model.q_w_psi.gamma.kl_divergence_from_prior.sum()

    return kl_w + kl_psi


def _kl_tau(model: BayesianFactorAnalysisParams) -> float:
    """Compute KL(q(tau)||p(tau)).

    Args:
        model: BayesianFactorAnalysis model parameters

    Returns:
        KL divergence
    """
    return model.q_tau.kl_divergence_from_prior.sum()


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

    # Expected log likelihood E[log p(X | Z, W, psi)]
    exp_ll = _expected_log_likelihood(model, X_dist, qz)

    # KL divergences
    kl_z = _kl_latent(qz)  # KL(q(Z) || p(Z))
    kl_w_psi = _kl_loading_and_psi(model)  # <KL(q(W, psi) || p(W|tau, psi)p(psi))>_q(tau)
    kl_tau = _kl_tau(model)  # KL(q(tau) || p(tau))

    return exp_ll - kl_z - kl_w_psi - kl_tau
