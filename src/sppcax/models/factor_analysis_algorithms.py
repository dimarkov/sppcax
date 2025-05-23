"""Bayesian Factor Analysis algorithm implementations."""

from typing import List, Tuple, Union, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import random as jr
from jax.scipy.linalg import solve_triangular
from multimethod import multimethod
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma
from sppcax.distributions.utils import safe_cholesky_and_logdet


import jax
from jax import lax
from jaxlib.xla_extension import ArrayImpl

from sppcax.bmr import reduce_model
from sppcax.distributions import Delta, Distribution, MultivariateNormal
from sppcax.types import Array, Matrix, PRNGKey
from .factor_analysis_params import BayesianFactorAnalysisParams, PPCA, PFA
from .dynamic_factor_analysis_params import DynamicFactorAnalysisParams


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
    X: Distribution,
    U: Optional[Distribution] = None,
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
    X_mean = X.mean if hasattr(X, "mean") else X.location
    mask = model._validate_mask(X_mean) if use_data_mask else jnp.ones_like(X_mean, dtype=bool)
    mean, _ = get_mean(model, U)

    X_centered = X_mean - mean

    # Get current loading matrix and noise precision expectations
    W = model.q_w_psi.mvn.mean.mT  # (n_components, n_features)
    sqrt_noise_precision = jnp.where(mask, jnp.sqrt(model.q_w_psi.expected_psi), 0.0)

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


@eqx.filter_jit
def _update_params(
    model: Union[PFA, PPCA], Exx: Array, Exz: Array, Ezz: Matrix, N: Array, BcovuB: Array
) -> MultivariateNormalInverseGamma:
    """Update the parameters of (q(W, psi)q(tau)).

    Args:
        model: The PFA or PPCA model to update.
        Exx: Sum of expectations over an outer product of observations (D, D) centered by the posterior expectation
            over mean parameter <mu>.
        Exz: Sum of expected outer products of latents and centerd observations masked by obsmask (D, K)
        Ezz: Sum of expected outer products of latent states (K, K).
        N: Effective number of observations for each feature dimension D.

    Returns:
        Updated PFA, or PPCA model.
    """
    D, K = Exz.shape

    # --- Update MVN part q(W | psi) ---
    exp_tau = model.q_tau.mean
    P_w = jnp.diag(exp_tau) + Ezz
    nat1_w = jnp.where(model.q_w_psi.mvn.mask, Exz, 0.0)
    updated_mvn = eqx.tree_at(lambda x: (x.nat1, x.nat2), model.q_w_psi.mvn, (nat1_w, -0.5 * P_w))
    W = updated_mvn.mean  # E[W]
    cov_w = updated_mvn.covariance  # Cov(W) - shape (D, K, K)
    sigma_sqr_w = jnp.diagonal(cov_w, axis1=-1, axis2=-2)  # diag(Cov(W)) - shape (D, K)

    # --- Update Gamma part q(psi) ---
    dnat1_psi = 0.5 * (N + updated_mvn.mask.sum(-1))

    # term1 sum_n diag( E[(x_n - m) (x_n - m)^T] )
    term1_psi = jnp.diag(Exx)
    # term2: sum_n diag( E[(x_n - m) z_n^T] E[W^T] )
    term2_psi = -2 * jnp.diag(Exz @ W.mT)
    # term3: sum_n diag( E[W z_n z_n^T W^T] )
    term3_psi = jnp.diag(W @ Ezz @ W.mT)
    # term4: psoterior expectation of the prior E[tau_k w_dk^2]
    term4_psi = jnp.square(W) @ exp_tau

    dnat2_psi = -0.5 * (term1_psi + term2_psi + term3_psi + term4_psi)
    if BcovuB is not None:
        dnat2_psi -= 0.5 * BcovuB

    # Handle isotropic noise
    if isinstance(model, PPCA):
        dnat1_psi = jnp.sum(dnat1_psi)
        dnat2_psi = jnp.sum(dnat2_psi)

    # Update Gamma distribution
    updated_q_psi = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.q_w_psi.gamma, (dnat1_psi, dnat2_psi))

    # --- Update ARD prior precision q(tau) ---
    if model.update_ard:
        exp_psi = updated_q_psi.mean  # Use updated E[psi]

        dnat1_tau = 0.5 * updated_mvn.mask.sum(0)
        dnat2_tau = -0.5 * jnp.sum((sigma_sqr_w + jnp.square(W)) * exp_psi[..., None], 0)
        updated_q_tau = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), model.q_tau, (dnat1_tau, dnat2_tau))
    else:
        updated_q_tau = model.q_tau

    # Combine updated MVN and Gamma parts
    return eqx.tree_at(
        lambda x: (x.q_w_psi.mvn, x.q_w_psi.gamma, x.q_tau), model, (updated_mvn, updated_q_psi, updated_q_tau)
    )


@eqx.filter_jit
def _update_control(
    model: Union[PFA, PPCA],
    Exu: Array,
    Euu: Matrix,
) -> MultivariateNormalInverseGamma:
    """Update the parameters of (q(W, psi)q(tau)).

    Args:
        model: The PFA or PPCA model to update.
        Exu: Sum of expectations over an outer product of observations and controls (D, U + 1).
        Euu: Sum of expected outer products of control states (U, U).
        N: Effective number of observations for each feature dimension D.

    Returns:
        Updated PFA, or PPCA model.
    """
    _, U = Exu.shape

    # --- Update MVN part q(W | psi) ---
    nat2_B = -0.5 * (model.control.prior_prec * jnp.eye(U) + Euu)
    nat1_B = Exu
    updated_q_b = eqx.tree_at(lambda x: (x.nat1, x.nat2), model.control.q_b, (nat1_B, nat2_B))

    # Combine updated MVN and Gamma parts
    return eqx.tree_at(lambda m: m.control.q_b, model, updated_q_b)


def m_step(
    model: BayesianFactorAnalysisParams,
    X: Distribution,
    qz: MultivariateNormal,
    U: Optional[Distribution] = None,
    use_bmr_opt: bool = False,
) -> BayesianFactorAnalysisParams:
    """VBM-step: Update parameters for Bayesian Factor Analysis.

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix of shape (n_samples, n_features) or Distribution instance
        qz: Posterior estimates of latent states obtained during the variational e-step (n_samples, n_components)
        use_bmr: Whether to apply Bayesian Model Reduction based optimisation for q(tau) and q(psi)

    Returns:
        Updated model instance
    """
    Ex = X.mean if hasattr(X, "mean") else X.location
    mask = model._validate_mask(Ex)
    N = mask.sum(0)

    exp_stats_z = qz.expected_sufficient_statistics
    Ez = exp_stats_z[..., : model.n_components]  # Shape (N, K)

    mean, B, Eu, Covu = get_mean(model, U, return_u_stat=True)

    if B is None:
        updated_model = model
        BcovuB = None
    else:
        Eu = jnp.broadcast_to(Eu, Ex.shape[:-1] + (Eu.shape[-1],))
        Euu = jnp.sum(Covu + Eu[..., None] * Eu[..., None, :], 0)
        Ex_centered = jnp.where(mask, Ex - Ez @ model.q_w_psi.mvn.mean.mT, 0.0)
        Exu = jnp.sum(Ex_centered[..., None] * Eu[..., None, :], 0)

        updated_model = _update_control(model=model, Exu=Exu, Euu=Euu)
        BcovuB = jnp.sum((B @ Covu.sum(0)) * B, -1)

    X_centered = jnp.where(mask, Ex - mean, 0.0)
    # Compute sum_n E[(x_n - m) (x_n - m)^T]
    Exx_centered = X.covariance + X_centered[..., None] * X_centered[..., None, :]
    # mask with missing observations and sum over data points n
    Exx_centered = jnp.where(mask[..., None] * mask[..., None, :], Exx_centered, 0.0).sum(0)

    # Compute sum_n E[(x_n - m) z_n^T]
    Exz_centered = jnp.sum(X_centered[..., None] * Ez[..., None, :], 0)  # Shape (D, K)

    # Compute sum_n E[z_n z_n^T]
    Ezz = jnp.sum(exp_stats_z[..., model.n_components :], 0).reshape(
        model.n_components, model.n_components
    )  # Shape (K, K)

    # --- Update PFA or PPCA model using helper function ---
    updated_model = _update_params(model=model, Exx=Exx_centered, Exz=Exz_centered, Ezz=Ezz, N=N, BcovuB=BcovuB)

    # --- Apply BMR if enabled ---
    if use_bmr_opt:
        # BMR optimisation of gamma prior hyper-parameters
        # Update q_tau prior rate (beta0)
        updated_q_tau = updated_model.q_tau
        nat2_0_tau = updated_q_tau.dnat2 * (updated_q_tau.nat1_0 + 1)
        nat2_0_tau = jnp.where(updated_q_tau.dnat1 > 0, nat2_0_tau / updated_q_tau.dnat1, -1e-4)
        updated_q_tau = eqx.tree_at(lambda x: x.nat2_0, updated_q_tau, nat2_0_tau)

        # Update q_psi prior rate (beta0) - Use updated_q_w_psi.gamma
        updated_q_psi = updated_model.q_w_psi.gamma
        nat2_0_psi = updated_q_psi.dnat2 * (updated_q_psi.nat1_0 + 1) / updated_q_psi.dnat1
        updated_q_psi = eqx.tree_at(lambda x: x.nat2_0, updated_q_psi, nat2_0_psi)

        # --- Update model ---
        updated_model = eqx.tree_at(
            lambda x: (x.q_w_psi.gamma, x.q_tau),
            updated_model,
            (updated_q_psi, updated_q_tau),
        )

    return updated_model


# --- Dynamic Factor Analysis Specific Functions ---


def _kalman_filter_scan_body(
    state_prev: Tuple[Array, Matrix], y_t: Array, mu_t: Array, nu_t: Array, params: Tuple
) -> Tuple[Tuple[Array, Matrix], Tuple[Array, Matrix, Array, Matrix, Array]]:
    """Associative scan function for Kalman filter forward pass."""
    m_prev, P_prev = state_prev
    EA, EQ, EC, ER = params

    y = y_t - mu_t

    # Prediction
    m_pred = EA @ m_prev if nu_t is None else EA @ m_prev + nu_t
    P_pred = EA @ P_prev @ EA.mT + EQ

    # Update
    y_pred = EC @ m_pred
    S = EC @ P_pred @ EC.mT + ER  # Innovation covariance S = C P_pred C^T + R
    S_chol = jnp.linalg.cholesky(S)
    K = jax.scipy.linalg.cho_solve((S_chol, True), EC @ P_pred).mT  # Kalman gain K = P_pred C^T S^-1

    m_filt = m_pred + K @ (y - y_pred)
    P_filt = P_pred - K @ S @ K.mT

    # Log likelihood contribution for this time step
    elbo_t = MultivariateNormal(loc=y_pred, covariance=S).log_prob(y)

    state_curr = (m_filt, P_filt)
    scan_output = (m_filt, P_filt, m_pred, P_pred, elbo_t)
    return state_curr, scan_output


def _rts_smoother_scan_body(
    state_next_smoothed: Tuple[Array, Matrix, Matrix],
    state_current_filtered: Tuple[Array, Matrix, Array, Matrix],
    EA: Matrix,
) -> Tuple[Tuple[Array, Matrix, Matrix], Tuple[Array, Matrix, Matrix]]:
    """Associative scan function for RTS smoother backward pass."""
    sm_next, sP_next = state_next_smoothed
    m_filt, P_filt, m_pred, P_pred = state_current_filtered

    # Smoother gain G = P_filt EA^T P_pred^-1
    P_pred_chol = jnp.linalg.cholesky(P_pred)
    G = jax.scipy.linalg.cho_solve((P_pred_chol, True), EA @ P_filt).mT  # G = P_filt A^T P_pred^-1

    # Smoothed state
    sm_curr = m_filt + G @ (sm_next - m_pred)
    sP_curr = P_filt + G @ (sP_next - P_pred) @ G.mT

    state_curr_smoothed = (sm_curr, sP_curr)
    scan_output = (sm_curr, sP_curr, G)  # Store G for E[z_t z_{t-1}^T | Y] calculation later
    return state_curr_smoothed, scan_output


def kalman_smoother_estep(
    model: DynamicFactorAnalysisParams, Y: Union[Array, Distribution], U: Union[Array, Distribution] = None
) -> Tuple[Array, Array, Array, Array]:
    """E-step for Dynamic Factor Analysis using Kalman Filter/Smoother.

    Args:
        model: DynamicFactorAnalysisParams instance.
        Y: Observations, shape (T, d).

    Returns:
        Ez: Smoothed mean E[z_t | Y], shape (T, k).
        Ezz: Smoothed second moment E[z_t z_t^T | Y], shape (T, k, k).
        Ezz_lag: Smoothed lagged second moment E[z_t z_{t-1}^T | Y], shape (T-1, k, k).
        total_log_likelihood: Marginal log likelihood log p(Y).
    """
    T, D = Y.shape
    K = model.likelihood.n_components

    Y = _to_distribution(Y)
    Ey = Y.mean if hasattr(Y, "mean") else Y.location

    # Get expected parameters
    EA = model.expected_A
    EQinv = model.expected_Q_inv
    EQ = jnp.linalg.inv(EQinv)  # Need E[Q] for filter prediction step
    EC = model.expected_C
    ERinv = model.expected_R_inv

    if U is None:
        Emu, _ = get_mean(model.likelihood, U)
        Emu = jnp.broadcast_to(Emu, Ey.shape)
        Enu = None
    else:
        if model.transition.control is None:
            Emu, _ = get_mean(model.likelihood, U)
            Emu = jnp.broadcast_to(Emu, Ey.shape)
            Enu = None
        else:
            Emu, _ = get_mean(model.likelihood, None)
            Emu = jnp.broadcast_to(Emu, Ey.shape)
            Enu, _ = get_mean(model.transition, U)

    # initial expectation and covariance
    Emu0 = jnp.zeros(K)
    ESigma0 = jnp.eye(K)

    # --- Kalman Filter (Forward Pass) ---
    initial_state_filter = (Emu0, ESigma0)
    filter_params = (EA, EQ, EC, ERinv)

    def filter_scan_fn(state_prev, inputs):
        y_t, mu_t, nu_t = inputs
        return _kalman_filter_scan_body(state_prev, y_t, mu_t, nu_t, filter_params)

    last_state_filter, filtered_results = lax.scan(filter_scan_fn, initial_state_filter, (Ey, Emu, Enu), unroll=2)
    m_filt_all, P_filt_all, m_pred_all, P_pred_all, elbo_all = filtered_results
    total_elbo = jnp.sum(elbo_all)

    # --- RTS Smoother (Backward Pass) ---
    # Initial state for smoother is the last filtered state
    initial_state_smoother = (last_state_filter[0], last_state_filter[1])

    # We need to scan over the filtered states (m_filt, P_filt, m_pred, P_pred)
    # Combine them for the scan input, reversing time order
    smoother_scan_inputs = (m_filt_all[:-1], P_filt_all[:-1], m_pred_all[:-1], P_pred_all[:-1])

    def smoother_scan_fn(state_next_smoothed, state_current_filtered_rev):
        # Unpack reversed current filtered state
        m_filt_rev, P_filt_rev, m_pred_rev, P_pred_rev = state_current_filtered_rev
        # Pass the non-reversed EA
        return _rts_smoother_scan_body(state_next_smoothed, (m_filt_rev, P_filt_rev, m_pred_rev, P_pred_rev), EA)

    # Run the scan backwards (over reversed filtered states, excluding the last time step T)
    _, smoothed_results_rev = lax.scan(
        smoother_scan_fn, initial_state_smoother, smoother_scan_inputs, reverse=True, unroll=2
    )

    sm, sP, G = smoothed_results_rev
    sm = jnp.concatenate([sm, jnp.expand_dims(last_state_filter[0], 0)], axis=0)
    sP = jnp.concatenate([sP, jnp.expand_dims(last_state_filter[1], 0)], axis=0)

    # --- Calculate Sufficient Statistics ---
    Ez = sm
    Ezz = sP + sm[..., None] * sm[..., None, :]  # E[zzT] = Cov[z] + E[z]E[z]T

    # Calculate E[z_t z_{t-1}^T | Y] = Cov[z_t, z_{t-1} | Y] + sm_t sm_{t-1}^T
    # Cov[z_t, z_{t-1} | Y] = sP_t G_{t-1}^T (Bishop 13.108)
    # Need G_{t-1} which is just G array
    Ezz_lag = sP[1:] @ G.mT + sm[1:, :, None] * sm[:-1, None, :]

    return Ez, Ezz, Ezz_lag, total_elbo


def dfa_mstep(
    model,
    X: Array,
    Ez: Array,
    Ezz: Array,
    Ezz_lag: Array,
    U: Distribution = None,
    data_mask: Optional[Matrix] = None,
):
    """M-step for Dynamic Factor Analysis: Update variational parameter posteriors.

    Args:
        model: Current DynamicFactorAnalysisParams instance.
        Y: Observations, shape (T, d).
        Ez: Smoothed mean E[z_t | Y], shape (T, k).
        Ezz: Smoothed second moment E[z_t z_t^T | Y], shape (T, k, k).
        Ezz_lag: Smoothed lagged second moment E[z_t z_{t-1}^T | Y], shape (T-1, k, k).

    Returns:
        Updated DynamicFactorAnalysisParams instance.
    """
    X_dist = _to_distribution(X)
    Ex = X_dist.mean if hasattr(X_dist, "mean") else X_dist.location

    datamask = model.likelihood._validate_mask(Ex)
    axis = 0 if datamask.ndim == 2 else datamask.sum((0, 1))
    N = datamask.sum(axis)

    U_lkl = U if model.transition.control is None else None
    mean, B, Eu, Covu = get_mean(model.likelihood, U_lkl, return_u_stat=True)

    if B is None:
        updated_likelihood = model.likelihood
        BcovuB = None
    else:
        Eu = jnp.broadcast_to(Eu, Ex.shape[:-1] + (Eu.shape[-1],))
        Euu = jnp.sum(Covu + Eu[..., None] * Eu[..., None, :], 0)
        Ex_centered = jnp.where(datamask, Ex - Ez @ model.expected_C_T, 0.0)
        Exu = jnp.sum(Ex_centered[..., None] * Eu[..., None, :], 0)

        updated_likelihood = _update_control(model=model.likelihood, Exu=Exu, Euu=Euu)
        mean, B = get_mean(updated_likelihood, U_lkl)
        BcovuB = jnp.sum((B @ Covu.sum(0)) * B, -1)

    X_centered = jnp.where(datamask, Ex - mean, 0.0)
    # Compute sum_n E[(x_n - m) (x_n - m)^T]
    Exx_centered = X_dist.covariance + X_centered[..., None] * X_centered[..., None, :]
    # mask with missing observations and sum over data points n
    Exx_centered = jnp.where(datamask[..., None] * datamask[..., None, :], Exx_centered, 0.0).sum(axis)

    # Compute sum_n E[(x_n - m) z_n^T]
    Exz_centered = jnp.sum(X_centered[..., None] * Ez[..., None, :], axis)  # Shape (D, K)

    # --- Sufficient Statistics Sums ---
    Ezz_sum = jnp.sum(Ezz, axis=axis)
    Ezz_lag = jnp.sum(Ezz_lag, axis=axis)  # Sum over t=1..T-1

    # --- Update q(C, R) ---
    updated_likelihood = _update_params(
        model=updated_likelihood,
        Exx=Exx_centered,  # Observations are Y
        Exz=Exz_centered,  # Latent states are Z
        Ezz=Ezz_sum,  # Sum E[z_t z_t^T] over timesteps
        N=N,
        BcovuB=BcovuB,
    )

    *shape, K = Ez.shape
    Ezz_tp1 = Ezz_sum - jnp.take(Ezz, 0, axis=-3)
    Ezz_t = Ezz_sum - jnp.take(Ezz, -1, axis=-3)
    if model.transition.control is None:
        updated_transition = model.transition
        HcovuH = None
    else:
        nu, H, Eu, Covu = get_mean(model.transition, U, return_u_stat=True)
        Eu = jnp.broadcast_to(Eu, Ez.shape[:-1] + (Eu.shape[-1],))[1:]
        Euu = jnp.sum(Covu[1:] + Eu[..., None] * Eu[..., None, :], 0)
        Eztp1 = Ez[1:] - Ez[:-1] @ model.expected_A_T
        Eztp1u = jnp.sum(Eztp1[..., None] * Eu[..., None, :], 0)

        updated_transition = _update_control(model=model.transition, Exu=Eztp1u, Euu=Euu)
        nu, H = get_mean(updated_transition, U)
        nu = nu[1:]
        HcovuH = jnp.sum((H @ Covu.sum(0)) * H, -1)

        Ezz_tp1 -= jnp.sum(2 * Ez[1:, :, None] * nu[..., None, :] - nu[..., None] * nu[..., None, :], 0)
        Ezz_lag -= jnp.sum(nu[..., None] * Ez[:-1, None, :], 0)

    update_transition = _update_params(
        model=updated_transition,
        Exx=Ezz_tp1,
        Exz=Ezz_lag,  # Latent states are z_t values
        Ezz=Ezz_t,
        N=jnp.full(K, jnp.prod(jnp.asarray(shape))),
        BcovuB=HcovuH,
    )

    # Keep prior fixed for initial state? Or update? Let's update.
    updated_model = eqx.tree_at(lambda m: (m.likelihood, m.transition), model, (updated_likelihood, update_transition))

    return updated_model


@multimethod
def fit(  # noqa: F811
    model: Union[PPCA, PFA],
    X: Union[ArrayImpl, Array, Matrix, Distribution],  # DFA expects a time series matrix
    key: Union[ArrayImpl, PRNGKey],
    U: Array = None,  # inputs/controls
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
    U_dist = _to_distribution(U) if U is not None else None

    # EM algorithm
    old_elbo = -jnp.inf
    elbos = []
    for n in range(n_iter):
        # E-step
        key, _key = jr.split(key)
        use_bmr = model.bmr_e_step.use and (n > 32)
        qz = eqx.filter_jit(e_step)(model, X_dist, U=U_dist, use_bmr=use_bmr, key=_key)
        elbo_val = eqx.filter_jit(compute_elbo)(model, X_dist, U_dist, qz)
        elbos.append(elbo_val)

        # M-step (returns updated model)
        # updated_model = eqx.filter_jit(m_step)(updated_model, X_dist, qz, use_bmr=updated_model.optimize_with_bmr)
        model = m_step(model, X_dist, qz, U=U_dist, use_bmr_opt=model.optimize_with_bmr)

        # Apply Bayesian Model Reduction if enabled
        if model.bmr_m_step.use and ((n + 1) % bmr_frequency == 0):
            key, _key = jr.split(key)
            model = eqx.filter_jit(reduce_model)(model, key=_key, **model.bmr_m_step.opts)

        # Check convergence
        if jnp.abs(elbo_val - old_elbo) < tol:
            break
        old_elbo = elbo_val

    return model, elbos


def transform(
    model: BayesianFactorAnalysisParams,
    X: Union[Matrix, Distribution],
    U: Union[Matrix, Distribution] = None,
    use_data_mask: bool = False,
    use_bmr: bool = False,
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

    X_dist = _to_distribution(X)
    U_dist = _to_distribution(U) if U is not None else None
    return e_step(model, X_dist, U=U_dist, use_data_mask=use_data_mask, use_bmr=use_bmr, key=key)


def get_mean(model: Union[PFA, PPCA], U: Union[Distribution, Matrix], return_u_stat: bool = False) -> Array:
    if U is None:
        if model.control is None:
            if return_u_stat:
                return jnp.zeros(()), None, None, None
            else:
                return jnp.zeros(()), None
        else:
            B = model.control.q_b.mean
            if return_u_stat:
                return B.squeeze(-1), B, jnp.ones(1), jnp.zeros((1, 1))
            else:
                return B.squeeze(-1), B
    else:
        U = _to_distribution(U)
        U_mean = U.mean if hasattr(U, "mean") else U.location
        B = model.control.q_b.mean
        pad = B.shape[-1] - U_mean.shape[-1]
        if return_u_stat:
            Covu = U.covariance
            Covu = jnp.pad(Covu, [(0, 0), (0, pad), (0, pad)])
            Eu = jnp.pad(U_mean, [(0, 0), (0, pad)], constant_values=1.0)
            return jnp.pad(U_mean, [(0, 0), (0, pad)], constant_values=1.0) @ B.mT, B, Eu, Covu
        else:
            return jnp.pad(U_mean, [(0, 0), (0, pad)], constant_values=1.0) @ B.mT, B


def inverse_transform(
    model: BayesianFactorAnalysisParams, Z: Union[Array, Distribution], U: Union[Array, Distribution] = None
) -> MultivariateNormal:
    """Transform latent states back to its original space.

    Args:
        model: BayesianFactorAnalysis model parameters
        Z: Data in transformed space of shape (n_samples, n_components) or a Distribution

    Returns:
        X_original: Prediction of the data as a MultivariateNormal distribution
    """
    Z_dist = _to_distribution(Z)
    W = model.q_w_psi.mvn.mean.mT
    mean, _ = get_mean(model, U)
    loc = Z_dist.mean @ W + mean
    # Use expected noise precision E[psi] = alpha/beta
    exp_noise_precision = model.q_w_psi.expected_psi
    covariance = jnp.diag(1 / exp_noise_precision) + W.mT @ Z_dist.covariance @ W
    return MultivariateNormal(loc=loc, covariance=covariance)


def _expected_log_likelihood(
    model: BayesianFactorAnalysisParams, X_dist: Distribution, U_dist: Distribution, qz: MultivariateNormal
) -> float:
    """Compute expected log likelihood E_q[log p(X|Z,W, psi)].

    Args:
        model: BayesianFactorAnalysis model parameters
        X_dist: Distribution over observations
        U_dist: Distribution over control variables
        qz: Posterior distribution over latent variables

    Returns:
        Expected log likelihood
    """
    # TODO: Add the corrections for the terms that depend on the uncertainty of U and B
    # Get parameters and expectations
    W = model.q_w_psi.mvn.mean
    exp_stats_psi = model.q_w_psi.expected_sufficient_statistics_psi
    exp_log_psi = exp_stats_psi[..., 0]  # E[log psi]
    exp_psi = exp_stats_psi[..., 1]  # E[psi]

    # Get expected sufficient statistics
    dim = X_dist.event_shape[0]
    exp_stats = X_dist.expected_sufficient_statistics
    m, _ = get_mean(model, U_dist)
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
    term1 = jnp.sum(exp_psi * mask * E_xx_centered)  # E[(x-m)^T psi (x-m)] related terms
    term2 = -2 * jnp.sum(
        exp_psi * mask * (E_x_centered * (W @ Ez[..., None]).squeeze(-1))
    )  # E[(x-m)^T psi W z] related terms
    # E[z^T W^T psi W z] related terms, including uncertainty in W
    term3 = jnp.trace((exp_psi * mask)[..., None] * (W @ Ezz @ W.T), axis1=-1, axis2=-2).sum()

    # Term accounting for covariance of W, Tr(E[psi] * E[Z Z^T] * Cov[W])
    cov_w = model.q_w_psi.mvn.covariance
    term4 = jnp.trace(jnp.expand_dims(mask, (-1, -2)) * (cov_w @ jnp.expand_dims(Ezz, -3)), axis1=-1, axis2=-2).sum()

    exp_ll -= 0.5 * (term1 + term2 + term3 + term4)

    # TODO add a contribution of E[Bu_t u_t^T B^T] - <B> <u_t> <u_t> <B^T>

    return exp_ll


def _kl_latent(qz: MultivariateNormal) -> float:
    """Compute KL(q(Z)||p(Z)).

    Args:
        qz: Posterior distribution over latent variables

    Returns:
        KL divergence
    """
    k = qz.shape[-1]
    mu = qz.mean
    cov = qz.covariance
    term1 = jnp.diagonal(cov, axis1=-1, axis2=-2).sum(-1) + jnp.square(mu).sum(-1) - k
    precision = qz.precision
    _, logdet = safe_cholesky_and_logdet(precision)
    kl_div = 0.5 * (term1 + logdet)
    return kl_div.sum()


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


def compute_elbo(
    model: BayesianFactorAnalysisParams, X: Distribution, U: Distribution, qz: MultivariateNormal
) -> float:
    """Compute Evidence Lower Bound (ELBO).

    Args:
        model: BayesianFactorAnalysis model parameters
        X: Data matrix or distribution over observations
        qz: Posterior distribution over latent variables

    Returns:
        ELBO value
    """
    # Expected log likelihood E[log p(X | Z, W, psi)]
    exp_ll = _expected_log_likelihood(model, X, U, qz)

    # KL divergences
    kl_z = _kl_latent(qz)  # KL(q(Z) || p(Z))
    kl_w_psi = _kl_loading_and_psi(model)  # <KL(q(W, psi) || p(W|tau, psi)p(psi))>_q(tau)
    kl_tau = _kl_tau(model)  # KL(q(tau) || p(tau))

    return exp_ll - kl_z - kl_w_psi - kl_tau
