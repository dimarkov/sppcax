"""Dynamic Factor Analysis parameter containers with variational posteriors."""

from typing import Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions.mvn_gamma import MultivariateNormalInverseGamma
from ..types import Array, Matrix, PRNGKey
from .factor_analysis_params import PFA, PPCA


class DynamicFactorAnalysisParams(eqx.Module):
    """Parameter container for Dynamic Factor Analysis models with variational posteriors."""

    n_samples: int
    n_timesteps: int
    likelihood: Union[PFA, PPCA]
    transition: PFA

    def __init__(
        self,
        n_samples: int,
        n_timesteps: int,
        n_components: int,
        n_features: int,
        n_controls: int = 0,
        control_latents: bool = False,
        update_ard: bool = True,
        isotropic_noise: bool = False,
        *,
        key: PRNGKey,
    ):
        """Initialize DynamicFactorAnalysis model parameters.

        Args:
            n_samples: Number of samples (N)
            n_timesteps: Number of time steps (T)
            n_components: Number of latent components (K)
            n_features: Number of observed features (D)
            n_contorls: Number of control/input variables (U)
            isotropic_noise: If True, use same noise precision R for all features
            control_latents: Apply control to latents (True) or observation (False).
            update_ard: Bool variable regyulating if we are updating the ARD prior or not
            isotropic_noise: Specify type of noise for the likelihood.
            key: A `jax.random.PRNGKey` for initialization.
        """
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps

        key_lkl, key_trns, key_cntrl = jr.split(key, 3)

        latent_control = n_controls if control_latents else 0
        t_model = PFA(
            n_components, n_components, n_controls=latent_control, use_bias=False, update_ard=update_ard, key=key_trns
        )
        self.transition = eqx.tree_at(
            lambda m: (m.q_w_psi.gamma.nat1_0, m.q_w_psi.gamma.nat2_0),
            t_model,
            (1e2 * t_model.q_w_psi.gamma.nat1_0, 1e1 * t_model.q_w_psi.gamma.nat2_0),
        )

        obs_control = n_controls if not control_latents else 0
        if isotropic_noise:
            self.likelihood = PPCA(
                n_components, n_features, n_controls=obs_control, use_bias=True, update_ard=update_ard, key=key_lkl
            )
        else:
            self.likelihood = PFA(
                n_components, n_features, n_controls=obs_control, use_bias=True, update_ard=update_ard, key=key_lkl
            )

    # --- Expected Parameter Properties ---

    @property
    def q_c_r(self) -> MultivariateNormalInverseGamma:
        return self.likelihood.q_w_psi

    @property
    def q_a_q(self) -> MultivariateNormalInverseGamma:
        return self.transition.q_w_psi

    @property
    def expected_A(self) -> Matrix:
        """E[A]"""
        return self.q_a_q.mvn.mean

    @property
    def expected_A_T(self) -> Matrix:
        """E[A^T]"""
        return self.q_a_q.mvn.mean.mT

    @property
    def expected_AAT(self) -> Array:
        """E[AA^T]"""
        EA = self.expected_A
        return self.q_a_q.mvn.covariance + EA[..., None] * EA[..., None, :]

    @property
    def expected_Q_inv(self) -> Matrix:
        """E[Q^-1]. Assumes diagonal Q."""
        return jnp.diag(self.q_a_q.expected_psi)

    @property
    def expected_log_Q_det(self) -> Array:
        """E[log|Q|]. Assumes diagonal Q."""
        # log|Q| = -log|Q^-1| = -sum(log Q_ii^-1)
        return -jnp.sum(self.q_a_q.expected_log_psi)

    @property
    def expected_A_T_Q_inv_A(self) -> Matrix:
        """E[A^T Q^-1 A]. Assumes diagonal Q."""
        Eq_inv_diag = self.q_a_q.expected_psi
        EAAT = self.expected_AAT  # Shape (k, k, k)
        return jnp.sum(Eq_inv_diag[..., None, None] * EAAT, axis=0)

    @property
    def expected_A_T_Q_inv(self) -> Matrix:
        """E[A^T Q^-1]. Assumes diagonal Q."""
        return self.expected_A_T @ self.expected_Q_inv

    @property
    def expected_C(self) -> Matrix:
        """E[C]"""
        return self.q_c_r.mvn.mean

    @property
    def expected_C_T(self) -> Matrix:
        """E[C^T]"""
        return self.q_c_r.mvn.mean.T

    @property
    def expected_CCT(self) -> Array:
        """E[CC^T]"""
        EC = self.expected_C
        return self.q_c_r.mvn.covariance + EC[..., None] * EC[..., None, :]

    @property
    def expected_R_inv(self) -> Matrix:
        """E[R^-1]. Assumes diagonal R."""
        # Handle isotropic case where expected_psi is scalar
        return jnp.diag(self.q_c_r.expected_psi)

    @property
    def expected_log_R_det(self) -> Array:
        """E[log|R|]. Assumes diagonal R."""
        return -jnp.sum(self.q_c_r.expected_log_psi)

    @property
    def expected_C_T_R_inv_C(self) -> Matrix:
        """E[C^T R^-1 C]. Assumes diagonal R."""
        Er_inv_diag = self.q_c_r.expected_psi
        ECCT = self.expected_CCT  # Shape (k, k, k)
        return jnp.sum(Er_inv_diag[..., None, None] * ECCT, axis=0)

    @property
    def expected_C_T_R_inv(self) -> Matrix:
        """E[C^T R^-1]. Assumes diagonal R."""
        return self.q_c_r.mvn.mean.T @ self.expected_R_inv  # Use expected_R_inv property


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
