"""PX-VB rotation helpers for parameter-expanded variational Bayes.

These functions compute and apply rotation matrices that accelerate
convergence of variational inference by exploiting the non-identifiability
of latent factor models.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from multipledispatch import dispatch

from dynamax.utils.distributions import NormalInverseWishart

from sppcax.distributions import MultivariateNormal, MeanField
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma
from sppcax.distributions.delta import Delta
from sppcax.distributions.utils import cho_inv
from sppcax.types import Matrix, Scalar


def _px_rotation_loss(
    R,
    emission_posterior,
    emission_prior,
    dynamics_posterior,
    dynamics_prior,
    initial_posterior,
    stats,
    state_dim,
    is_static: bool,
) -> Scalar:
    r"""Compute E_q[-ln p(H̃, F̃, x̃)] w.r.t. rotation R.

    R_block = blkdiag(R, I) is applied on the right of H (and on F columns),
    R_inv applied on the left of F (rows).  Q is fixed at I.

    Returns scalar loss (summed over features / time steps, not averaged).
    """
    K = state_dim
    init_stats, dynamics_stats, _ = stats

    R_inv = jnp.linalg.inv(R)
    logdet_R = jnp.linalg.slogdet(R)[1]

    # ── Term 0: L_initial  E[-ln p(x_0|R)]  ────────────────────────────────
    sum_x0, sum_x0x0T, N_seqs = init_stats
    S_0 = initial_posterior.scale / initial_posterior.df
    m_0 = initial_posterior.loc
    S_0_inv = cho_inv(S_0)

    RinvX = R_inv @ sum_x0x0T @ R_inv.T
    L_init = 0.5 * (jnp.einsum("ij,ji->", S_0_inv, RinvX) - 2 * m_0 @ (S_0_inv @ R_inv) @ sum_x0) + N_seqs * logdet_R

    # ── Term 1: L_emission  E[-ln p(H̃)]  ──────────────────────────────────
    D, dim_em = emission_posterior.mvn.mean.shape
    R_block_em = jnp.eye(dim_em).at[:K, :K].set(R)

    m_em = emission_posterior.mvn.mean
    Sigma_em = emission_posterior.mvn.covariance
    psi = jnp.broadcast_to(emission_posterior.expected_psi, (D,))
    Lambda = emission_prior.mvn.precision
    mu = emission_prior.mvn.mean

    # For MVNIG: H|psi ~ N(m, Sigma/psi), so E[psi * HH^T] = psi * m m^T + Sigma
    # For MeanField: H ~ N(m, Sigma) independent of psi, so E[psi * HH^T] = psi * (m m^T + Sigma)
    if isinstance(emission_posterior, MeanField):
        second_moment_em = psi[:, None, None] * (m_em[:, :, None] * m_em[:, None, :] + Sigma_em)
    else:
        second_moment_em = psi[:, None, None] * m_em[:, :, None] * m_em[:, None, :] + Sigma_em
    SM_rot = jnp.einsum("ij,djk,kl->dil", R_block_em.T, second_moment_em, R_block_em)
    trace_em = jnp.einsum("dij,dji->d", Lambda, SM_rot)
    cross_em = jnp.einsum("di,dij,jk,dk->d", mu, Lambda, R_block_em.T, m_em)
    L_em = 0.5 * jnp.sum((trace_em - 2 * psi * cross_em)) - D * logdet_R

    if is_static:
        return L_em + L_init

    # ── Term 2: L_dynamics_prior  E[-ln p(F|R)]  ────────────────────────────
    K, dim_dyn = dynamics_posterior.mean.shape
    R_block_dyn = jnp.eye(dim_dyn).at[:K, :K].set(R)

    F_bar = dynamics_posterior.mean
    Sigma_F = dynamics_posterior.covariance
    mu_F = dynamics_prior.mean
    Lambda_F = dynamics_prior.precision

    second_moment_F = F_bar[:, :, None] * F_bar[:, None, :] + Sigma_F
    weighted_sm = jnp.einsum("kj,jab->kab", R_inv**2, second_moment_F)
    EF_outer = jnp.einsum("ij,kjl,lm->kim", R_block_dyn.T, weighted_sm, R_block_dyn)
    trace_F = jnp.einsum("kij,kji->k", Lambda_F, EF_outer)

    EF_mean = R_inv @ F_bar @ R_block_dyn
    cross_F = jnp.einsum("ki,kij,kj->k", mu_F, Lambda_F, EF_mean)
    L_dyn_prior = 0.5 * jnp.sum(trace_F - 2 * cross_F) + (dim_dyn - K) * logdet_R

    # ── Term 3: L_dynamics_likelihood  E[-Σ_t ln p(x̃_t | x̃_{t-1})]  ──────
    sum_zpzpT, sum_zpxnT, sum_xnxnT, T_total = dynamics_stats
    S_res = sum_xnxnT - F_bar @ sum_zpxnT - sum_zpxnT.T @ F_bar.T + F_bar @ sum_zpzpT @ F_bar.T
    diag_corr = jnp.einsum("kij,ji->k", Sigma_F, sum_zpzpT)
    S_res = S_res + jnp.diag(diag_corr)

    RRT_inv = R_inv.T @ R_inv
    L_dyn_lik = 0.5 * jnp.einsum("ij,ji->", RRT_inv, S_res) + T_total * logdet_R

    return L_em + L_dyn_prior + L_dyn_lik + L_init


def compute_px_rotation_numerical(
    emission_posterior,
    emission_prior,
    dynamics_posterior,
    dynamics_prior,
    initial_posterior,
    stats,
    state_dim,
    is_static,
    n_steps: int = 32,
    lr: float = 1e-3,
) -> Tuple[Matrix, Matrix]:
    """Find PX-VB rotation R by numerically minimizing E_q[-ln p(H̃, F̃, x̃)].

    Uses gradient descent with Anderson acceleration (m=1).
    Falls back to identity if the optimization does not reduce the loss.

    Args:
        emission_posterior: Posterior distribution over emission parameters.
        emission_prior: Prior distribution over emission parameters.
        dynamics_posterior: Posterior distribution over dynamics parameters.
        dynamics_prior: Prior distribution over dynamics parameters.
        initial_posterior: Posterior distribution over initial state.
        stats: Sufficient statistics (init_stats, dynamics_stats, emission_stats).
        state_dim: Dimension of the latent state.
        is_static: If True, skip dynamics-related terms (FA/PCA mode).
        n_steps: Number of gradient descent steps.
        lr: Learning rate.

    Returns:
        Tuple of (R, R_inv) rotation matrices, both of shape (K, K).
    """
    K = state_dim

    def loss_fn(R):
        return _px_rotation_loss(
            R,
            emission_posterior,
            emission_prior,
            dynamics_posterior,
            dynamics_prior,
            initial_posterior,
            stats,
            state_dim,
            is_static,
        )

    def step(state, i):
        R, R_prev, f_prev = state

        value, g = jax.value_and_grad(loss_fn)(R)
        R_new = R - lr * g
        f = R_new - R  # residual = -lr * grad

        # Anderson acceleration (m=1)
        delta_f = f - f_prev
        delta_x = R - R_prev
        theta = jnp.sum(f * delta_f) / (jnp.sum(delta_f**2) + 1e-8)
        R_aa = R_new - theta * (delta_x + delta_f)

        # Plain GD on first step, AA afterwards
        R_next = jnp.where(i > 0, R_aa, R_new)

        return (R_next, R, f), value

    R0 = jnp.eye(K)
    state0 = (R0, jnp.zeros((K, K)), jnp.zeros((K, K)))
    (R, _, _), values = lax.scan(step, state0, jnp.arange(n_steps))

    # Fall back to identity if optimization did not reduce the loss
    converged = values[-1] < values[0]
    R = jnp.where(converged, R, jnp.eye(K))
    R_inv = jnp.linalg.inv(R)
    return R, R_inv


# ---------------------------------------------------------------------------
# Rotation via multiple dispatch
# ---------------------------------------------------------------------------


@dispatch(MultivariateNormalInverseGamma, object, object, int)
def rotate_distribution(dist, R, R_inv, state_dim) -> MultivariateNormalInverseGamma:
    """Rotate MVNIG emission posterior: H -> H @ R.

    Transforms the MVN component's natural parameters so that the mean
    (loading matrix) is right-multiplied by R for the first K columns.
    The InverseGamma (noise precision) is unchanged.
    """
    K = state_dim
    dim = dist.mvn.nat1.shape[-1]

    # Build block rotation matrices (only first K dims)
    R_block = jnp.eye(dim).at[:K, :K].set(R)
    R_inv_block = jnp.eye(dim).at[:K, :K].set(R_inv)

    # Transform natural parameters directly:
    # nat1_new[d] = R_inv_block^T @ nat1[d] (per-row, as column vectors)
    # In row form: nat1_new = nat1 @ R_inv_block
    new_mean = dist.mvn.mean @ R_block

    # nat2_new = R_inv_block^T @ nat2 @ R_inv_block
    new_nat2 = R_inv_block @ dist.mvn.nat2 @ R_inv_block.T
    new_nat1 = jnp.squeeze((-2 * new_nat2) @ new_mean[..., None], -1)

    mvn_new = eqx.tree_at(lambda m: (m.nat1, m.nat2), dist.mvn, (new_nat1, new_nat2))
    return eqx.tree_at(lambda d: d.mvn, dist, mvn_new)


@dispatch(MultivariateNormal, object, object, int)
def rotate_distribution(dist, R, R_inv, state_dim) -> MultivariateNormal:  # noqa: F811
    """Rotate MVN dynamics posterior: F -> R_inv @ F @ R.

    The left-multiply by R_inv mixes rows, so the per-row covariance becomes
    a weighted sum of original row covariances:
        Σ_new_k = R_block^T @ (sum_j (R_inv)_{kj}^2 * Σ_j) @ R_block
    """
    K = state_dim
    dim = dist.nat1.shape[-1]

    R_block = jnp.eye(dim).at[:K, :K].set(R)
    R_inv_block = jnp.eye(dim).at[:K, :K].set(R_inv)

    # Mean: F_new = R_inv @ F @ R_block
    new_mean = R_inv @ dist.mean @ R_block

    # Covariance: weighted sum of original row covariances, then column-rotated
    # w[k,j] = (R_inv)_{kj}^2 — contribution of original row j to new row k
    w = jnp.sum(jnp.square(R_inv), -1)  # (K, K)
    new_nat2 = (R_inv_block @ dist.nat2 @ R_inv_block.T) / w[:, None, None]
    new_nat1 = jnp.squeeze((-2 * new_nat2) @ new_mean[..., None], -1)

    return eqx.tree_at(lambda d: (d.nat1, d.nat2), dist, (new_nat1, new_nat2))


@dispatch(NormalInverseWishart, object, object, int)
def rotate_distribution(dist, R, R_inv, state_dim) -> NormalInverseWishart:  # noqa: F811
    """Rotate NIW initial posterior: mu -> R_inv @ mu, Sigma -> R_inv @ Sigma @ R_inv^T."""
    new_loc = R_inv @ dist.loc
    new_scale = R_inv @ dist.scale @ R_inv.T
    return NormalInverseWishart(new_loc, dist.mean_concentration, dist.df, new_scale)


@dispatch(MultivariateNormalInverseGamma, object, object, int, str)
def rotate_distribution(dist, R, R_inv, state_dim, role) -> MultivariateNormalInverseGamma:  # noqa: F811
    """Rotate MVNIG with role parameter (role ignored, forwards to 4-arg version)."""
    return rotate_distribution(dist, R, R_inv, state_dim)


@dispatch(MultivariateNormal, object, object, int, str)
def rotate_distribution(dist, R, R_inv, state_dim, role) -> MultivariateNormal:  # noqa: F811
    """Rotate MVN with role parameter (role ignored, forwards to 4-arg version)."""
    return rotate_distribution(dist, R, R_inv, state_dim)


@dispatch(NormalInverseWishart, object, object, int, str)
def rotate_distribution(dist, R, R_inv, state_dim, role) -> NormalInverseWishart:  # noqa: F811
    """Rotate NIW with role parameter (role ignored, forwards to 4-arg version)."""
    return rotate_distribution(dist, R, R_inv, state_dim)


@dispatch(MeanField, object, object, int, str)
def rotate_distribution(dist, R, R_inv, state_dim, role) -> MeanField:  # noqa: F811
    """Rotate MeanField distribution based on its role.

    Args:
        dist: MeanField distribution.
        R: Rotation matrix (K, K).
        R_inv: Inverse rotation matrix (K, K).
        state_dim: Latent state dimension K.
        role: 'emission', 'dynamics', or 'initial'.

    Returns:
        Rotated MeanField distribution.
    """
    K = state_dim

    if role == "emission":
        # Emission weights: H -> H @ R (right multiply)
        # Rotate the MVN weights component
        if isinstance(dist.weights, Delta):
            # Frozen weights: rotate the Delta value directly
            new_mean = dist.weights.mean  # no rotation for Delta (frozen)
            new_weights = dist.weights
        else:
            dim = dist.weights.nat1.shape[-1]
            R_block = jnp.eye(dim).at[:K, :K].set(R)
            R_inv_block = jnp.eye(dim).at[:K, :K].set(R_inv)

            new_mean = dist.weights.mean @ R_block
            new_nat2 = R_inv_block @ dist.weights.nat2 @ R_inv_block.T
            new_nat1 = jnp.squeeze((-2 * new_nat2) @ new_mean[..., None], -1)

            new_weights = eqx.tree_at(lambda m: (m.nat1, m.nat2), dist.weights, (new_nat1, new_nat2))

        # Noise unchanged for emission (per-row, rows not mixed)
        return MeanField(weights=new_weights, noise=dist.noise)

    elif role == "dynamics":
        # Dynamics weights: F -> R_inv @ F @ R (left+right multiply)
        if isinstance(dist.weights, Delta):
            new_weights = dist.weights
        else:
            dim = dist.weights.nat1.shape[-1]
            R_block = jnp.eye(dim).at[:K, :K].set(R)
            R_inv_block = jnp.eye(dim).at[:K, :K].set(R_inv)

            new_mean = R_inv @ dist.weights.mean @ R_block
            w = jnp.sum(jnp.square(R_inv), -1)  # (K, K)
            new_nat2 = (R_inv_block @ dist.weights.nat2 @ R_inv_block.T) / w[:, None, None]
            new_nat1 = jnp.squeeze((-2 * new_nat2) @ new_mean[..., None], -1)

            new_weights = eqx.tree_at(lambda d: (d.nat1, d.nat2), dist.weights, (new_nat1, new_nat2))

        # Noise unchanged for dynamics (Delta(I) stays identity)
        return MeanField(weights=new_weights, noise=dist.noise)

    elif role == "initial":
        # Initial mean: m -> R_inv @ m
        if isinstance(dist.weights, Delta):
            new_weights = dist.weights
        else:
            new_mean = R_inv @ dist.weights.mean
            # For initial, precision rotates as: P_new = R @ P @ R^T
            new_nat2 = R_inv @ dist.weights.nat2 @ R_inv.T
            new_nat1 = jnp.squeeze((-2 * new_nat2) @ new_mean[..., None], -1)
            new_weights = eqx.tree_at(lambda d: (d.nat1, d.nat2), dist.weights, (new_nat1, new_nat2))

        # Noise (covariance): S -> R_inv @ S @ R_inv^T
        if isinstance(dist.noise, Delta):
            # Rotate Delta value if it's a matrix (covariance)
            val = dist.noise.mean
            if val.ndim >= 2:
                new_val = R_inv @ val @ R_inv.T
                new_noise = Delta(new_val)
            else:
                new_noise = dist.noise
        else:
            # Generic rotation for noise component (e.g., InverseWishart)
            new_noise = dist.noise  # TODO: implement IW rotation if needed

        return MeanField(weights=new_weights, noise=new_noise)

    else:
        raise ValueError(f"Unknown role: {role}. Must be 'emission', 'dynamics', or 'initial'.")
