import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx
import jax
from jax import lax, jit, vmap, tree, random as jr

from fastprogress.fastprogress import progress_bar

from typing import Any, Optional, Union, Tuple, NamedTuple

from functools import partial

from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM
from dynamax.utils.utils import ensure_array_has_batch_dim, pytree_stack
from dynamax.parameters import ParameterSet, ParameterProperties
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    lgssm_smoother as dynamax_lgssm_smoother,
    lgssm_posterior_sample,
)
from dynamax.utils.distributions import (
    mniw_posterior_update,
    niw_posterior_update,
    MatrixNormalInverseWishart,
    NormalInverseWishart,
)
from dynamax.linear_gaussian_ssm.models import SuffStatsLGSSM, Scalar
from dynamax.linear_gaussian_ssm.parallel_inference import (
    lgssm_posterior_sample as parallel_lgssm_posterior_sample,
    lgssm_smoother as parallel_dynamax_lgssm_smoother,
)
from dynamax.utils.bijectors import RealToPSDBijector

from sppcax.types import Array, Vector, Matrix, PRNGKey, Float
from sppcax.distributions import Distribution, Gamma, MultivariateNormal
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma, mvnig_posterior_update
from sppcax.distributions.utils import cho_inv
from sppcax.inference.utils import ParamsLGSSMVB
from sppcax.inference.smoothing import lgssm_smoother as sppcax_smoother
from sppcax.distributions.delta import Delta

from sppcax.metrics import kl_divergence
from sppcax.metrics.kl_divergence import multidigamma, digamma

from sppcax.bmr import prune_params
from sppcax.models.utils import _make_mvnig_prior, _make_mvn_prior


def _to_distribution(X):
    """Convert input to a Distribution if it isn't already."""
    if isinstance(X, Distribution):
        return X
    return Delta(X)


def _mniw_posterior_update(dist, stats, props):
    # TODO: filter stats based on props
    if not (props.weights.trainable and props.cov.trainable):
        return dist

    return mniw_posterior_update(dist, stats)


def _mvnig_posterior_update(dist, stats, props):
    if not (props.weights.trainable and props.cov.trainable):
        return dist

    return mvnig_posterior_update(dist, stats, props)


def _niw_posterior_update(dist, stats, props):
    # TODO: filter stats based on props
    if not (props.mean.trainable and props.cov.trainable):
        return dist

    return niw_posterior_update(dist, stats)


def _mvn_posterior_update(dist, stats, props):
    """Update the multivaraite normal inverse distribution using sufficient statistics
    Returns:
        posterior MVN distribution
    """

    if not props.weights.trainable:
        return dist

    nat1 = dist.nat1
    prior_precision = -2.0 * dist.nat2

    # unpack the sufficient statistics
    SxxT, SxyT, *_ = stats

    # compute parameters of the posterior distribution
    nat2_post = -0.5 * (prior_precision + SxxT)
    nat1_post = dist.apply_mask_vector(nat1 + SxyT.mT)

    mvn_post = eqx.tree_at(lambda d: (d.nat1, d.nat2), dist, (nat1_post, nat2_post))

    return mvn_post


def _posterior_update(dist, stats, props):
    if isinstance(dist, MultivariateNormalInverseGamma):
        return _mvnig_posterior_update(dist, stats, props)
    elif isinstance(dist, MatrixNormalInverseWishart):
        return _mniw_posterior_update(dist, stats, props)
    elif isinstance(dist, MultivariateNormal):
        return _mvn_posterior_update(dist, stats, props)
    else:
        raise NotImplementedError


def _get_mode(dist):
    if isinstance(dist, MultivariateNormal):
        mean = dist.mean
        covariance = jnp.eye(mean.shape[-2])
        return covariance, mean

    elif hasattr(dist, "mode"):
        return dist.mode()

    else:
        raise NotImplementedError


def _get_sample(dist, key):
    if isinstance(dist, MultivariateNormal):
        mean = dist.sample(key)
        covariance = jnp.eye(mean.shape[-2])
        return covariance, mean

    elif hasattr(dist, "sample"):
        return dist.sample(seed=key)

    else:
        raise NotImplementedError


def _get_moments(dist):
    if isinstance(dist, Union[NormalInverseWishart, MatrixNormalInverseWishart]):
        # inverse of expected precision
        covariance = jnp.einsum("...,...ij->...ij", 1 / dist.df, dist.scale)
        return covariance, dist.loc

    elif isinstance(dist, MultivariateNormalInverseGamma):
        mean = dist.mean
        # inverse of expected precision
        covariance = jnp.diag(dist.expected_psi)
        return covariance, mean

    elif isinstance(dist, MultivariateNormal):
        mean = dist.mean
        covariance = jnp.eye(mean.shape[-2])
        return covariance, mean
    else:
        raise NotImplementedError


def _get_ll_correction(dist):
    if isinstance(dist, MatrixNormalInverseWishart):
        dim, _ = dist._matrix_normal_shape
        x = dist.df / 2
        return (multidigamma(x, dim) - dim * jnp.log(x)) / 2

    elif isinstance(dist, MultivariateNormalInverseGamma):
        # inverse of expected precision
        alpha = dist.alpha
        return jnp.sum(digamma(alpha) - jnp.log(alpha)) / 2

    elif isinstance(dist, MultivariateNormal):
        return 0.0

    else:
        raise NotImplementedError


def _get_correction(dist):
    if isinstance(dist, MatrixNormalInverseWishart):
        dim, _ = dist._matrix_normal_shape
        col_precision = dist.col_precision
        return jnp.broadcast_to(cho_inv(col_precision), (dim,) + col_precision.shape)

    elif isinstance(dist, MultivariateNormalInverseGamma):
        # inverse of expected precision
        return dist.col_covariance

    elif isinstance(dist, MultivariateNormal):
        return dist.covariance

    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PX-VB rotation helpers
# ---------------------------------------------------------------------------


def _px_rotation_loss(
    R,
    emission_posterior,
    emission_prior,
    dynamics_posterior,
    dynamics_prior,
    initial_posterior,
    stats,
    state_dim,
    is_static,
):
    """Compute E_q[-ln p(H̃, F̃, x̃)] w.r.t. rotation R.

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
    psi = emission_posterior.expected_psi
    Lambda = emission_prior.mvn.precision
    mu = emission_prior.mvn.mean

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


def _compute_px_rotation_numerical(
    emission_posterior,
    emission_prior,
    dynamics_posterior,
    dynamics_prior,
    initial_posterior,
    stats,
    state_dim,
    is_static,
    n_steps=32,
    lr=1e-3,
):
    """Find PX-VB rotation R by numerically minimizing E_q[-ln p(H̃, F̃, x̃)].

    Uses gradient descent with Anderson acceleration (m=1).
    Falls back to identity if the optimization does not reduce the loss.

    Returns:
        (R, R_inv) both of shape (K, K).
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


def _rotate_mvnig(dist, R, R_inv, state_dim):
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


def _rotate_mvn_dynamics(dist, R, R_inv, state_dim):
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


def _rotate_niw(dist, R_inv):
    """Rotate NIW initial posterior: mu -> R_inv @ mu, Sigma -> R_inv @ Sigma @ R_inv^T."""
    new_loc = R_inv @ dist.loc
    new_scale = R_inv @ dist.scale @ R_inv.T
    return NormalInverseWishart(new_loc, dist.mean_concentration, dist.df, new_scale)


class ParamsLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsLGSSMInitial
    dynamics: Union[ParamsLGSSMDynamics, ParamsLGSSMVB]
    emissions: Union[ParamsLGSSMEmissions, ParamsLGSSMVB]


class ParamsBMR(NamedTuple):
    initial: bool
    dynamics: bool
    emissions: bool


class ARDDist(NamedTuple):
    emission: Gamma
    dynamics: Gamma


class BayesianDynamicFactorAnalysis(LinearGaussianConjugateSSM):
    r"""
    Bayesian Dynamic Factor Analysis based on Linear Gaussian State Space Model with conjugate
    priors for the model parameters.

    The parameters are the same as LG-SSM. The priors are as follows:

    * p(m, S) = NIW(loc, mean_concentration, df, scale) # normal inverse wishart
    * p([F, B, b], Q) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
      or MVNIG(loc, col_precision, alpha, beta) # multivariate normal inverse gamma
    * p([H, D, d], R) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
      or MVNIG(loc, col_precision, alpha, beta) # multivariate normal inverse gamma

    The MVNIG gamma supports element vise pruning of the parameters via mask, an array of boolen
    values defining free and frozen parameters. In the case MNIW distribution one can only prune
    entire columns of parameters, e.g. F_{ij} = 0 for j = 1.

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term b. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term d. Defaults to True.

    """

    def __init__(
        self,
        state_dim,
        emission_dim,
        input_dim=0,
        has_dynamics_bias=True,
        has_emissions_bias=True,
        parallel_scan=False,
        use_bmr=False,
        is_static=False,
        has_ard=True,
        isotropic_noise=False,
        use_px=True,
        **kw_priors,
    ):
        emission_prior = _make_mvnig_prior(
            emission_dim,
            state_dim,
            input_dim,
            has_bias=has_emissions_bias,
            isotropic_noise=isotropic_noise,
        )

        dynamics_prior = _make_mvn_prior(state_dim, input_dim, has_bias=has_dynamics_bias)

        kw_priors["emission_prior"] = kw_priors.pop("emission_prior", emission_prior)
        kw_priors["dynamics_prior"] = kw_priors.pop("dynamics_prior", dynamics_prior)

        super().__init__(
            state_dim,
            emission_dim,
            input_dim=input_dim,
            has_dynamics_bias=has_dynamics_bias,
            has_emissions_bias=has_emissions_bias,
            **kw_priors,
        )

        self.parallel_scan = parallel_scan
        self.is_static = is_static
        self.has_ard = has_ard
        self.isotropic_noise = isotropic_noise
        self.use_px = use_px

        if use_bmr:
            # For static models, initial distribution is fixed z~N(0,I) — no BMR needed
            self.use_bmr = ParamsBMR(initial=False, dynamics=True, emissions=True)
        else:
            self.use_bmr = ParamsBMR(initial=False, dynamics=False, emissions=False)

        # ARD priors over emission and dynamics weight columns
        if has_ard:
            self.ard_prior = ARDDist(
                emission=Gamma(alpha0=0.5 * jnp.ones(state_dim), beta0=0.5 * jnp.ones(state_dim)),
                dynamics=Gamma(alpha0=0.5 * jnp.ones(state_dim), beta0=0.5 * jnp.ones(state_dim)),
            )
        else:
            self.ard_prior = None

    def initialize(
        self,
        key: PRNGKey,
        initial_mean: Optional[Float[Array, "state_dim"]] = None,  # noqa F772
        initial_covariance: Optional[Float[Array, "state_dim state_dim"]] = None,  # noqa F772
        dynamics_weights: Optional[Float[Array, "state_dim state_dim"]] = None,  # noqa F772
        dynamics_bias: Optional[Float[Array, "state_dim"]] = None,  # noqa F772
        dynamics_input_weights: Optional[Float[Array, "state_dim input_dim"]] = None,  # noqa F772
        dynamics_covariance: Optional[Float[Array, "state_dim state_dim"]] = None,  # noqa F772
        emission_weights: Optional[Float[Array, "emission_dim state_dim"]] = None,  # noqa F772
        emission_bias: Optional[Float[Array, "emission_dim"]] = None,  # noqa F772
        emission_input_weights: Optional[Float[Array, "emission_dim input_dim"]] = None,  # noqa F772
        emission_covariance: Optional[Float[Array, "emission_dim emission_dim"]] = None,  # noqa F772
        variational_bayes: bool = False,
    ) -> Tuple[ParamsLGSSM, ParamsLGSSM]:
        r"""Initialize model parameters that are set to None, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            initial_mean: parameter $m$. Defaults to None.
            initial_covariance: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_covariance: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_covariance: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
        """

        # Arbitrary default values, for demo purposes.
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_covariance = jnp.eye(self.state_dim)

        # Static mode: F=0, b=0, Q=I (FA/PCA)
        _dynamics_bias = jnp.zeros((self.state_dim,))
        _dynamics_covariance = jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_weights = jnp.zeros((self.state_dim, self.state_dim))

        # MVNIG mean has shape (D, K+U+bias), extract H part
        key, _key = jr.split(key)
        _emission_covariance, mean = self.emission_prior.sample(_key)
        _emission_weights = mean[:, : self.state_dim]

        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,))

        # Only use the values above if the user hasn't specified their own
        def default(x, x0):
            return x0 if x is None else x

        # Create nested dictionary of params
        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean), cov=default(initial_covariance, _initial_covariance)
            ),
            dynamics=ParamsLGSSMDynamics(
                weights=default(dynamics_weights, _dynamics_weights),
                bias=default(dynamics_bias, _dynamics_bias),
                input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                cov=default(dynamics_covariance, _dynamics_covariance),
            ),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
            ),
        )

        if variational_bayes:
            dim = self.state_dim + self.input_dim + self.has_dynamics_bias
            C_dyn = jnp.zeros((self.state_dim, dim, dim))
            dim = self.state_dim + self.input_dim + self.has_emissions_bias
            C_em = jnp.zeros((self.emission_dim, dim, dim))

            dynamics = ParamsLGSSMVB(
                weights=params.dynamics.weights,
                bias=params.dynamics.bias,
                input_weights=params.dynamics.input_weights,
                cov=params.dynamics.cov,
                correction=C_dyn,
                ll=0.0,
            )
            emissions = ParamsLGSSMVB(
                weights=params.emissions.weights,
                bias=params.emissions.bias,
                input_weights=params.emissions.input_weights,
                cov=params.emissions.cov,
                correction=C_em,
                ll=0.0,
            )

            params = eqx.tree_at(lambda p: (p.dynamics, p.emissions), params, (dynamics, emissions))

        # In static mode (FA), dynamics and initial distribution are fixed by default (non-trainable)
        dynamics_trainable = not self.is_static
        initial_trainable = not self.is_static
        props = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(trainable=initial_trainable),
                cov=ParameterProperties(trainable=initial_trainable, constrainer=RealToPSDBijector()),
            ),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(trainable=dynamics_trainable),
                bias=ParameterProperties(trainable=dynamics_trainable),
                input_weights=ParameterProperties(trainable=dynamics_trainable),
                cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector()),
            ),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
            ),
        )
        return params, props

    def _update_params_from_stats(
        self,
        stats,
        props: ParamsLGSSM,
        params: ParamsLGSSM,
        m_step_state: Any,
        key: PRNGKey = None,
        extract_fn: str = "mode",
        variational_bayes: bool = False,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        enable_bmr: bool = True,
    ):
        """Unified parameter update for EM, VBEM, and Gibbs.

        Pipeline: posterior update → PX-VB rotation → BMR → KL → ARD → extract.

        Args:
            stats: sufficient statistics (summed across batches).
            props: parameter properties.
            params: current model parameters (used for non-trainable fallback).
            m_step_state: state for the M-step (ARD prior).
            key: random key for BMR/sampling.
            extract_fn: 'mode' (EM), 'moments' (VBEM), or 'sample' (Gibbs).
            variational_bayes: if True, build ParamsLGSSMVB with correction terms.
            px_n_steps: number of gradient descent steps for PX-VB rotation.
            px_lr: learning rate for PX-VB rotation.
            enable_bmr: whether to apply BMR pruning this iteration.

        Returns:
            (params, kl_div, m_step_state)
        """
        init_stats, dynamics_stats, emission_stats = stats
        kl_div = 0.0

        # --- 1. Compute posteriors ---
        if self.has_ard and isinstance(self.emission_prior, MultivariateNormalInverseGamma):
            emission_prior = self._apply_ard_to_emission_prior(self.emission_prior, m_step_state.emission)
        else:
            emission_prior = self.emission_prior

        if self.has_ard and isinstance(self.dynamics_prior, MultivariateNormal) and not self.is_static:
            dynamics_prior = self._apply_ard_to_dynamics_prior(self.dynamics_prior, m_step_state.dynamics)
        else:
            dynamics_prior = self.dynamics_prior

        initial_posterior = _niw_posterior_update(self.initial_prior, init_stats, props.initial)
        dynamics_posterior = _posterior_update(dynamics_prior, dynamics_stats, props.dynamics)
        emission_posterior = _posterior_update(emission_prior, emission_stats, props.emissions)

        # --- 2. PX-VB rotation on posteriors ---
        if self.use_px:
            R, R_inv = _compute_px_rotation_numerical(
                emission_posterior,
                emission_prior,
                dynamics_posterior,
                dynamics_prior,
                initial_posterior,
                stats,
                self.state_dim,
                self.is_static,
                n_steps=px_n_steps,
                lr=px_lr,
            )
            emission_posterior = _rotate_mvnig(emission_posterior, R, R_inv, self.state_dim)
            if not self.is_static:
                dynamics_posterior = _rotate_mvn_dynamics(
                    dynamics_posterior,
                    R,
                    R_inv,
                    self.state_dim,
                )
                initial_posterior = _rotate_niw(initial_posterior, R_inv)

        # --- 3. BMR on posteriors ---
        if self.use_bmr.initial and key is not None:
            key, _key = jr.split(key)
            initial_posterior = lax.cond(
                enable_bmr,
                lambda p: prune_params(p, self.initial_prior, key=_key),
                lambda p: p,
                initial_posterior,
            )
        if self.use_bmr.dynamics and key is not None:
            key, _key = jr.split(key)
            dynamics_posterior = lax.cond(
                enable_bmr,
                lambda p: prune_params(p, dynamics_prior, key=_key),
                lambda p: p,
                dynamics_posterior,
            )
        if self.use_bmr.emissions and key is not None:
            key, _key = jr.split(key)
            emission_posterior = lax.cond(
                enable_bmr,
                lambda p: prune_params(p, emission_prior, key=_key),
                lambda p: p,
                emission_posterior,
            )

        # --- 4. KL divergence ---
        if props.initial.mean.trainable or props.initial.cov.trainable:
            kl_div += kl_divergence(initial_posterior, self.initial_prior)
        if props.dynamics.weights.trainable or props.dynamics.cov.trainable:
            kl_div += kl_divergence(dynamics_posterior, dynamics_prior)
        kl_div += kl_divergence(emission_posterior, emission_prior)

        # --- 5. ARD updates ---
        if self.has_ard and isinstance(emission_posterior, MultivariateNormalInverseGamma):
            em_updates = self._compute_ard_updates(emission_posterior, self.state_dim)
            ard_em_post = eqx.tree_at(lambda d: (d.dnat1, d.dnat2), self.ard_prior.emission, em_updates)
            kl_div += ard_em_post.kl_divergence_from_prior.sum()

            if isinstance(dynamics_posterior, MultivariateNormal) and not self.is_static:
                dyn_updates = self._compute_dynamics_ard_updates(dynamics_posterior, self.state_dim)
                ard_dyn_post = eqx.tree_at(lambda d: (d.dnat1, d.dnat2), self.ard_prior.dynamics, dyn_updates)
                kl_div += ard_dyn_post.kl_divergence_from_prior.sum()
            else:
                ard_dyn_post = self.ard_prior.dynamics

            m_step_state = ARDDist(emission=ard_em_post, dynamics=ard_dyn_post)

        # --- 6. Extract params from posteriors ---
        # Initial
        if props.initial.mean.trainable or props.initial.cov.trainable:
            if extract_fn == "sample":
                key, _key = jr.split(key)
                S, m = _get_sample(initial_posterior, _key)
            elif extract_fn == "moments":
                S, m = _get_moments(initial_posterior)
            else:
                S, m = _get_mode(initial_posterior)
        else:
            m, S = params.initial.mean, params.initial.cov

        # Dynamics
        if props.dynamics.weights.trainable or props.dynamics.cov.trainable:
            if extract_fn == "sample":
                key, _key = jr.split(key)
                Q, FB = _get_sample(dynamics_posterior, _key)
            elif extract_fn == "moments":
                Q, FB = _get_moments(dynamics_posterior)
            else:
                Q, FB = _get_mode(dynamics_posterior)
        else:
            Q = params.dynamics.cov
            FB = jnp.column_stack([params.dynamics.weights, params.dynamics.input_weights])
            if self.has_dynamics_bias:
                FB = jnp.column_stack([FB, params.dynamics.bias[:, None]])

        # Unpack dynamics
        F = FB[:, : self.state_dim]
        B, b = (
            (FB[:, self.state_dim : -1], FB[:, -1])
            if self.has_dynamics_bias
            else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
        )

        # Emissions
        if extract_fn == "sample":
            key, _key = jr.split(key)
            R_cov, HD = _get_sample(emission_posterior, _key)
        elif extract_fn == "moments":
            R_cov, HD = _get_moments(emission_posterior)
        else:
            R_cov, HD = _get_mode(emission_posterior)

        H = HD[:, : self.state_dim]
        D_in, d_bias = (
            (HD[:, self.state_dim : -1], HD[:, -1])
            if self.has_emissions_bias
            else (HD[:, self.state_dim :], jnp.zeros(self.emission_dim))
        )

        # --- 7. Build params ---
        if variational_bayes:
            if props.dynamics.weights.trainable or props.dynamics.cov.trainable:
                C_dyn = _get_correction(dynamics_posterior)
                ll_const_dyn = _get_ll_correction(dynamics_posterior)
            else:
                C_dyn = params.dynamics.correction
                ll_const_dyn = params.dynamics.ll

            C_em = _get_correction(emission_posterior)
            ll_const_em = _get_ll_correction(emission_posterior)

            dynamics_p = ParamsLGSSMVB(
                weights=F,
                bias=b,
                input_weights=B,
                cov=Q,
                correction=C_dyn,
                ll=ll_const_dyn,
            )
            emissions_p = ParamsLGSSMVB(
                weights=H,
                bias=d_bias,
                input_weights=D_in,
                cov=R_cov,
                correction=C_em,
                ll=ll_const_em,
            )
        else:
            dynamics_p = ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q)
            emissions_p = ParamsLGSSMEmissions(weights=H, bias=d_bias, input_weights=D_in, cov=R_cov)

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=dynamics_p,
            emissions=emissions_p,
        )
        return params, kl_div, m_step_state

    def m_step(
        self,
        params: ParamsLGSSM,
        props: ParamsLGSSM,
        batch_stats: SuffStatsLGSSM,
        m_step_state: Any,
        key: PRNGKey = None,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        enable_bmr: bool = True,
    ):
        """Perform the M-step of the EM algorithm.

        Args:
            params: model parameters.
            props: parameter properties.
            batch_stats: expected sufficient statistics.
            m_step_state: state for the M-step.
            px_n_steps: number of gradient descent steps for PX-VB rotation.
            px_lr: learning rate for PX-VB rotation.
            enable_bmr: whether to apply BMR pruning this iteration.

        Returns:
            updated model parameters and updated M-step state.
        """
        stats = jtu.tree_map(partial(jnp.sum, axis=0), batch_stats)
        return self._update_params_from_stats(
            stats,
            props,
            params,
            m_step_state,
            key=key,
            extract_fn="mode",
            px_n_steps=px_n_steps,
            px_lr=px_lr,
            enable_bmr=enable_bmr,
        )

    @staticmethod
    def _compute_ard_updates(emission_posterior: MultivariateNormalInverseGamma, state_dim: int):
        """Compute ARD natural parameter updates from emission posterior.

        Returns (dnat1_tau, dnat2_tau) to be applied to the ARD prior outside JIT.
        """
        W = emission_posterior.mvn.mean  # (D, K+U+bias)
        W_emission = W[:, :state_dim]  # (D, K)
        cov_w = emission_posterior.mvn.covariance  # (D, K+U+bias, K+U+bias)
        sigma_sqr_w = jnp.diagonal(cov_w, axis1=-1, axis2=-2)[:, :state_dim]  # (D, K)
        exp_psi = emission_posterior.expected_psi  # (D,) or scalar

        mask = emission_posterior.mvn.mask[:, :state_dim]  # (D, K)
        dnat1_tau = 0.5 * mask.sum(0)
        dnat2_tau = -0.5 * jnp.sum((sigma_sqr_w + jnp.square(W_emission)) * exp_psi[..., None], 0)

        return dnat1_tau, dnat2_tau

    @staticmethod
    def _compute_dynamics_ard_updates(dynamics_posterior: MultivariateNormal, state_dim: int):
        """Compute ARD natural parameter updates from dynamics posterior (MVN).

        For MVN dynamics with fixed Q=I, there is no per-row noise scaling.

        Returns (dnat1_tau, dnat2_tau) to be applied to the dynamics ARD prior.
        """
        W = dynamics_posterior.mean[:, :state_dim]  # (K, K)
        cov_w = dynamics_posterior.covariance  # (K, dim, dim)
        sigma_sqr_w = jnp.diagonal(cov_w, axis1=-1, axis2=-2)[:, :state_dim]  # (K, K)

        mask = dynamics_posterior.mask[:, :state_dim]  # (K, K)
        dnat1_tau = 0.5 * mask.sum(0)
        dnat2_tau = -0.5 * jnp.sum(sigma_sqr_w + jnp.square(W), 0)

        return dnat1_tau, dnat2_tau

    @staticmethod
    def _apply_ard_to_emission_prior(emission_prior, m_step_state):
        """Modify emission prior precision by incorporating ARD tau from previous iteration.

        Reconstructs the ARD posterior from m_step_state, computes E[tau], and adds it
        to the diagonal of the emission prior's precision matrix (first K elements only).
        Only nat2 is modified — ARD is a zero-centered shrinkage prior.

        Returns:
            Tuple of (modified_prior, ard_dist).
        """
        ard_dist = m_step_state
        E_tau = ard_dist.mean  # (K,)
        K = len(E_tau)

        loc = emission_prior.mvn.mean
        precision = emission_prior.mvn.precision
        precision = precision.at[:, :K, :K].set(jnp.diag(E_tau))
        mask = emission_prior.mvn.mask
        mvn = MultivariateNormal(loc=loc, precision=precision, mask=mask)

        modified_prior = eqx.tree_at(lambda p: p.mvn, emission_prior, mvn)
        return modified_prior

    @staticmethod
    def _apply_ard_to_dynamics_prior(dynamics_prior, m_step_state):
        """Modify dynamics MVN prior precision by incorporating ARD tau.

        Sets the first K×K block of the per-row precision to diag(E[tau]).
        """
        E_tau = m_step_state.mean  # (K,)
        K = len(E_tau)
        precision = dynamics_prior.precision.at[:, :K, :K].set(jnp.diag(E_tau))
        return MultivariateNormal(loc=dynamics_prior.mean, precision=precision, mask=dynamics_prior.mask)

    def transform(
        self,
        params: ParamsLGSSM,
        Y: Union[Matrix, Distribution],
        U: Optional[Union[Matrix, Distribution]] = None,
    ) -> MultivariateNormal:
        """Project data to latent space (convenience method for static/FA mode).

        Args:
            params: model parameters.
            Y: observations (n_samples, emission_dim).
            U: optional inputs (n_samples, input_dim).

        Returns:
            Posterior distribution over latent factors as MultivariateNormal.
        """
        from jax.scipy.linalg import solve_triangular

        Y_dist = _to_distribution(Y)
        Ey = Y_dist.mean if hasattr(Y, "mean") else Y_dist.location

        if U is not None:
            U_dist = _to_distribution(U)
            Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
        else:
            Eu = None

        H = params.emissions.weights
        d = params.emissions.bias
        R = params.emissions.cov
        D_in = params.emissions.input_weights

        mean_offset = d
        if Eu is not None and Eu.shape[-1] > 0:
            mean_offset = mean_offset + Eu @ D_in.T

        y_centered = Ey - mean_offset

        R_inv_diag = 1.0 / jnp.diag(R) if R.ndim == 2 else 1.0 / R
        sqrt_prec = jnp.sqrt(R_inv_diag)
        scaled_H = H * sqrt_prec[:, None]
        P = scaled_H.T @ scaled_H + jnp.eye(self.state_dim)

        q, r = jnp.linalg.qr(P)
        rhs = (y_centered * R_inv_diag) @ H
        Ez = solve_triangular(r, q.T @ rhs.T).T

        return MultivariateNormal(loc=Ez, precision=P)

    def inverse_transform(
        self,
        params: ParamsLGSSM,
        Z: Union[Array, Distribution],
        U: Optional[Union[Array, Distribution]] = None,
    ) -> MultivariateNormal:
        """Project latent factors back to observation space.

        Args:
            params: model parameters.
            Z: latent factors (n_samples, state_dim) or Distribution.
            U: optional inputs (n_samples, input_dim).

        Returns:
            Reconstructed observations as MultivariateNormal.
        """
        Z_dist = _to_distribution(Z)
        Ez = Z_dist.mean if hasattr(Z, "mean") else Z_dist.location

        H = params.emissions.weights
        d = params.emissions.bias
        R = params.emissions.cov

        loc = Ez @ H.T + d
        if U is not None:
            U_dist = _to_distribution(U)
            Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
            loc = loc + Eu @ params.emissions.input_weights.T

        R_diag = jnp.diag(R) if R.ndim == 2 else R
        # Cov(y) = H Cov(z) H^T + R; handles batched Z_cov via einsum
        Z_cov = Z_dist.covariance  # (K, K) or (N, K, K)
        HZH = jnp.einsum("ij,...jk,lk->...il", H, Z_cov, H)  # (..., D, D)
        covariance = jnp.diag(R_diag) + HZH
        return MultivariateNormal(loc=loc, covariance=covariance)

    def e_step(
        self,
        params: ParamsLGSSM,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],  # noqa F772
            Float[Array, "num_batches num_timesteps emission_dim"],  # noqa F772
        ],
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],  # noqa F772
                Float[Array, "num_batches num_timesteps input_dim"],  # noqa F772
            ]
        ] = None,
    ) -> Tuple[SuffStatsLGSSM, Scalar]:
        """Compute expected sufficient statistics for the E-step of the EM algorithm.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            expected sufficient statistics and marginal log likelihood.
        """
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        if self.parallel_scan:
            posterior = parallel_dynamax_lgssm_smoother(params, emissions, inputs)
        else:
            posterior = dynamax_lgssm_smoother(params, emissions, inputs)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        # TODO: use pad instead concatenate
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[: self.state_dim, : self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT, num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[: self.state_dim, : self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

    def vbm_step(
        self,
        params: ParamsLGSSM,
        props: ParamsLGSSM,
        batch_stats: SuffStatsLGSSM,
        m_step_state: Any,
        key: PRNGKey = None,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        enable_bmr: bool = True,
    ) -> Tuple[ParamsLGSSM, Any]:
        """Perform the variational M-step of the VBEM algorithm.

        Args:
            params: model parameters.
            props: parameter properties.
            batch_stats: expected sufficient statistics.
            m_step_state: state for the M-step.
            px_n_steps: number of gradient descent steps for PX-VB rotation.
            px_lr: learning rate for PX-VB rotation.
            enable_bmr: whether to apply BMR pruning this iteration.

        Returns:
            updated model parameters and updated M-step state.
        """
        stats = jtu.tree_map(partial(jnp.sum, axis=0), batch_stats)
        return self._update_params_from_stats(
            stats,
            props,
            params,
            m_step_state,
            key=key,
            extract_fn="moments",
            variational_bayes=True,
            px_n_steps=px_n_steps,
            px_lr=px_lr,
            enable_bmr=enable_bmr,
        )

    # Variational Bayes Expectation-maximization (VBEM) code
    def vbe_step(
        self,
        params: ParamsLGSSMVB,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],  # noqa F772
            Float[Array, "num_batches num_timesteps emission_dim"],  # noqa F772
        ],
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],  # noqa F772
                Float[Array, "num_batches num_timesteps input_dim"],  # noqa F772
            ]
        ] = None,
    ) -> Tuple[SuffStatsLGSSM, Scalar]:
        """Compute expected sufficient statistics for the E-step of the EM algorithm.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            expected sufficient statistics and marginal log likelihood.
        """
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = sppcax_smoother(params, emissions, inputs, variational_bayes=True, parallel_scan=self.parallel_scan)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        # TODO: use pad instead concatenate
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[: self.state_dim, : self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT, num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[: self.state_dim, : self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

    def initialize_m_step_state(self, params, props):
        """Initialize M-step state, including ARD update placeholders."""
        state = super().initialize_m_step_state(params, props)
        if self.has_ard:
            # Initialize ARD updates as zeros with correct shape for lax.scan compatibility
            return self.ard_prior
        return state

    def fit_em(
        self,
        params: ParamsLGSSM,
        props: ParamsLGSSM,
        Y: Union[Matrix, Distribution],  # DFA expects a time series matrix
        key: PRNGKey,
        U: Union[Matrix, Distribution] = None,  # inputs/controls
        num_iters: int = 100,
        verbose: bool = False,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        bmr_start_iter: int = 4,
    ) -> Tuple[ParameterSet, Vector]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM).

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``Y`` *and* ``U`` *can either be single sequences or batches of sequences.*

        Args:
            model: Dynamax Linear Gaussian SSM
            Y: one or more sequences of emissions
            U: one or more sequences of corresponding inputs
            n_iters: number of iterations of EM to run
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.
        """

        # Convert input to distribution if needed
        Y_dist = _to_distribution(Y)
        U_dist = _to_distribution(U) if U is not None else None

        Ey = Y_dist.mean if hasattr(Y, "mean") else Y_dist.location

        # Make sure the emissions and inputs have batch dimensions
        if self.is_static:
            # Each observation is an independent batch of T=1
            batch_emissions = Ey[:, None, :]  # (N, 1, D)
            if U_dist is not None:
                Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
                batch_inputs = Eu[:, None, :] if Eu.ndim == 2 else Eu
            else:
                batch_inputs = jnp.zeros((Ey.shape[0], 1, 0))
        else:
            batch_emissions = ensure_array_has_batch_dim(Ey, self.emission_shape)
            if U_dist is not None:
                Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
                batch_inputs = ensure_array_has_batch_dim(Eu, self.inputs_shape)
            else:
                batch_inputs = None

        @jit
        def em_step(params, kl_div, m_step_state, key, enable_bmr):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            elbo = lls.sum() - kl_div
            params, kl_div, m_step_state = self.m_step(
                params,
                props,
                batch_stats,
                m_step_state,
                key=key,
                px_n_steps=px_n_steps,
                px_lr=px_lr,
                enable_bmr=enable_bmr,
            )
            return params, kl_div, elbo, m_step_state

        m_step_state = self.initialize_m_step_state(params, props)
        kl_div = 0.0
        carry = (params, kl_div, m_step_state, key)

        def step_fn(carry, iteration):
            params, kl_div, m_step_state, key = carry
            key, _key = jr.split(key)
            enable_bmr = iteration >= bmr_start_iter
            params, kl_div, elbo, m_step_state = em_step(params, kl_div, m_step_state, _key, enable_bmr)
            return (params, kl_div, m_step_state, key), elbo

        if verbose:
            pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
            elbos = []
            for i in pbar:
                carry, elbo = step_fn(carry, i)
                elbos.append(elbo)
                params = carry[0]
            elbos = jnp.stack(elbos).squeeze()
            m_step_state = carry[2]
        else:
            (params, _, m_step_state, _), elbos = lax.scan(step_fn, carry, jnp.arange(num_iters))
            elbos = elbos.squeeze()

        # Apply ARD updates outside JIT
        self.ard_prior = m_step_state

        return params, elbos[1:]

    def fit_vbem(
        self,
        params: ParamsLGSSMVB,
        props: ParamsLGSSM,
        Y: Union[Matrix, Distribution],  # DFA expects a time series matrix
        key: PRNGKey,
        U: Union[Matrix, Distribution] = None,  # inputs/controls
        num_iters: int = 100,
        verbose: bool = False,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        bmr_start_iter: int = 4,
    ) -> Tuple[ParameterSet, Vector]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM).

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``Y`` *and* ``U`` *can either be single sequences or batches of sequences.*

        Args:
            model: Dynamax Linear Gaussian SSM
            Y: one or more sequences of emissions
            U: one or more sequences of corresponding inputs
            n_iters: number of iterations of EM to run
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.
        """

        # Convert input to distribution if needed
        Y_dist = _to_distribution(Y)
        U_dist = _to_distribution(U) if U is not None else None

        Ey = Y_dist.mean if hasattr(Y, "mean") else Y_dist.location

        # Make sure the emissions and inputs have batch dimensions
        if self.is_static:
            batch_emissions = Ey[:, None, :]  # (N, 1, D)
            if U_dist is not None:
                Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
                batch_inputs = Eu[:, None, :] if Eu.ndim == 2 else Eu
            else:
                batch_inputs = jnp.zeros((Ey.shape[0], 1, 0))
        else:
            batch_emissions = ensure_array_has_batch_dim(Ey, self.emission_shape)
            if U_dist is not None:
                Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
                batch_inputs = ensure_array_has_batch_dim(Eu, self.inputs_shape)
            else:
                batch_inputs = None

        def em_step(params, kl_div, m_step_state, key, enable_bmr):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.vbe_step, params))(batch_emissions, batch_inputs)
            elbo = -kl_div + lls.sum()
            params, kl_div, m_step_state = self.vbm_step(
                params,
                props,
                batch_stats,
                m_step_state,
                key=key,
                px_n_steps=px_n_steps,
                px_lr=px_lr,
                enable_bmr=enable_bmr,
            )
            return params, kl_div, elbo, m_step_state

        m_step_state = self.initialize_m_step_state(params, props)
        kl_div = 0.0
        carry = (params, kl_div, m_step_state, key)

        def step_fn(carry, iteration):
            params, kl_div, m_step_state, key = carry
            key, _key = jr.split(key)
            enable_bmr = iteration >= bmr_start_iter
            params, kl_div, elbo, m_step_state = em_step(params, kl_div, m_step_state, _key, enable_bmr)
            return (params, kl_div, m_step_state, key), elbo

        if verbose:
            pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
            elbos = []
            for i in pbar:
                carry, elbo = jit(step_fn)(carry, i)
                elbos.append(elbo)
                params = carry[0]
            elbos = jnp.stack(elbos)
            m_step_state = carry[2]
        else:
            (params, _, m_step_state, _), elbos = lax.scan(step_fn, carry, jnp.arange(num_iters))

        # Apply ARD updates outside JIT
        self.ard_prior = m_step_state

        return params, elbos.squeeze()[1:]

    def fit_blocked_gibbs(
        self,
        key: PRNGKey,
        initial_params: ParamsLGSSM,
        props: ParamsLGSSM,
        sample_size: int,
        emissions: Float[Array, "nbatch num_timesteps emission_dim"],  # noqa: F722
        inputs: Optional[Float[Array, "nbatch num_timesteps input_dim"]] = None,  # noqa: F722
        verbose: bool = False,
        burn_in: int = 0,
        px_n_steps: int = 32,
        px_lr: float = 1e-3,
        bmr_start_iter: int = 4,
    ) -> ParamsLGSSM:
        r"""Estimate parameter posterior using block-Gibbs sampler.

        Args:
            key: random number key.
            initial_params: starting parameters.
            props: parameter properties.
            sample_size: how many samples to draw.
            emissions: set of observation sequences.
            inputs: optional set of input sequences.
            bmr_start_iter: iteration at which BMR pruning begins.

        Returns:
            parameter object, where each field has `sample_size` copies as leading batch dimension.
        """
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        num_batches, num_timesteps = batch_emissions.shape[:2]

        if batch_inputs is None:
            batch_inputs = jnp.zeros((num_batches, num_timesteps, 0))

        def sufficient_stats_from_sample(y, inputs, states):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.pad(inputs, [(0, 0), (0, 1)], constant_values=1.0)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:-1], states[1:]
            u, up = inputs_joint, inputs_joint[:-1]

            init_stats = (x[0], jnp.outer(x[0], x[0]), 1)

            # Quantities for the dynamics distribution
            # Let zp[t] = [x[t], u[t]] for t = 0...T-2
            sum_zpzpT = jnp.block([[xp.T @ xp, xp.T @ up], [up.T @ xp, up.T @ up]])
            sum_zpxnT = jnp.block([[xp.T @ xn], [up.T @ xn]])
            sum_xnxnT = xn.T @ xn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
            if not self.has_dynamics_bias:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT, num_timesteps - 1)

            # Quantities for the emissions
            # Let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[x.T @ x, x.T @ u], [u.T @ x, u.T @ u]])
            sum_zyT = jnp.block([[x.T @ y], [u.T @ y]])
            sum_yyT = y.T @ y
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
            if not self.has_emissions_bias:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

            return init_stats, dynamics_stats, emission_stats

        def lgssm_params_sample(rng, stats, m_step_state, enable_bmr):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            return self._update_params_from_stats(
                stats,
                props,
                initial_params,
                m_step_state,
                key=rng,
                extract_fn="sample",
                px_n_steps=px_n_steps,
                px_lr=px_lr,
                enable_bmr=enable_bmr,
            )

        def one_sample(_params, kl_div, m_step_state, rng, enable_bmr):
            """Sample a single set of states and compute their sufficient stats."""
            rngs = jr.split(rng, 2)
            # Sample latent states
            batch_keys = jr.split(rngs[0], num=num_batches)
            if self.parallel_scan:
                forward_backward_batched = vmap(partial(parallel_lgssm_posterior_sample, params=_params))
            else:
                forward_backward_batched = vmap(partial(lgssm_posterior_sample, params=_params))

            batch_states, batch_ll = forward_backward_batched(
                batch_keys, emissions=batch_emissions, inputs=batch_inputs
            )

            elbo = batch_ll.sum() - kl_div
            _batch_stats = vmap(sufficient_stats_from_sample)(batch_emissions, batch_inputs, batch_states)
            # Aggregate statistics from all observations.
            _stats = tree.map(lambda x: jnp.sum(x, axis=0), _batch_stats)
            # Sample parameters
            return lgssm_params_sample(rngs[1], _stats, m_step_state, enable_bmr), elbo

        def step_fn(carry, iteration):
            params, kl_div, m_step_state, key = carry
            key, _key = jr.split(key)
            enable_bmr = iteration >= bmr_start_iter
            (current_params, kl_div, m_step_state), elbo = one_sample(params, kl_div, m_step_state, _key, enable_bmr)
            return (current_params, kl_div, m_step_state, key), (params, elbo)

        kl_div = 0.0
        m_step_state = self.initialize_m_step_state(initial_params, props)
        carry = (initial_params, kl_div, m_step_state, key)
        if verbose:
            sample_of_params = []
            elbos = []
            pb = progress_bar(range(sample_size))
            for i in pb:
                carry, (current_params, elbo) = jit(step_fn)(carry, i)
                sample_of_params.append(current_params)
                elbos.append(elbo)

            sample_of_params = pytree_stack(sample_of_params[burn_in:])
            elbos = jnp.stack(elbos).squeeze()[burn_in:]
        else:
            carry, (sample_of_params, elbos) = lax.scan(step_fn, carry, jnp.arange(sample_size))
            sample_of_params = jtu.tree_map(lambda x: x[burn_in:], sample_of_params)
            elbos = elbos.squeeze()[burn_in:]

        self.ard_prior = carry[-1]

        return sample_of_params, elbos
