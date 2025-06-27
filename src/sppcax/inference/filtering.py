"""
This module contains functions for filtering in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp

from jax import lax
from jaxtyping import Array, Float
from typing import Optional

from jax.scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiagPlusLowRankCovariance as MVNLowRank,
    MultivariateNormalFullCovariance as MVN,
)

from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, _predict, _condition_on
from .utils import _get_params, ParamsLGSSMVB, preprocess_args


def _slogdet_lu(lu: Array, pivot: Array) -> tuple[Array, Array]:
    dtype = lax.dtype(lu)
    diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
    is_zero = jnp.any(diag == jnp.array(0, dtype=dtype), axis=-1)
    logdet = jnp.where(
        is_zero, jnp.array(-jnp.inf, dtype=dtype), jnp.sum(jnp.log(jnp.abs(diag)).astype(dtype), axis=-1)
    )
    return logdet


def correct_for_vb(mean, cov, u, C, dim) -> tuple[Array, Array]:
    _C = jnp.eye(dim) + cov @ C[:dim, :dim]
    lup = lu_factor(_C)

    # terms needed for log-likelihood correction
    cho_l = cho_factor(cov)

    # natural parameter
    eta = cho_solve(cho_l, mean)

    # correcting mean and covariancezs
    pred_cov = lu_solve(lup, cov)
    f = jnp.pad(u, ([0, 1]), constant_values=1.0)
    u_contrib = C[:dim, dim:] @ f
    pred_mean = lu_solve(lup, mean - cov @ u_contrib)

    # correction of log-likelihood due to variational expectation of model parameters
    ll_corr = 0.5 * jnp.inner(pred_mean - mean, eta)
    ll_corr -= 0.5 * jnp.inner(pred_mean, u_contrib)
    ll_corr -= 0.5 * _slogdet_lu(*lup)

    ll_corr -= 0.5 * jnp.sum(C[dim:, dim:] * (f[..., None] * f))

    return pred_mean, pred_cov, ll_corr


# TODO: Check if we can write vb filter in associative scan
@preprocess_args
def lgssm_filter(
    params: ParamsLGSSMVB,
    emissions: Float[Array, "ntime emission_dim"],  # noqa F772
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,  # noqa F772
    variational_bayes=True,
) -> PosteriorGSSMFiltered:
    r"""Run a Kalman filter to produce the marginal likelihood and filtered state estimates.

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.

    Returns:
        PosteriorGSSMFiltered: filtered posterior object

    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    def _log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y):
        """Compute the log likelihood of an observation under a linear Gaussian model."""
        m = H @ pred_mean + D @ u + d
        if R.ndim == 2:
            S = R + H @ pred_cov @ H.T
            return MVN(m, S).log_prob(y)
        else:
            L = H @ jnp.linalg.cholesky(pred_cov)
            return MVNLowRank(m, R, L).log_prob(y)

    def _step(carry, t):
        """Run one step of the Kalman filter."""
        ll, pred_mean, pred_cov = carry
        dim = len(pred_mean)

        # Shorthand: get parameters and inputs for time index t
        F, B, b, Q, Cx, H, D, d, R, Cy = _get_params(params, num_timesteps, t)
        u = inputs[t]
        y = emissions[t]

        # TODO: Add corrections for Tr(Cyzz^t) where z = (x, u, 1)

        # correct for uncertainty of H, <H^TRH>=H^TRH + Cy in VBE step
        if variational_bayes:
            pred_mean, pred_cov, ll_corr = correct_for_vb(pred_mean, pred_cov, u, Cy, dim)
            # correction of log-likelihood
            # ll_const_y = 0.5 * (mvgamma(df_y/2) + dim * (jnp.log(2) - jnp.log(df_y))) for inverse wishart dist
            ll += ll_corr + params.emissions.ll

        # Update the log likelihood
        ll += _log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # correct for uncertainty of F, <F^TQF>=F^TQF + Cx in VBE step
        if variational_bayes:
            filtered_mean, filtered_cov, ll_corr = correct_for_vb(filtered_mean, filtered_cov, u, Cx, dim)
            # correction of log-likelihood
            ll += ll_corr + params.dynamics.ll

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial.mean, params.initial.cov)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)
