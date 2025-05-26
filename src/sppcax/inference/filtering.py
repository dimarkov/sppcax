"""
This module contains functions for filtering in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp

from jax import lax
from jaxtyping import Array, Float
from typing import Optional

from jax.scipy.linalg import solve_triangular

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiagPlusLowRankCovariance as MVNLowRank,
    MultivariateNormalFullCovariance as MVN,
)

from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
from .utils import _get_params, _predict, _condition_on, ParamsLGSSM, preprocess_args


# TODO: Check if we can write vb filter in associative scan
@preprocess_args
def lgssm_filter(
    params: ParamsLGSSM,
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

        # Shorthand: get parameters and inputs for time index t
        F, B, b, Q, Cx, H, D, d, R, Cy = _get_params(params, num_timesteps, t)
        u = inputs[t]
        y = emissions[t]

        # correct for uncertainty of H, <H^TRH>=H^TRH + Cy in VBE step
        if variational_bayes:
            dim = len(pred_mean)
            C = jnp.eye(dim) + pred_cov @ Cy[..., :dim, :dim]
            q, r = jnp.linalg.qr(C)
            pred_cov = solve_triangular(r, q.mT @ pred_cov)
            pred_mean = solve_triangular(r, q.mT @ pred_mean)

        # Update the log likelihood
        ll += _log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # correct for uncertainty of F, <F^TQF>=F^TQF + Cx in VBE step
        if variational_bayes:
            C = jnp.eye(dim) + filtered_cov @ Cx[..., :dim, :dim]
            q, r = jnp.linalg.qr(C)
            filtered_cov = solve_triangular(r, q.mT @ filtered_cov)
            filtered_mean = solve_triangular(r, q.mT @ filtered_mean)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial.mean, params.initial.cov)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)
