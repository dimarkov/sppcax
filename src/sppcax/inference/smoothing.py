"""
This module contains functions for inference in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp
from functools import partial
from jax import lax, vmap
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Float
from dynamax.utils.utils import symmetrize
from typing import NamedTuple, Optional, Union

from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMSmoothed, _zeros_if_none, _get_one_param
from dynamax.linear_gaussian_ssm.parallel_inference import lgssm_filter
from .utils import ParamsLGSSM, ParamsLGSSMVB, preprocess_args
from .filtering import lgssm_filter as sppcax_filter


class SmoothMessage(NamedTuple):
    """
    Smoothing associative scan elements.

    Attributes:
        E: P(z_i | y_{1:j}, z_{j+1}) weights.
        g: P(z_i | y_{1:j}, z_{j+1}) bias.
        L: P(z_i | y_{1:j}, z_{j+1}) covariance.
    """

    E: Float[Array, "num_timesteps state_dim state_dim"]  # noqa F772
    g: Float[Array, "num_timesteps state_dim"]  # noqa F772
    L: Float[Array, "num_timesteps state_dim state_dim"]  # noqa F772


def _initialize_smoothing_messages(
    params: Union[ParamsLGSSM, ParamsLGSSMVB],
    filtered_means: Float[Array, "num_timesteps state_dim"],  # noqa F772
    filtered_covariances: Float[Array, "num_timesteps state_dim state_dim"],  # noqa F772
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,  # noqa F772
) -> SmoothMessage:
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    def _last_message(m, P):
        """Compute the last smoothing message."""
        return jnp.zeros_like(P), m, P

    num_timesteps = filtered_means.shape[0]
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _generic_message(params, m, P, ut, t):
        """Compute the generic smoothing message."""
        F = _get_one_param(params.dynamics.weights, 2, t)
        B = _get_one_param(params.dynamics.input_weights, 2, t)
        b = _get_one_param(params.dynamics.bias, 1, t)
        Q = _get_one_param(params.dynamics.cov, 2, t)

        CF, low = cho_factor(F @ P @ F.T + Q)
        E = cho_solve((CF, low), F @ P).T
        g = m - E @ (F @ m + B @ ut + b)
        L = symmetrize(P - E @ F @ P)
        return E, g, L

    En, gn, Ln = _last_message(filtered_means[-1], filtered_covariances[-1])
    Et, gt, Lt = _generic_message(
        params, filtered_means[:-1], filtered_covariances[:-1], inputs[:-1], jnp.arange(len(filtered_means) - 1)
    )

    return SmoothMessage(
        E=jnp.concatenate([Et, En[None]]), g=jnp.concatenate([gt, gn[None]]), L=jnp.concatenate([Lt, Ln[None]])
    )


@preprocess_args
def lgssm_smoother(
    params: Union[ParamsLGSSM, ParamsLGSSMVB],
    emissions: Float[Array, "ntime emission_dim"],  # noqa F772
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,  # noqa F772
    *,
    variational_bayes: bool,
) -> PosteriorGSSMSmoothed:
    """A parallel version of the lgssm smoothing algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    """
    filtered_posterior = (
        sppcax_filter(params, emissions, inputs) if variational_bayes else lgssm_filter(params, emissions, inputs)
    )
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    @vmap
    def _operator(elem1, elem2):
        """Parallel smoothing operator."""
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2
        E = E2 @ E1
        g = E2 @ g1 + g2
        L = symmetrize(E2 @ L1 @ E2.T + L2)
        return E, g, L

    initial_messages = _initialize_smoothing_messages(params, filtered_means, filtered_covs, inputs)
    final_messages = lax.associative_scan(_operator, initial_messages, reverse=True)

    smoothed_mean = final_messages.g[:-1]
    smoothed_mean_next = final_messages.g[1:]
    smoothed_cov_next = final_messages.L[1:]

    G = initial_messages.E[:-1]
    smoothed_cross = G @ smoothed_cov_next + vmap(jnp.outer)(smoothed_mean, smoothed_mean_next)

    return PosteriorGSSMSmoothed(
        marginal_loglik=filtered_posterior.marginal_loglik,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=final_messages.g,
        smoothed_covariances=final_messages.L,
        smoothed_cross_covariances=smoothed_cross,
    )
