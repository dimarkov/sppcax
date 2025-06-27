"""
This module contains utility functions for inference in linear Gaussian state space models (LGSSMs).
"""
import inspect
import warnings

from functools import wraps
from dynamax.parameters import ParameterProperties
from typing import NamedTuple, Optional, Union, Tuple

from dynamax.linear_gaussian_ssm.inference import (
    _get_one_param,
    _zeros_if_none,
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
)
from sppcax.types import Array, Float
import equinox as eqx


class ParamsLGSSMVB(NamedTuple):
    r"""Parameters of the distributions with correction term used for variational bayes.

    The tuple doubles as a container for the ParameterProperties.

    :param weights: weights
    :param bias: bias
    :param input_weights: input weights
    :param cov: covariance
    :param correction: Cov(A^T Q A) or Cov(H^T R H)

    """
    weights: Union[ParameterProperties, Float[Array, "..."]]  # noqa F772
    bias: Union[ParameterProperties, Float[Array, "..."]]  # noqa F772
    input_weights: Union[ParameterProperties, Float[Array, "..."]]  # noqa F772
    cov: Union[ParameterProperties, Float[Array, "..."]]  # noqa F772
    correction: Union[ParameterProperties, Float[Array, "..."]]  # noqa F772
    ll: float


# Helper functions
def _get_params(params, num_timesteps, t):
    """Helper function to get all parameters at time t."""
    assert not callable(params.emissions.cov), "Emission covariance cannot be a callable."

    F = _get_one_param(params.dynamics.weights, 2, t)
    B = _get_one_param(params.dynamics.input_weights, 2, t)
    b = _get_one_param(params.dynamics.bias, 1, t)
    Q = _get_one_param(params.dynamics.cov, 2, t)
    Cx = _get_one_param(params.dynamics.correction, 2, t)
    H = _get_one_param(params.emissions.weights, 2, t)
    D = _get_one_param(params.emissions.input_weights, 2, t)
    d = _get_one_param(params.emissions.bias, 1, t)
    Cy = _get_one_param(params.emissions.correction, 2, t)

    if len(params.emissions.cov.shape) == 1:
        R = _get_one_param(params.emissions.cov, 1, t)
    elif len(params.emissions.cov.shape) > 2:
        R = _get_one_param(params.emissions.cov, 2, t)
    elif params.emissions.cov.shape[0] != num_timesteps:
        R = _get_one_param(params.emissions.cov, 2, t)
    elif params.emissions.cov.shape[1] != num_timesteps:
        R = _get_one_param(params.emissions.cov, 1, t)
    else:
        R = _get_one_param(params.emissions.cov, 2, t)
        warnings.warn(
            "Emission covariance has shape (N,N) where N is the number of timesteps. "
            "The covariance will be interpreted as static and non-diagonal. To "
            "specify a dynamic and diagonal covariance, pass it as a 3D array.",
            stacklevel=2,
        )

    return F, B, b, Q, Cx, H, D, d, R, Cy


def make_lgssm_params(
    initial_mean: Float[Array, " state_dim"],  # noqa F772
    initial_cov: Float[Array, "state_dim state_dim"],  # noqa F772
    dynamics_weights: Float[Array, "state_dim state_dim"],  # noqa F772
    dynamics_cov: Float[Array, "state_dim state_dim"],  # noqa F772
    emissions_weights: Float[Array, "emission_dim state_dim"],  # noqa F772
    emissions_cov: Float[Array, "emission_dim emission_dim"],  # noqa F772
    dynamics_bias: Optional[Float[Array, " state_dim"]] = None,  # noqa F772
    dynamics_input_weights: Optional[Float[Array, "state_dim input_dim"]] = None,  # noqa F772
    emissions_bias: Optional[Float[Array, " emission_dim"]] = None,  # noqa F772
    emissions_input_weights: Optional[Float[Array, "emission_dim input_dim"]] = None,  # noqa F772
) -> ParamsLGSSM:
    """Helper function to construct a ParamsLGSSM object from arguments.

    See `ParamsLGSSM`, `ParamsLGSSMInitial`, `ParamsLGSSMDynamics`, and `ParamsLGSSMEmissions` for
    more details on the parameters.
    """
    state_dim = len(initial_mean)
    emission_dim = emissions_cov.shape[-1]
    input_dim = max(
        dynamics_input_weights.shape[-1] if dynamics_input_weights is not None else 0,
        emissions_input_weights.shape[-1] if emissions_input_weights is not None else 0,
    )

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=initial_mean, cov=initial_cov),
        dynamics=ParamsLGSSMDynamics(
            weights=dynamics_weights,
            bias=_zeros_if_none(dynamics_bias, state_dim),
            input_weights=_zeros_if_none(dynamics_input_weights, (state_dim, input_dim)),
            cov=dynamics_cov,
        ),
        emissions=ParamsLGSSMEmissions(
            weights=emissions_weights,
            bias=_zeros_if_none(emissions_bias, emission_dim),
            input_weights=_zeros_if_none(emissions_input_weights, (emission_dim, input_dim)),
            cov=emissions_cov,
        ),
    )
    return params


def preprocess_params_and_inputs(
    params: ParamsLGSSM, num_timesteps: int, inputs: Optional[Float[Array, "num_timesteps input_dim"]]  # noqa F772
) -> Tuple[ParamsLGSSM, Float[Array, "num_timesteps input_dim"]]:  # noqa F772
    """Preprocess parameters in case some are set to None.

    Args:
        params: model parameters
        num_timesteps: number of timesteps
        inputs: optional array of inputs.

    Returns:
        full_params: full parameters with zeros for missing parameters
        inputs: processed inputs (zero if None)
    """

    # Make sure all the required parameters are there
    assert params.initial.mean is not None
    assert params.initial.cov is not None
    assert params.dynamics.weights is not None
    assert params.dynamics.cov is not None
    assert params.emissions.weights is not None
    assert params.emissions.cov is not None

    # Get shapes
    emission_dim, state_dim = params.emissions.weights.shape[-2:]

    # Default the inputs to zero
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))
    input_dim = inputs.shape[-1]

    # Default other parameters to zero
    dynamics_input_weights = _zeros_if_none(params.dynamics.input_weights, (state_dim, input_dim))
    dynamics_bias = _zeros_if_none(params.dynamics.bias, (state_dim,))
    emissions_input_weights = _zeros_if_none(params.emissions.input_weights, (emission_dim, input_dim))
    emissions_bias = _zeros_if_none(params.emissions.bias, (emission_dim,))

    full_params = eqx.tree_at(
        lambda p: (p.dynamics.bias, p.dynamics.input_weights, p.emissions.bias, p.emissions.input_weights),
        params,
        (dynamics_bias, dynamics_input_weights, emissions_bias, emissions_input_weights),
    )

    return full_params, inputs


def preprocess_args(f):
    """Preprocess the parameter and input arguments in case some are set to None."""
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        """Wrapper function to preprocess arguments."""
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments["params"]
        emissions = bound_args.arguments["emissions"]
        inputs = bound_args.arguments["inputs"]
        variational_bayes = bound_args.arguments["variational_bayes"]

        num_timesteps = len(emissions)
        full_params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

        return f(full_params, emissions, inputs=inputs, variational_bayes=variational_bayes)

    return wrapper
