import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx
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
from sppcax.distributions import Distribution
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma, mvnig_posterior_update
from sppcax.distributions.utils import cho_inv
from sppcax.inference.utils import ParamsLGSSMVB
from sppcax.inference.smoothing import lgssm_smoother as sppcax_smoother
from .factor_analysis_algorithms import _to_distribution

from sppcax.metrics import kl_divergence
from sppcax.metrics.kl_divergence import multidigamma, digamma

from sppcax.bmr import prune_params


def _mniw_posterior_update(dist, stats, props):
    # TODO: filter stats based on props
    return mniw_posterior_update(dist, stats)


def _niw_posterior_update(dist, stats, props):
    # TODO: filter stats based on props
    return niw_posterior_update(dist, stats)


def _posterior_update(dist, stats, props):
    if isinstance(dist, MultivariateNormalInverseGamma):
        return mvnig_posterior_update(dist, stats, props)
    elif isinstance(dist, MatrixNormalInverseWishart):
        return _mniw_posterior_update(dist, stats, props)
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

    else:
        raise NotImplementedError


def _get_correction(dist):
    if isinstance(dist, MatrixNormalInverseWishart):
        dim, _ = dist._matrix_normal_shape
        col_precision = dist.col_precision
        return dim * cho_inv(col_precision)

    elif isinstance(dist, MultivariateNormalInverseGamma):
        # inverse of expected precision
        return dist.col_covariance.sum(-3)

    else:
        raise NotImplementedError


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
        **kw_priors,
    ):
        super().__init__(
            state_dim,
            emission_dim,
            input_dim=input_dim,
            has_dynamics_bias=has_dynamics_bias,
            has_emissions_bias=has_emissions_bias,
            **kw_priors,
        )

        self.parallel_scan = parallel_scan

        if use_bmr:
            self.use_bmr = ParamsBMR(initial=True, dynamics=False, emissions=True)
        else:
            self.use_bmr = ParamsBMR(initial=False, dynamics=False, emissions=False)

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
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
        _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

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
            C_dyn = jnp.zeros((dim, dim))
            dim = self.state_dim + self.input_dim + self.has_emissions_bias
            C_em = jnp.zeros((dim, dim))

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

        # The keys of param_props must match those of params!
        props = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(), cov=ParameterProperties(constrainer=RealToPSDBijector())
            ),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
            ),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
            ),
        )
        return params, props

    def m_step(
        self,
        params: ParamsLGSSM,
        props: ParamsLGSSM,
        batch_stats: SuffStatsLGSSM,
        m_step_state: Any,
        key: PRNGKey = None,
    ):
        """Perform the M-step of the EM algorithm.

        Note: This function currently ignores any `trainable` constraints specified
        in the `props` argument.

        Args:
            params: model parameters.
            props: parameter properties.
            batch_stats: expected sufficient statistics.
            m_step_state: state for the M-step.

        Returns:
            updated model parameters and updated M-step state.
        """
        # Sum the statistics across all batches
        stats = jtu.tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        initial_posterior = _niw_posterior_update(self.initial_prior, init_stats, props)
        if self.use_bmr.initial and key is not None:
            key, _key = jr.split(key)
            initial_posterior = prune_params(initial_posterior, self.initial_prior, key=_key)

        S, m = initial_posterior.mode()
        kl_div = kl_divergence(initial_posterior, self.initial_prior)

        dynamics_posterior = _posterior_update(self.dynamics_prior, dynamics_stats, props)
        if self.use_bmr.dynamics and key is not None:
            key, _key = jr.split(key)
            dynamics_posterior = prune_params(dynamics_posterior, self.dynamics_prior, key=_key)

        Q, FB = dynamics_posterior.mode()
        kl_div += kl_divergence(dynamics_posterior, self.dynamics_prior)
        F = FB[:, : self.state_dim]
        B, b = (
            (FB[:, self.state_dim : -1], FB[:, -1])
            if self.has_dynamics_bias
            else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
        )

        emission_posterior = _posterior_update(self.emission_prior, emission_stats, props)
        if self.use_bmr.emissions and key is not None:
            key, _key = jr.split(key)
            emission_posterior = prune_params(emission_posterior, self.emission_prior, key=_key)
        R, HD = emission_posterior.mode()
        kl_div += kl_divergence(emission_posterior, self.emission_prior)
        H = HD[:, : self.state_dim]
        D, d = (
            (HD[:, self.state_dim : -1], HD[:, -1])
            if self.has_emissions_bias
            else (HD[:, self.state_dim :], jnp.zeros(self.emission_dim))
        )

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R),
        )
        return params, kl_div, m_step_state

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
        # posterior = lgssm_smoother(params, emissions, inputs, variational_bayes=False)
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
    ) -> Tuple[ParamsLGSSM, Any]:
        """Perform the variational M-step of the VBEM algorithm.

        Note: This function currently ignores any `trainable` constraints specified
        in the `props` argument.

        Args:
            params: model parameters.
            props: parameter properties.
            batch_stats: expected sufficient statistics.
            m_step_state: state for the M-step.

        Returns:
            updated model parameters and updated M-step state.
        """
        # Sum the statistics across all batches
        stats = jtu.tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        initial_posterior = _niw_posterior_update(self.initial_prior, init_stats, props)
        if self.use_bmr.initial and key is not None:
            key, _key = jr.split(key)
            initial_posterior = prune_params(initial_posterior, self.initial_prior, key=_key)

        kl_div = kl_divergence(initial_posterior, self.initial_prior)
        S, m = _get_moments(initial_posterior)

        emission_posterior = _posterior_update(self.emission_prior, emission_stats, props)
        if self.use_bmr.emissions and key is not None:
            key, _key = jr.split(key)
            emission_posterior = prune_params(emission_posterior, self.emission_prior, key=_key)

        kl_div += kl_divergence(emission_posterior, self.emission_prior)
        R, HD = _get_moments(emission_posterior)
        H = HD[:, : self.state_dim]
        D, d = (
            (HD[:, self.state_dim : -1], HD[:, -1])
            if self.has_emissions_bias
            else (HD[:, self.state_dim :], jnp.zeros(self.emission_dim))
        )

        dynamics_posterior = _posterior_update(self.dynamics_prior, dynamics_stats, props)
        if self.use_bmr.dynamics and key is not None:
            key, _key = jr.split(key)
            dynamics_posterior = prune_params(dynamics_posterior, self.dynamics_prior, key=_key)

        kl_div += kl_divergence(dynamics_posterior, self.dynamics_prior)
        Q, FB = _get_moments(dynamics_posterior)

        F = FB[:, : self.state_dim]
        B, b = (
            (FB[:, self.state_dim : -1], FB[:, -1])
            if self.has_dynamics_bias
            else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
        )

        # Get correction for Q, R
        C_em = _get_correction(emission_posterior)
        C_dyn = _get_correction(dynamics_posterior)

        ll_const_em = _get_ll_correction(emission_posterior)
        ll_const_dyn = _get_ll_correction(dynamics_posterior)

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMVB(weights=F, bias=b, input_weights=B, cov=Q, correction=C_dyn, ll=ll_const_dyn),
            emissions=ParamsLGSSMVB(weights=H, bias=d, input_weights=D, cov=R, correction=C_em, ll=ll_const_em),
        )
        return params, kl_div, m_step_state

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

    def fit_em(
        self,
        params: ParamsLGSSM,
        props: ParamsLGSSM,
        Y: Union[Matrix, Distribution],  # DFA expects a time series matrix
        key: PRNGKey,
        U: Union[Matrix, Distribution] = None,  # inputs/controls
        num_iters: int = 100,
        verbose: bool = True,
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
        batch_emissions = ensure_array_has_batch_dim(Ey, self.emission_shape)

        if U_dist is not None:
            Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
            batch_inputs = ensure_array_has_batch_dim(Eu, self.inputs_shape)
        else:
            batch_inputs = None
        # TODO: figure out how to deal with (y, u) uncertainties in dynamax

        @jit
        def em_step(params, kl_div, m_step_state, key):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            elbo = lls.sum() - kl_div
            params, kl_div, m_step_state = self.m_step(params, props, batch_stats, m_step_state, key=key)
            return params, kl_div, elbo, m_step_state

        m_step_state = self.initialize_m_step_state(params, props)
        kl_div = 0.0
        carry = (params, kl_div, m_step_state, key)

        def step_fn(carry, *args):
            params, kl_div, m_step_state, key = carry
            key, _key = jr.split(key)
            params, kl_div, elbo, m_step_state = em_step(params, kl_div, m_step_state, _key)
            return (params, kl_div, m_step_state, key), elbo

        if verbose:
            pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
            elbos = []
            for _ in pbar:
                carry, elbo = step_fn(carry)
                elbos.append(elbo)
                params = carry[0]
            elbos = jnp.stack(elbos).squeeze()
        else:
            (params, *_), elbos = lax.scan(step_fn, carry, jnp.arange(num_iters))
            elbos = elbos.squeeze()

        return params, elbos[1:]

    def fit_vbem(
        self,
        params: ParamsLGSSMVB,
        props: ParamsLGSSM,
        Y: Union[Matrix, Distribution],  # DFA expects a time series matrix
        key: PRNGKey,
        U: Union[Matrix, Distribution] = None,  # inputs/controls
        num_iters: int = 100,
        verbose: bool = True,
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
        batch_emissions = ensure_array_has_batch_dim(Ey, self.emission_shape)

        if U_dist is not None:
            Eu = U_dist.mean if hasattr(U, "mean") else U_dist.location
            batch_inputs = ensure_array_has_batch_dim(Eu, self.inputs_shape)
        else:
            batch_inputs = None
        # TODO: figure out how to deal with (y, u) uncertainties in dynamax

        def em_step(params, kl_div, m_step_state, key):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.vbe_step, params))(batch_emissions, batch_inputs)
            elbo = -kl_div + lls.sum()
            params, kl_div, m_step_state = self.vbm_step(params, props, batch_stats, m_step_state, key=key)
            return params, kl_div, elbo, m_step_state

        m_step_state = self.initialize_m_step_state(params, props)
        kl_div = 0.0
        carry = (params, kl_div, m_step_state, key)

        def step_fn(carry, *args):
            params, kl_div, m_step_state, key = carry
            key, _key = jr.split(key)
            params, kl_div, elbo, m_step_state = em_step(params, kl_div, m_step_state, _key)
            return (params, kl_div, m_step_state, key), elbo

        if verbose:
            pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
            elbos = []
            for _ in pbar:
                carry, elbo = jit(step_fn)(carry)
                elbos.append(elbo)
                params = carry[0]
            elbos = jnp.stack(elbos)
        else:
            (params, *_), elbos = lax.scan(step_fn, carry, jnp.arange(num_iters))

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
    ) -> ParamsLGSSM:
        r"""Estimate parameter posterior using block-Gibbs sampler.

        Args:
            key: random number key.
            initial_params: starting parameters.
            props: parameter properties.
            sample_size: how many samples to draw.
            emissions: set of observation sequences.
            inputs: optional set of input sequences.

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

        def lgssm_params_sample(rng, stats):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats

            # Sample the initial params
            initial_posterior = _niw_posterior_update(self.initial_prior, init_stats, props)
            if self.use_bmr.initial:
                rng, key = jr.split(rng)
                initial_posterior = prune_params(initial_posterior, self.initial_prior, key=key)
            kl_div = kl_divergence(initial_posterior, self.initial_prior)

            rng, key = jr.split(rng)
            S, m = initial_posterior.sample(seed=key)

            # Sample the dynamics params
            dynamics_posterior = _posterior_update(self.dynamics_prior, dynamics_stats, props)
            if self.use_bmr.dynamics:
                rng, key = jr.split(rng)
                dynamics_posterior = prune_params(dynamics_posterior, self.dynamics_prior, key=key)

            kl_div += kl_divergence(dynamics_posterior, self.dynamics_prior)

            rng, key = jr.split(rng)
            Q, FB = dynamics_posterior.sample(seed=key)
            F = FB[:, : self.state_dim]
            B, b = (
                (FB[:, self.state_dim : -1], FB[:, -1])
                if self.has_dynamics_bias
                else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
            )

            # Sample the emission params
            emission_posterior = _posterior_update(self.emission_prior, emission_stats, props)
            if self.use_bmr.emissions:
                rng, key = jr.split(rng)
                emission_posterior = prune_params(emission_posterior, self.emission_prior, key=key)

            kl_div += kl_divergence(emission_posterior, self.emission_prior)

            rng, key = jr.split(rng)
            R, HD = emission_posterior.sample(seed=key)

            H = HD[:, : self.state_dim]
            D, d = (
                (HD[:, self.state_dim : -1], HD[:, -1])
                if self.has_emissions_bias
                else (HD[:, self.state_dim :], jnp.zeros(self.emission_dim))
            )

            params = ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R),
            )

            return params, kl_div

        def one_sample(_params, kl_div, rng):
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
            return lgssm_params_sample(rngs[1], _stats), elbo

        def step_fn(carry, *args):
            params, kl_div, key = carry
            key, _key = jr.split(key)
            (current_params, kl_div), elbo = one_sample(params, kl_div, _key)
            return (current_params, kl_div, key), (params, elbo)

        kl_div = 0.0
        carry = (initial_params, kl_div, key)
        if verbose:
            sample_of_params = []
            elbos = []
            pb = progress_bar(range(sample_size))
            for _ in pb:
                carry, (current_params, elbo) = jit(step_fn)(carry)
                sample_of_params.append(current_params)
                elbos.append(elbo)

            sample_of_params = pytree_stack(sample_of_params[burn_in:])
            elbos = jnp.stack(elbos).squeeze()[burn_in:]
        else:
            _, (sample_of_params, elbos) = lax.scan(step_fn, carry, jnp.arange(sample_size))
            sample_of_params = jtu.tree_map(lambda x: x[burn_in:], sample_of_params)
            elbos = elbos.squeeze()[burn_in:]

        return sample_of_params, elbos
