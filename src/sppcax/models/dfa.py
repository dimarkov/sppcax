import jax.tree_util as jtu
import jax.numpy as jnp
from jax import jit, vmap, tree, random as jr

from fastprogress.fastprogress import progress_bar

from typing import Any, Optional, Union, Tuple

from functools import partial

from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM
from dynamax.utils.utils import ensure_array_has_batch_dim, pytree_stack
from dynamax.parameters import ParameterSet
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
)
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update, MatrixNormalInverseWishart
from dynamax.linear_gaussian_ssm.models import SuffStatsLGSSM
from dynamax.linear_gaussian_ssm.inference import (
    lgssm_posterior_sample,
)

from sppcax.types import Array, Vector, Matrix, PRNGKey, Float
from sppcax.distributions import Distribution
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma, mvnig_posterior_update
from .factor_analysis_algorithms import _to_distribution


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
        self, state_dim, emission_dim, input_dim=0, has_dynamics_bias=True, has_emissions_bias=True, **kw_priors
    ):
        super().__init__(
            state_dim,
            emission_dim,
            input_dim=input_dim,
            has_dynamics_bias=has_dynamics_bias,
            has_emissions_bias=has_emissions_bias,
            **kw_priors,
        )

    def m_step(self, params: ParamsLGSSM, props: ParamsLGSSM, batch_stats: SuffStatsLGSSM, m_step_state: Any):
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
        S, m = initial_posterior.mode()

        dynamics_posterior = _posterior_update(self.dynamics_prior, dynamics_stats, props)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, : self.state_dim]
        B, b = (
            (FB[:, self.state_dim : -1], FB[:, -1])
            if self.has_dynamics_bias
            else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
        )

        emission_posterior = _posterior_update(self.emission_prior, emission_stats, props)
        R, HD = emission_posterior.mode()
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
        return params, m_step_state

    def vbm_step(
        self, params: ParamsLGSSM, props: ParamsLGSSM, batch_stats: SuffStatsLGSSM, m_step_state: Any
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
        S, m = initial_posterior.mode()

        dynamics_posterior = _posterior_update(self.dynamics_prior, dynamics_stats, props)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, : self.state_dim]
        B, b = (
            (FB[:, self.state_dim : -1], FB[:, -1])
            if self.has_dynamics_bias
            else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
        )

        emission_posterior = _posterior_update(self.emission_prior, emission_stats, props)
        R, HD = emission_posterior.mode()
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
        return params, m_step_state

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
        def em_step(params, m_step_state):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        elbos = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_logprob = em_step(params, m_step_state)
            # TODO: compute full elbo
            elbos.append(marginal_logprob)

        return params, jnp.array(elbos)

    def fit_vbem(
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
        def em_step(params, m_step_state):
            """Perform one EM step."""
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        elbos = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_logprob = em_step(params, m_step_state)
            # TODO: compute full elbo
            elbos.append(marginal_logprob)

        return params, jnp.array(elbos)

    def fit_blocked_gibbs(
        self,
        key: PRNGKey,
        initial_params: ParamsLGSSM,
        sample_size: int,
        emissions: Float[Array, "nbatch num_timesteps emission_dim"],  # noqa: F722
        inputs: Optional[Float[Array, "nbatch num_timesteps input_dim"]] = None,  # noqa: F722
    ) -> ParamsLGSSM:
        r"""Estimate parameter posterior using block-Gibbs sampler.

        Args:
            key: random number key.
            initial_params: starting parameters.
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
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
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
            rngs = iter(jr.split(rng, 3))

            # Sample the initial params
            initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
            S, m = initial_posterior.sample(seed=next(rngs))

            # Sample the dynamics params
            dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
            Q, FB = dynamics_posterior.sample(seed=next(rngs))
            F = FB[:, : self.state_dim]
            B, b = (
                (FB[:, self.state_dim : -1], FB[:, -1])
                if self.has_dynamics_bias
                else (FB[:, self.state_dim :], jnp.zeros(self.state_dim))
            )

            # Sample the emission params
            emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
            R, HD = emission_posterior.sample(seed=next(rngs))
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
            return params

        @jit
        def one_sample(_params, rng):
            """Sample a single set of states and compute their sufficient stats."""
            rngs = jr.split(rng, 2)
            # Sample latent states
            batch_keys = jr.split(rngs[0], num=num_batches)
            forward_backward_batched = vmap(partial(lgssm_posterior_sample, params=_params))
            batch_states = forward_backward_batched(batch_keys, emissions=batch_emissions, inputs=batch_inputs)
            _batch_stats = vmap(sufficient_stats_from_sample)(batch_emissions, batch_inputs, batch_states)
            # Aggregate statistics from all observations.
            _stats = tree.map(lambda x: jnp.sum(x, axis=0), _batch_stats)
            # Sample parameters
            return lgssm_params_sample(rngs[1], _stats)

        sample_of_params = []
        keys = iter(jr.split(key, sample_size))
        current_params = initial_params
        for _ in progress_bar(range(sample_size)):
            sample_of_params.append(current_params)
            current_params = one_sample(current_params, next(keys))

        return pytree_stack(sample_of_params)


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
