"""Functions for Bayesian model reduction."""

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from multipledispatch import dispatch

from ..distributions import Beta, MultivariateNormal, Gamma, MultivariateNormalInverseGamma as MVNIG
from ..models.factor_analysis_params import BayesianFactorAnalysisParams, PFA
from ..types import PRNGKey
from .delta_f import gibbs_sampler_mvn, gibbs_sampler_pfa, gibbs_sampler_mvnig

from dynamax.utils.distributions import NormalInverseWishart as NIW


@dispatch(NIW, NIW)
def prune_params(post: NIW, prior: NIW, *, key: PRNGKey, max_iter: int = 4) -> NIW:
    """Applies Bayesian model reduction to distribution, where Inverse Gamma priors
    get optimized, and individual elements of the Multivariate Normal get pruned to
    maximize change in the variational free energy, hence minimize the upper bound
    on marginal log-likelihood.

    Args:
        post: Posterior MultivariateNormalInverseGamma distribution
        prior: Prior MultivariateNormalInverseGamma distribution
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Optimized posterior MultivariateNormalInverseGamma distribution
    """

    # optimize Inverse Gamma priors and consequently posterior estimates

    nu = post.df
    nu_0 = prior.df
    diff_scale = post.scale - prior.scale
    new_scale = nu * diff_scale / (nu - nu_0)

    opt_post = NIW(post.loc, post.mean_concentration, nu, new_scale)

    # TODO: prune parameters
    # Decide if we should to force a priori some latents to zero.

    return opt_post


@dispatch(MVNIG, MVNIG)
def prune_params(post: MVNIG, prior: MVNIG, *, key: PRNGKey, max_iter: int = 10) -> MVNIG:  # noqa: F811
    """Applies Bayesian model reduction to distribution, where Inverse Gamma priors
    get optimized, and individual elements of the Multivariate Normal get pruned to
    maximize change in the variational free energy, hence minimize the upper bound
    on marginal log-likelihood.

    Args:
        post: Posterior MultivariateNormalInverseGamma distribution
        prior: Prior MultivariateNormalInverseGamma distribution
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Optimized posterior MultivariateNormalInverseGamma distribution
    """

    # prune parameters
    def step_fn(carry, t):
        delta_f, mask, key = carry

        alpha = 1.0 + mask.sum(0)
        beta = 1.0 + (1 - mask).sum(0)

        key, _key = jr.split(key)
        pi = Beta(alpha, beta).sample(_key)

        key, _key = jr.split(key)
        delta_f, mask = gibbs_sampler_mvnig(_key, post, prior, pi, mask, delta_f)

        return (delta_f, mask, key), delta_f

    init = (jnp.zeros(post.batch_shape), prior.mvn.mask, key)
    (last_df, mask, key), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter), unroll=2)

    # correct beta parameter of inverse-gamma distribution
    prior_nat1 = prior.mvn.nat1
    pruned_prior_mean = (~mask) * prior.mvn.mean

    post_nat1 = post.mvn.nat1
    pruned_post_mean = (~mask) * post.mvn.mean

    dnat2 = post.inv_gamma.dnat2
    dnat2 -= jnp.sum(pruned_post_mean * post_nat1, -1) / 2
    dnat2 += jnp.sum(pruned_prior_mean * prior_nat1, -1) / 2

    # optimize Inverse Gamma priors
    nat2_0 = jnp.minimum(dnat2 / (post.inv_gamma.alpha - 1), -1 / (post.inv_gamma.alpha - 1))
    dnat2 = jnp.minimum(dnat2, 0.0)
    opt_post = eqx.tree_at(lambda d: (d.mvn.mask, d.inv_gamma.nat2_0, d.inv_gamma.dnat2), post, (mask, nat2_0, dnat2))

    return opt_post


@dispatch(PFA)
def reduce_model(model: PFA, *, key: PRNGKey, max_iter: int = 4) -> PFA:
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        model: Probabilistic factor analysis model
        key: Random number generator key
        max_iter: Maximal number of iterations for the Gibbs sampler

    Returns:
        Pruned PFA model
    """

    def step_fn(carry, t):
        delta_f, sparsity_post, lam, key = carry

        # Compute delta F and sparsity matrix lambda for each parameter
        key, _key = jr.split(key)
        pi = sparsity_post.sample(_key)
        key, _key = jr.split(key)
        delta_f, lam = gibbs_sampler_pfa(_key, model, pi, lam, delta_f)

        dnat1 = lam.astype(delta_f.dtype).sum(0)
        dnat2 = (1 - lam).astype(delta_f.dtype).sum(0)
        sparsity_post = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), sparsity_post, (dnat1, dnat2))

        return (delta_f, sparsity_post, lam, key), delta_f

    # Initialize state for scan
    init = (jnp.zeros(model.n_features), model.sparsity_prior, model.q_w_psi.mvn.mask, key)
    (last_df, sparsity_post, lam, key), delta_fs = lax.scan(step_fn, init, jnp.arange(max_iter))

    # --- Update model components based on BMR results ---

    # Update q(W|psi) mask
    nat1_w = jnp.where(lam, model.q_w_psi.mvn.nat1, 0.0)
    updated_mvn = eqx.tree_at(lambda x: (x.mask, x.nat1), model.q_w_psi.mvn, (lam, nat1_w))

    # Update q(tau) shape parameter based on new mask
    dnat1_tau = lam.sum(0) / 2
    updated_q_tau = eqx.tree_at(lambda x: x.dnat1, model.q_tau, dnat1_tau)

    # Update q(psi) rate parameter based on pruned components
    dnat2_psi = model.q_w_psi.gamma.dnat2  # Start with original rate parameter
    pruned_mask_diff = (
        model.q_w_psi.mvn.mask.astype(jnp.int8) - lam
    )  # Identify pruned elements (1 where pruned, 0 otherwise)
    tilde_mu = pruned_mask_diff * model.q_w_psi.mvn.mean  # Get means of pruned elements

    # Adjust rate parameter by removing contribution of pruned elements' variance
    dnat2_psi -= 0.5 * (tilde_mu[..., None, :] @ (model.q_w_psi.mvn.covariance @ tilde_mu[..., None])).squeeze((-1, -2))
    updated_q_psi = eqx.tree_at(lambda x: x.dnat2, model.q_w_psi.gamma, dnat2_psi)

    # Combine updated mvn and psi into q_w_psi
    updated_q_w_psi = eqx.tree_at(lambda x: (x.mvn, x.gamma), model.q_w_psi, (updated_mvn, updated_q_psi))

    # Update the final model state
    model = eqx.tree_at(
        lambda x: (x.q_w_psi, x.sparsity_prior, x.q_tau),
        model,
        (updated_q_w_psi, sparsity_post, updated_q_tau),
    )

    return model


@dispatch(MultivariateNormal)
def reduce_model(  # noqa: F811
    model: MultivariateNormal, *, key: PRNGKey, pi: float = 0.5, max_iter: int = 1
) -> MultivariateNormal:
    """Reduce model by pruning parameters with insufficient evidence.

    Args:
        model: MultivariateNormal distribution
        pi: Prior expected probability of parameter being pruned
        max_iter: Maximal number of iterations for the Gibbs sampler
        key: Random state for initialization

    Returns:
        Pruned MultivariateNormal distribution
    """

    def step_fn(carry, t):
        delta_f, sparsity_dist, lam, key = carry

        # Compute delta F and sparsity matrix lambda for each parameter
        key, _key = jr.split(key)
        pi = sparsity_dist.sample(_key)
        key, _key = jr.split(key)
        delta_f, lam = gibbs_sampler_mvn(_key, model, pi, lam, delta_f)

        dnat1 = lam.astype(delta_f.dtype).sum(0)
        dnat2 = (1 - lam).astype(delta_f.dtype).sum(0)
        sparsity_dist = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), sparsity_dist, (dnat1, dnat2))

        return (delta_f, sparsity_dist, lam, key), delta_f

    K = model.mask.shape[-1]
    sparity_prior = Beta(10 * pi * jnp.ones(K), 10 * (1 - pi) * jnp.ones(K))
    init = (jnp.zeros(model.batch_shape), sparity_prior, model.mask, key)
    (delta_f, sparsity_dist, lam, key), _ = lax.scan(step_fn, init, jnp.arange(max_iter))

    # Update the MVN distribution
    model = eqx.tree_at(lambda x: (x.mask, x.nat1), model, (lam, jnp.where(lam, model.nat1, 0.0)))

    return model


def remove_redundant_latents(model: BayesianFactorAnalysisParams) -> BayesianFactorAnalysisParams:
    """Remove redundant latent variables (columns) from the model based on q_w_psi mask."""
    # Find components (columns) where at least one feature weight is active
    remaining_components_idx = jnp.nonzero(model.q_w_psi.mvn.mask.sum(0) > 0)[0]  # Get indices

    # Prune q_tau (ARD prior precision)
    alpha_tau = model.q_tau.alpha0[
        remaining_components_idx
    ]  # Assuming alpha0/beta0 are priors, dnat1/dnat2 are updates
    beta_tau = model.q_tau.beta0[remaining_components_idx]
    dnat1_tau = model.q_tau.dnat1[remaining_components_idx]
    dnat2_tau = model.q_tau.dnat2[remaining_components_idx]
    # Recreate q_tau with pruned parameters
    pruned_q_tau = Gamma(alpha0=alpha_tau, beta0=beta_tau)
    pruned_q_tau = eqx.tree_at(lambda x: (x.dnat1, x.dnat2), pruned_q_tau, (dnat1_tau, dnat2_tau))

    # Prune q_w_psi (MVN part) - select columns corresponding to remaining components
    # This requires careful handling of MultivariateNormalGamma structure
    # Assuming loc, mask, covariance etc. can be indexed along the component dimension (last dim for mean/loc, last two for cov)
    pruned_mvn_loc = model.q_w_psi.mvn.loc[:, remaining_components_idx]
    pruned_mvn_mask = model.q_w_psi.mvn.mask[:, remaining_components_idx]
    # Covariance pruning is tricky. Assuming diagonal/block structure allows simple selection.
    # If cov has shape (Features, K, K), select blocks along last two dims.
    # If cov has shape (Features * K, Features * K), it's more complex.
    # Assuming (Features, K, K) structure for simplicity:
    # TODO: Verify covariance structure and pruning logic in MultivariateNormalGamma
    pruned_mvn_cov = model.q_w_psi.mvn.covariance[:, remaining_components_idx][:, :, remaining_components_idx]
    # Recreate the MVN part - need to handle precision/nat params if used internally
    # This assumes MultivariateNormalGamma can be reconstructed this way.
    pruned_mvn = eqx.tree_at(lambda x: x.loc, model.q_w_psi.mvn, pruned_mvn_loc)
    pruned_mvn = eqx.tree_at(lambda x: x.mask, pruned_mvn, pruned_mvn_mask)
    pruned_mvn = eqx.tree_at(lambda x: x.covariance, pruned_mvn, pruned_mvn_cov)
    # Need to update precision/nat params based on new cov/loc if necessary

    # Combine pruned MVN with original Gamma (psi) part
    pruned_q_w_psi = eqx.tree_at(lambda x: x.mvn, model.q_w_psi, pruned_mvn)

    # Update model with pruned components
    model = eqx.tree_at(lambda x: x.q_w_psi, model, pruned_q_w_psi)
    model = eqx.tree_at(lambda x: x.q_tau, model, pruned_q_tau)
    # Update n_components count
    model = eqx.tree_at(lambda x: x.n_components, model, len(remaining_components_idx))

    return model
