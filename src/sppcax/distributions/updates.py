"""Distribution posterior update and extraction routines.

Functions for computing posterior updates from sufficient statistics,
and extracting point estimates (mode, moments, samples) from posterior distributions.
"""

import jax.numpy as jnp
import equinox as eqx
from multipledispatch import dispatch

from dynamax.utils.distributions import (
    mniw_posterior_update,
    niw_posterior_update,
    MatrixNormalInverseWishart,
    NormalInverseWishart,
)

from sppcax.distributions import Distribution, MultivariateNormal
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma, mvnig_posterior_update
from sppcax.distributions.utils import cho_inv
from sppcax.distributions.delta import Delta
from sppcax.metrics.kl_divergence import multidigamma, digamma


def _to_distribution(X):
    """Convert input to a Distribution if it isn't already."""
    if isinstance(X, Distribution):
        return X
    return Delta(X)


# ---------------------------------------------------------------------------
# Posterior updates via multiple dispatch
# ---------------------------------------------------------------------------


@dispatch(NormalInverseWishart, tuple, object)
def posterior_update(dist, stats, props):
    """Update NIW posterior from sufficient statistics."""
    if not (props.mean.trainable and props.cov.trainable):
        return dist
    return niw_posterior_update(dist, stats)


@dispatch(MatrixNormalInverseWishart, tuple, object)
def posterior_update(dist, stats, props):  # noqa: F811
    """Update MNIW posterior from sufficient statistics."""
    if not (props.weights.trainable and props.cov.trainable):
        return dist
    return mniw_posterior_update(dist, stats)


@dispatch(MultivariateNormalInverseGamma, tuple, object)
def posterior_update(dist, stats, props):  # noqa: F811
    """Update MVNIG posterior from sufficient statistics."""
    if not (props.weights.trainable and props.cov.trainable):
        return dist
    return mvnig_posterior_update(dist, stats, props)


@dispatch(MultivariateNormal, tuple, object)
def posterior_update(dist, stats, props):  # noqa: F811
    """Update MVN posterior from sufficient statistics.

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


# ---------------------------------------------------------------------------
# Point estimate extraction via multiple dispatch
# ---------------------------------------------------------------------------


# -- get_mode --

@dispatch(MultivariateNormal)
def get_mode(dist):
    """Mode of MVN distribution (Q=I fixed)."""
    mean = dist.mean
    covariance = jnp.eye(mean.shape[-2])
    return covariance, mean


@dispatch(NormalInverseWishart)
def get_mode(dist):  # noqa: F811
    """Mode of NIW distribution."""
    return dist.mode()


@dispatch(MatrixNormalInverseWishart)
def get_mode(dist):  # noqa: F811
    """Mode of MNIW distribution."""
    return dist.mode()


@dispatch(MultivariateNormalInverseGamma)
def get_mode(dist):  # noqa: F811
    """Mode of MVNIG distribution."""
    return dist.mode()


# -- get_sample --

@dispatch(MultivariateNormal, object)
def get_sample(dist, key):
    """Sample from MVN distribution (Q=I fixed)."""
    mean = dist.sample(key)
    covariance = jnp.eye(mean.shape[-2])
    return covariance, mean


@dispatch(NormalInverseWishart, object)
def get_sample(dist, key):  # noqa: F811
    """Sample from NIW distribution."""
    return dist.sample(seed=key)


@dispatch(MatrixNormalInverseWishart, object)
def get_sample(dist, key):  # noqa: F811
    """Sample from MNIW distribution."""
    return dist.sample(seed=key)


@dispatch(MultivariateNormalInverseGamma, object)
def get_sample(dist, key):  # noqa: F811
    """Sample from MVNIG distribution."""
    return dist.sample(seed=key)


# -- get_moments --

@dispatch(NormalInverseWishart)
def get_moments(dist):
    """Expected moments of NIW distribution."""
    covariance = jnp.einsum("...,...ij->...ij", 1 / dist.df, dist.scale)
    return covariance, dist.loc


@dispatch(MatrixNormalInverseWishart)
def get_moments(dist):  # noqa: F811
    """Expected moments of MNIW distribution."""
    covariance = jnp.einsum("...,...ij->...ij", 1 / dist.df, dist.scale)
    return covariance, dist.loc


@dispatch(MultivariateNormalInverseGamma)
def get_moments(dist):  # noqa: F811
    """Expected moments of MVNIG distribution."""
    mean = dist.mean
    covariance = jnp.diag(dist.expected_psi)
    return covariance, mean


@dispatch(MultivariateNormal)
def get_moments(dist):  # noqa: F811
    """Expected moments of MVN distribution (Q=I fixed)."""
    mean = dist.mean
    covariance = jnp.eye(mean.shape[-2])
    return covariance, mean


# -- get_ll_correction --

@dispatch(MatrixNormalInverseWishart)
def get_ll_correction(dist):
    """Log-likelihood correction for MNIW distribution."""
    dim, _ = dist._matrix_normal_shape
    x = dist.df / 2
    return (multidigamma(x, dim) - dim * jnp.log(x)) / 2


@dispatch(MultivariateNormalInverseGamma)
def get_ll_correction(dist):  # noqa: F811
    """Log-likelihood correction for MVNIG distribution."""
    alpha = dist.alpha
    return jnp.sum(digamma(alpha) - jnp.log(alpha)) / 2


@dispatch(MultivariateNormal)
def get_ll_correction(dist):  # noqa: F811
    """Log-likelihood correction for MVN distribution."""
    return 0.0


# -- get_correction --

@dispatch(MatrixNormalInverseWishart)
def get_correction(dist):
    """Posterior correction term for MNIW distribution."""
    dim, _ = dist._matrix_normal_shape
    col_precision = dist.col_precision
    return jnp.broadcast_to(cho_inv(col_precision), (dim,) + col_precision.shape)


@dispatch(MultivariateNormalInverseGamma)
def get_correction(dist):  # noqa: F811
    """Posterior correction term for MVNIG distribution."""
    return dist.col_covariance


@dispatch(MultivariateNormal)
def get_correction(dist):  # noqa: F811
    """Posterior correction term for MVN distribution."""
    return dist.covariance
