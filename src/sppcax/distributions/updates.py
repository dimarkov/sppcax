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

from sppcax.distributions import Distribution, MultivariateNormal, MeanField
from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma, mvnig_posterior_update
from sppcax.distributions.utils import cho_inv
from sppcax.distributions.delta import Delta
from sppcax.distributions.inverse_wishart import InverseWishart
from sppcax.distributions.gamma import InverseGamma
from sppcax.metrics.kl_divergence import multidigamma, digamma


def _to_distribution(X: object) -> Distribution:
    """Convert input to a Distribution if it isn't already.

    Args:
        X: A Distribution instance or an array to wrap in a Delta distribution.

    Returns:
        The input unchanged if already a Distribution, otherwise Delta(X).
    """
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

    # unpack the sufficient statistics
    SxxT, SxyT, *_ = stats

    # Data contributions only — prior preserved in nat1_0/nat2_0
    dnat2 = -0.5 * SxxT
    dnat1 = SxyT.mT

    return eqx.tree_at(lambda d: (d.dnat1, d.dnat2), dist, (dnat1, dnat2))


# ---------------------------------------------------------------------------
# Point estimate extraction via multiple dispatch
# ---------------------------------------------------------------------------


# -- get_mode --


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


@dispatch(MeanField)
def get_mode(dist):  # noqa: F811
    """Mode of MeanField distribution."""
    return dist.mode()


# -- get_sample --


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


@dispatch(MeanField, object)
def get_sample(dist, key):  # noqa: F811
    """Sample from MeanField distribution."""
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
    psi = dist.expected_psi
    n = dist.mean.shape[0]
    covariance = jnp.broadcast_to(1.0 / psi, (n,))
    return covariance, mean


@dispatch(MeanField)
def get_moments(dist):  # noqa: F811
    """Expected moments of MeanField distribution."""
    mean = dist.mean
    if isinstance(dist.noise, Delta):
        cov = dist.noise.mean
        if hasattr(cov, "ndim") and cov.ndim < 2:
            n = dist.batch_shape[0] if dist.batch_shape else 1
            cov = jnp.broadcast_to(cov, (n,))
        covariance = cov
    elif isinstance(dist.noise, InverseGamma):
        psi = dist.expected_psi
        n = dist.mean.shape[0]
        covariance = jnp.broadcast_to(1.0 / psi, (n,))
    else:
        # InverseWishart: E[Sigma^{-1}]^-1 = scale / df
        covariance = dist.noise.scale / dist.noise.df
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


@dispatch(MeanField)
def get_ll_correction(dist):  # noqa: F811
    """Log-likelihood correction for MVN distribution."""
    if isinstance(dist.noise, Delta):
        return 0.0
    elif isinstance(dist.noise, InverseGamma):
        alpha = dist.noise.alpha
        return jnp.sum(digamma(alpha) - jnp.log(alpha)) / 2
    elif isinstance(dist.noise, InverseWishart):
        dim, _ = dist.weights.shape
        x = dist.noise.df / 2
        return (multidigamma(x, dim) - dim * jnp.log(x)) / 2
    else:
        raise NotImplementedError


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


@dispatch(MeanField)
def get_correction(dist):  # noqa: F811
    """Posterior correction term for Mean-Field distribution.

    Unlike MVNIG/MNIW where conjugate coupling cancels the noise scaling,
    MeanField components are independent, so the correction must include
    E[ψ] (expected noise precision) multiplied by the weight covariance.
    """
    E_psi = dist.expected_psi  # scalar, (D,), or (D, D)
    V = dist.weights.covariance  # (D, dim, dim) or (dim, dim)

    # For matrix precision (IW noise), extract diagonal for per-row scaling
    if E_psi.ndim >= 2:
        E_psi = jnp.diag(E_psi)  # (D,)

    return V * E_psi[..., None, None]


# ---------------------------------------------------------------------------
# MeanField dispatches
# ---------------------------------------------------------------------------


@dispatch(MeanField, tuple, object)
def posterior_update(dist, stats, props):  # noqa: F811
    """Update MeanField posterior via coordinate ascent.

    Alternates between updating the weights component (using noise expectations)
    and the noise component (using weights expectations).
    Delta components are no-ops, so trainability is implicit in the distribution type.
    """
    weights = dist.weights
    noise = dist.noise

    # Single coordinate ascent step: update weights then noise.
    # Delta.mf_update is a no-op, so no explicit trainability check needed.
    noise_exp = noise.mf_expectations()
    weights = weights.mf_update(stats, noise_exp)

    weights_exp = weights.mf_expectations()
    noise = noise.mf_update(stats, weights_exp)

    return MeanField(weights=weights, noise=noise)
