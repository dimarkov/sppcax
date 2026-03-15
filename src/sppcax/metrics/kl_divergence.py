"""KL divergence computations for various distribution pairs via multiple dispatch."""

import jax.numpy as jnp
from jax.scipy.special import digamma, multigammaln
from multipledispatch import dispatch

from sppcax.distributions import (
    MultivariateNormalInverseGamma as MVNIG,
    InverseGamma,
    MultivariateNormal as MVN,
    MeanField,
)
from sppcax.distributions.delta import Delta
from sppcax.types import Array, Scalar
from dynamax.utils.distributions import NormalInverseWishart as NIW, MatrixNormalInverseWishart as MNIW, InverseWishart
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def multidigamma(a: Array, p: int) -> Array:
    """Compute the multivariate digamma function.

    ψ_p(a) = Σ_{i=1}^{p} ψ(a + (1-i)/2)

    Args:
        a: Input array.
        p: Dimension parameter.

    Returns:
        Multivariate digamma values with same shape as a.
    """
    k = (1 - jnp.arange(1, p + 1)) / 2
    return jnp.sum(digamma(jnp.expand_dims(a, -1) + k), axis=-1)


@dispatch(InverseWishart, InverseWishart)
def kl_divergence(q: InverseWishart, p: InverseWishart) -> Scalar:
    """Compute KL divergence KL(q||p) between two InverseWishart distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    df_q = q.df
    df_p = p.df

    inv_scale_q = q.inv_scale
    scale_p = p.scale

    k = inv_scale_q.shape[-1]

    P = scale_p @ inv_scale_q
    kl_div = -df_p / 2 * jnp.linalg.slogdet(P)[1]
    kl_div += df_q / 2 * (jnp.trace(P, axis1=-1, axis2=-2) - k)
    kl_div += multigammaln(df_p / 2, k) - multigammaln(df_q / 2, k)
    kl_div += (df_q - df_p) / 2 * multidigamma(df_q / 2, k)

    return kl_div


@dispatch(NIW, NIW)
def kl_divergence(q: NIW, p: NIW) -> Scalar:  # noqa: F811
    """Compute KL divergence KL(q||p) between two NormalInverseWishart distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    kl_div1 = kl_divergence(q.model[0], p.model[0])
    mc_q = q.mean_concentration
    mc_p = p.mean_concentration

    E_Precision = (q.model[0].inv_scale * q.model[0].df) * mc_p
    k = E_Precision.shape[-1]

    diff = p.loc - q.loc

    kl_div2 = 0.5 * (
        k * mc_p / mc_q + jnp.sum(diff * (E_Precision @ diff), -1) + k * (-1.0 + jnp.log(mc_q) - jnp.log(mc_p))
    )
    return kl_div1 + kl_div2


@dispatch(MNIW, MNIW)
def kl_divergence(q: MNIW, p: MNIW) -> Scalar:  # noqa: F811
    """Compute KL divergence KL(q||p) between two MatrixNormalInverseWishart distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    kl_div1 = kl_divergence(q.model[0], p.model[0])

    diff = p.loc - q.loc
    n, k = diff.shape
    V2_inv = p.col_precision

    E_Precision = q.model[0].inv_scale * q.model[0].df

    P = V2_inv @ q.col_covariance
    kl_div2 = 0.5 * jnp.sum(diff * (E_Precision @ diff @ V2_inv), axis=(-1, -2))
    kl_div2 += 0.5 * n * jnp.trace(P, axis1=-1, axis2=-2)
    kl_div2 -= 0.5 * (n * jnp.linalg.slogdet(P)[1] + k * jnp.log(n) + n * k)

    return kl_div1 + kl_div2


@dispatch(InverseGamma, InverseGamma)
def kl_divergence(q: InverseGamma, p: InverseGamma) -> Scalar:  # noqa: F811
    """Compute KL divergence KL(q||p) between two InverseGamma distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    return q.kl_divergence(p)


@dispatch(MVNIG, MVNIG)
def kl_divergence(q: MVNIG, p: MVNIG) -> Scalar:  # noqa: F811
    """Compute KL divergence KL(q||p) between two MVNIG distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    kl_div1 = q.inv_gamma.kl_divergence_from_prior.sum()

    diff = p.mean - q.mean
    n, k = diff.shape

    # shape (n, k, k)
    P_p = p.precision
    E_Precision = P_p * q.expected_psi[..., None, None]
    P = P_p @ q.col_covariance

    kl_div2 = 0.5 * (
        jnp.trace(P, axis1=-1, axis2=-2).sum()
        + jnp.sum(diff * (E_Precision @ diff[..., None]).squeeze(-1), axis=(-1, -2))
        - n * k
    )

    # todo: find a better way to compute logdet if covariance has zero diagonal values due to masking
    kl_div2 -= 0.5 * (jnp.linalg.slogdet(P_p)[1] - jnp.linalg.slogdet(q.precision)[1]).sum()

    return kl_div1 + kl_div2


@dispatch(MVN, MVN)
def kl_divergence(q: MVN, p: MVN) -> Scalar:  # noqa: F811
    """Compute KL divergence KL(q||p) between two MultivariateNormal distributions.

    Args:
        q: Variational (posterior) distribution.
        p: Reference (prior) distribution.

    Returns:
        KL divergence scalar.
    """
    diff = p.mean - q.mean
    n, k = diff.shape

    # shape (n, k, k)
    P_p = p.precision
    C_q = q.covariance
    P = P_p @ C_q

    kl_div = 0.5 * (
        jnp.trace(P, axis1=-1, axis2=-2).sum()
        + jnp.sum(diff * (P_p @ diff[..., None]).squeeze(-1), axis=(-1, -2))
        - n * k
    )

    # todo: find a better way to compute logdet if covariance has zero diagonal values due to masking
    kl_div -= 0.5 * (jnp.linalg.slogdet(P_p)[1] - jnp.linalg.slogdet(q.precision)[1]).sum()

    return kl_div


@dispatch(Delta, Delta)
def kl_divergence(q: Delta, p: Delta) -> Scalar:  # noqa: F811
    """KL divergence between two Delta distributions is always 0."""
    return 0.0


@dispatch(MeanField, MeanField)
def kl_divergence(q: MeanField, p: MeanField) -> Scalar:  # noqa: F811
    """KL divergence KL(q||p) between two MeanField distributions.

    Sum of component KL divergences (independence assumption).
    """
    kl_weights = kl_divergence(q.weights, p.weights)
    kl_noise = kl_divergence(q.noise, p.noise)
    # Reduce noise KL to scalar (e.g. per-feature InverseGamma returns (D,))
    if hasattr(kl_noise, "ndim") and kl_noise.ndim > 0:
        kl_noise = jnp.sum(kl_noise)
    return kl_weights + kl_noise
