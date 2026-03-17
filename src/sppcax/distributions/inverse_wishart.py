"""Inverse Wishart distribution with mean-field interface.

Wraps the dynamax InverseWishart with natural parameters and
coordinate ascent update methods for use in MeanField composites.
"""

from typing import ClassVar

import jax.numpy as jnp
from jax.scipy.special import multigammaln, digamma
from jax.scipy.linalg import cho_solve
import equinox as eqx

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily
from .utils import cho_inv, safe_cholesky_and_logdet


def multidigamma(a: Array, p: int) -> Array:
    """Multivariate digamma: psi_p(a) = sum_{i=1}^{p} psi(a + (1-i)/2)."""
    k = (1 - jnp.arange(1, p + 1)) / 2
    return jnp.sum(digamma(jnp.expand_dims(a, -1) + k), axis=-1)


class InverseWishart(ExponentialFamily):
    """Inverse Wishart distribution in natural parameters.

    For Sigma ~ IW(df, Psi):
        p(Sigma | df, Psi) propto |Sigma|^{-(df+k+1)/2} exp(-tr(Psi Sigma^{-1})/2)

    Exponential family form:
        eta1 = -(df + k + 1) / 2       (scalar)
        eta2 = -Psi / 2                 (k x k matrix)
        T(Sigma) = [log|Sigma|, Sigma^{-1}]
        A(eta) = -(df/2) log|Psi/2| + multigammaln(df/2, k) + (df*k/2) log(2)

    Key moments:
        E[Sigma^{-1}] = df * Psi^{-1}
        E[log|Sigma^{-1}|] = multidigamma(df/2, k) + k*log(2) - log|Psi|
    """

    nat1_0: Array  # prior: -(df0 + k + 1) / 2
    nat2_0: Array  # prior: -Psi0 / 2
    dnat1: Array  # learned delta for nat1
    dnat2: Array  # learned delta for nat2
    _dim: int
    natural_param_shape: ClassVar[Shape] = ()  # complex structure, handle manually

    def __init__(self, df0: float | Array, scale0: Array):
        """Initialize InverseWishart distribution.

        Args:
            df0: Degrees of freedom (scalar).
            scale0: Scale matrix Psi with shape (k, k).
        """
        df0 = jnp.asarray(df0, dtype=float)
        scale0 = jnp.asarray(scale0, dtype=float)
        k = scale0.shape[-1]

        batch_shape = scale0.shape[:-2]
        super().__init__(batch_shape=batch_shape, event_shape=(k, k))

        self._dim = k
        self.nat1_0 = -(df0 + k + 1) / 2
        self.nat2_0 = -scale0 / 2
        self.dnat1 = jnp.zeros_like(self.nat1_0)
        self.dnat2 = jnp.zeros_like(self.nat2_0)

    @property
    def nat1(self) -> Array:
        """First natural parameter: -(df + k + 1) / 2."""
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        """Second natural parameter: -Psi / 2."""
        return self.nat2_0 + self.dnat2

    @property
    def df(self) -> Array:
        """Degrees of freedom: df = -2*nat1 - k - 1."""
        return -2 * self.nat1 - self._dim - 1

    @property
    def scale(self) -> Array:
        """Scale matrix Psi = -2 * nat2."""
        return -2 * self.nat2

    @property
    def inv_scale(self) -> Array:
        """Inverse scale matrix Psi^{-1}."""
        return cho_inv(self.scale)

    @property
    def dim(self) -> int:
        """Dimension k of the k x k matrices."""
        return self._dim

    @property
    def mean(self) -> Array:
        """Mean E[Sigma] = Psi / (df - k - 1), requires df > k + 1."""
        return self.scale / (self.df - self._dim - 1)

    def mode(self) -> Array:
        """Mode = Psi / (df + k + 1)."""
        return self.scale / (self.df + self._dim + 1)

    # --- Exponential family interface ---

    @property
    def natural_parameters(self) -> tuple:
        """Natural parameters (eta1, eta2)."""
        return (self.nat1, self.nat2)

    def sufficient_statistics(self, x: Array) -> tuple:
        """T(Sigma) = (log|Sigma|, Sigma^{-1})."""
        L, logdet = safe_cholesky_and_logdet(x)
        identity = jnp.eye(x.shape[-1], dtype=x.dtype)
        return (logdet, cho_solve((L, True), jnp.broadcast_to(identity, x.shape)))

    @property
    def expected_sufficient_statistics(self) -> tuple:
        """E[T(Sigma)] = (E[log|Sigma|], E[Sigma^{-1}])."""
        k = self._dim
        # E[log|Sigma|] = -E[log|Sigma^{-1}|] = -(multidigamma(df/2, k) + k*ln(2) - ln|Psi|)
        _, log_det_scale = safe_cholesky_and_logdet(self.scale)
        E_log_det = -multidigamma(self.df / 2, k) - k * jnp.log(2.0) + log_det_scale
        E_inv = self.expected_psi
        return (E_log_det, E_inv)

    @property
    def log_normalizer(self) -> Array:
        """Log normalizer A(df, Psi)."""
        k = self._dim
        df = self.df
        _, log_det_scale = safe_cholesky_and_logdet(self.scale)
        return multigammaln(df / 2, k) - (df / 2) * log_det_scale + (df * k / 2) * jnp.log(2.0)

    def log_base_measure(self, x: Array = None) -> Array:
        return jnp.zeros(())

    def _check_support(self, x: Array) -> Array:
        # Check positive definite via Cholesky
        try:
            jnp.linalg.cholesky(x)
            return jnp.ones(x.shape[:-2], dtype=bool)
        except Exception:
            return jnp.zeros(x.shape[:-2], dtype=bool)

    def log_prob(self, x: Array) -> Array:
        """Log probability of x under IW(df, Psi)."""
        k = self._dim
        sign, logdet_x = jnp.linalg.slogdet(x)
        x_inv = jnp.linalg.inv(x)

        log_p = -(self.df + k + 1) / 2 * logdet_x
        log_p -= 0.5 * jnp.trace(self.scale @ x_inv, axis1=-2, axis2=-1)
        log_p -= self.log_normalizer
        return log_p

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from IW(df, Psi) via Bartlett decomposition."""
        import jax.random as jr

        k = self._dim
        df = self.df

        full_shape = sample_shape + self.batch_shape
        # Sample W ~ Wishart(df, Psi^{-1}) then Sigma = W^{-1}
        L_inv = jnp.linalg.cholesky(self.inv_scale)

        # Bartlett decomposition: A is lower triangular
        # A_ii ~ chi(df - i + 1), A_ij ~ N(0,1) for j < i
        keys = jr.split(key, 2)
        # Chi-squared diagonal
        chi_sq = jnp.stack([jr.gamma(keys[0], (df - i) / 2, shape=full_shape) * 2 for i in range(k)], axis=-1)
        diag = jnp.sqrt(chi_sq)

        # Off-diagonal standard normals
        A = jnp.zeros(full_shape + (k, k))
        # Set diagonal
        A = A.at[..., jnp.arange(k), jnp.arange(k)].set(diag)
        # Set lower triangular
        if k > 1:
            tril_indices = jnp.tril_indices(k, -1)
            n_tril = len(tril_indices[0])
            normals = jr.normal(keys[1], shape=full_shape + (n_tril,))
            A = A.at[..., tril_indices[0], tril_indices[1]].set(normals)

        # W = L_inv @ A @ A^T @ L_inv^T (Wishart sample)
        LA = L_inv @ A
        W = LA @ LA.mT
        # Sigma = W^{-1}
        return jnp.linalg.inv(W)

    @property
    def entropy(self) -> Array:
        """Entropy of IW(df, Psi)."""
        k = self._dim
        df = self.df
        sign, log_det_scale = jnp.linalg.slogdet(self.scale)
        return (
            self.log_normalizer
            + (df + k + 1) / 2 * (-multidigamma(df / 2, k) - k * jnp.log(2.0) + log_det_scale)
            + df * k / 2
        )

    # --- Mean-field interface ---

    @property
    def expected_psi(self) -> Array:
        """E[Sigma^{-1}] = df * Psi^{-1}."""
        return self.df * self.inv_scale

    def mf_expectations(self) -> dict:
        """Return expectations for mean-field coordinate ascent partner."""
        return {
            "expected_precision": self.expected_psi,
        }

    def mf_update(self, stats: tuple, partner_expectations: dict) -> "InverseWishart":
        """Mean-field coordinate ascent update for InverseWishart noise.

        For IW noise in a linear model y = W x + e, e ~ N(0, Sigma):
            df_post = df_prior + N
            Psi_post = Psi_prior + E[sum_t (y_t - W x_t)(y_t - W x_t)^T]
                     = Psi_prior + SyyT - SxyT^T E[W]^T - E[W] SxyT + E[WW^T] SxxT

        The quadratic term E[W SxxT W^T] decomposes under mean-field q(W) = prod_d N(w_d; m_d, V_d):
            off-diag (i!=j): m_i^T SxxT m_j   (independent rows)
            diag (i=i): tr(E[w_i w_i^T] SxxT) = m_i^T SxxT m_i + tr(V_i SxxT)
        """
        SxxT, SxyT, SyyT, N = stats

        M = partner_expectations["mean"]  # (D, dim)
        E_wwT = partner_expectations["second_moment"]  # (D, dim, dim)

        cross = M @ SxyT  # (D, D)
        quad = M @ SxxT @ M.mT  # (D, D) — correct off-diagonal

        # Replace diagonal: tr(E[w_d w_d^T] @ SxxT) instead of m_d^T SxxT m_d
        diag_quad = jnp.einsum("dij,ji->d", E_wwT, SxxT)  # (D,)
        quad = quad - jnp.diag(jnp.diag(quad)) + jnp.diag(diag_quad)

        residual = SyyT - cross - cross.mT + quad

        dnat1 = -N / 2
        dnat2 = -residual / 2

        return eqx.tree_at(lambda m: (m.dnat1, m.dnat2), self, (dnat1, dnat2))

    @property
    def kl_divergence_from_prior(self) -> Array:
        """KL(posterior || prior) using stored prior natural parameters."""
        k = self._dim

        df_q = self.df
        df_p = -2 * self.nat1_0 - k - 1  # prior df
        scale_q = self.scale
        scale_p = -2 * self.nat2_0  # prior scale

        inv_scale_q = jnp.linalg.inv(scale_q)
        P = scale_p @ inv_scale_q

        kl = -df_p / 2 * jnp.linalg.slogdet(P)[1]
        kl += df_q / 2 * (jnp.trace(P, axis1=-1, axis2=-2) - k)
        kl += multigammaln(df_p / 2, k) - multigammaln(df_q / 2, k)
        kl += (df_q - df_p) / 2 * multidigamma(df_q / 2, k)

        return kl
