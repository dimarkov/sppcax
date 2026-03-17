"""Gamma distribution implementation."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import equinox as eqx

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily


class Gamma(ExponentialFamily):
    """Gamma distribution in natural parameters.

    The gamma distribution has density:
    p(x|α,β) = β^α * x^(α-1) * exp(-βx) / Γ(α)

    In exponential family form:
    h(x) = 1
    η = [α-1, -β]
    T(x) = [log(x), x]
    A(η) = log(Γ(η₁ + 1)) - (η₁ + 1)*log(-η₂)
    """

    nat1_0: Array  # prior value of the first naural parameter (α0 - 1)
    nat2_0: Array  # prior value of the second natural parameter (-β0)
    dnat1: Array  # Change in the first natural parameter (α-1)
    dnat2: Array  # Change in the second natural parameter (-β)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha0: float | Array = 1.0, beta0: float | Array = 1.0):
        """Initialize gamma distribution with shape and rate parameters.

        Args:
            alpha0: Shape parameter α (default: 1.0)
            beta0: Rate parameter β (default: 1.0)
        """
        # Convert to arrays
        shape = jnp.asarray(alpha0)
        rate = jnp.asarray(beta0)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        self.nat1_0 = jnp.broadcast_to(shape - 1, self.batch_shape)  # α-1
        self.nat2_0 = jnp.broadcast_to(-rate, self.batch_shape)  # -β

        # Broadcast parameters
        self.dnat1 = jnp.zeros(self.batch_shape)
        self.dnat2 = jnp.zeros(self.batch_shape)

    @property
    def nat1(self) -> Array:
        """First natural parameter η₁ = α - 1."""
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        """Second natural parameter η₂ = -β."""
        return self.nat2_0 + self.dnat2

    @property
    def alpha(self) -> Array:
        """Get shape parameter α."""
        return self.nat1 + 1.0

    @property
    def beta(self) -> Array:
        """Get rate parameter β."""
        return -self.nat2

    @property
    def mean(self) -> Array:
        """Get mean E[x] = α/β."""
        return self.alpha / self.beta

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [α-1, -β].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [log(x), x].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [log(x), x] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([jnp.log(x), x], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [ψ(α) - log(β), α/β].

        Returns:
            Expected sufficient statistics [E[log(x)], E[x]] with shape:
            batch_shape + (2,)
        """
        alpha = self.alpha
        beta = self.beta
        return jnp.stack([jsp.digamma(alpha) - jnp.log(beta), alpha / beta], axis=-1)  # E[log(x)]  # E[x]

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(Γ(η₁ + 1)) - (η₁ + 1)*log(-η₂).

        Returns:
            Log normalizer with shape: batch_shape
        """
        alpha = self.alpha
        beta = self.beta
        return jsp.gammaln(alpha) - alpha * jnp.log(beta)

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values.
        """
        return x > 0

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            zero
        """
        return jnp.zeros(())

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        return jr.gamma(key, self.alpha, shape=shape) / self.beta

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "Gamma":
        """Create gamma distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            Gamma distribution.
        """
        shape = eta[..., 0] + 1.0  # α = η₁ + 1
        rate = -eta[..., 1]  # β = -η₂

        return cls(alpha0=shape, beta0=rate)

    @property
    def kl_divergence_from_prior(self) -> Array:
        """Compute KL divergence KL(post||prior).

        Returns:
            KL divergence KL(post||prior) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = jnp.stack([self.nat1_0, self.nat2_0], axis=-1)
        alpha = eta_other[..., 0] + 1
        beta = -eta_other[..., 1]
        other_log_normalizer = jsp.gammaln(alpha) - alpha * jnp.log(beta)
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other_log_normalizer + inner_product


class InverseGamma(ExponentialFamily):
    """Inverse Gamma distribution in natural parameters.

    The inverse gamma distribution has density:
    p(x|α,β) = β^α * x^(-α-1) * exp(-β/x) / Γ(α)

    In exponential family form:
    h(x) = 1
    η = [-α-1, -β]
    T(x) = [log(x), 1/x]
    A(η) = log(Γ(-η₁ - 1)) + (η₁ + 1)*log(-η₂)
    """

    nat1_0: Array  # prior value of the first natural parameter (-α0 - 1)
    nat2_0: Array  # prior value of the second natural parameter (-β0)
    dnat1: Array  # Change in the first natural parameter (-α-1)
    dnat2: Array  # Change in the second natural parameter (-β)
    natural_param_shape: ClassVar[Shape] = (2,)  # [η₁, η₂]

    def __init__(self, alpha0: float | Array = 1.0, beta0: float | Array = 1.0):
        """Initialize inverse gamma distribution with shape and scale parameters.

        Args:
            alpha0: Shape parameter α (default: 1.0)
            beta0: Scale parameter β (default: 1.0)
        """
        # Convert to arrays
        shape = jnp.asarray(alpha0)
        rate = jnp.asarray(beta0)

        # Set shapes
        batch_shape = jnp.broadcast_shapes(jnp.shape(shape), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, event_shape=())

        # Convert to natural parameters
        self.nat1_0 = -shape - 1  # -α-1
        self.nat2_0 = jnp.broadcast_to(-rate, self.batch_shape)  # -β

        # Broadcast parameters
        self.dnat1 = jnp.zeros(())
        self.dnat2 = jnp.zeros(self.batch_shape)

    @property
    def nat1(self) -> Array:
        """First natural parameter η₁ = -α - 1."""
        return self.nat1_0 + self.dnat1

    @property
    def nat2(self) -> Array:
        """Second natural parameter η₂ = -β."""
        return self.nat2_0 + self.dnat2

    @property
    def alpha(self) -> Array:
        """Get shape parameter α."""
        return -self.nat1 - 1.0

    @property
    def beta(self) -> Array:
        """Get scale parameter β."""
        return -self.nat2

    @property
    def mean(self) -> Array:
        """Get mean E[x] = β/(α-1)."""
        return self.beta / (self.alpha - 1)

    def mode(self) -> Array:
        return self.beta / (self.alpha + 1)

    @property
    def expected_psi(self) -> Array:
        """Expected precision E[1/x] = alpha/beta for InverseGamma."""
        return self.alpha / self.beta

    @property
    def natural_parameters(self) -> Array:
        """Get natural parameters η = [-α-1, -β].

        Returns:
            Natural parameters [η₁, η₂] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([self.nat1, self.nat2], axis=-1)

    def sufficient_statistics(self, x: Array) -> Array:
        """Compute sufficient statistics T(x) = [log(x), 1/x].

        Args:
            x: Value to compute sufficient statistics for.
               Shape: batch_shape + event_shape

        Returns:
            Sufficient statistics [log(x), 1/x] with shape:
            batch_shape + (2,)
        """
        return jnp.stack([jnp.log(x), 1 / x], axis=-1)

    @property
    def expected_sufficient_statistics(self) -> Array:
        """Compute E[T(x)] = [log(β) - ψ(α), α/β].

        Returns:
            Expected sufficient statistics [E[log(x)], E[1/x]] with shape:
            batch_shape + (2,)
        """
        alpha = self.alpha
        beta = self.beta
        return jnp.stack([jnp.log(beta) - jsp.digamma(alpha), alpha / beta], axis=-1)

    @property
    def log_normalizer(self) -> Array:
        """Compute log normalizer A(η) = log(Γ(-η₁ - 1)) + (η₁ + 1)*log(-η₂)

        Returns:
            Log normalizer with shape: batch_shape
        """
        alpha = self.alpha
        beta = self.beta
        return jsp.gammaln(alpha) - alpha * jnp.log(beta)

    def _check_support(self, x: Array) -> Array:
        """Check if values are within distribution support.

        Args:
            x: Values to check.
               Shape: batch_shape + event_shape

        Returns:
            Boolean mask of valid values.
        """
        return x > 0

    def log_base_measure(self, x: Array = None) -> Array:
        """Compute log of base measure h(x).

        Args:
            x: Data to compute base measure for.
               Shape: batch_shape + event_shape

        Returns:
            zero
        """
        return jnp.zeros(())

    def sample(self, key: PRNGKey, sample_shape: Shape = ()) -> Array:
        """Sample from the distribution.

        Args:
            key: PRNG key for random sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape: sample_shape + batch_shape + event_shape
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        return self.beta / jr.gamma(key, self.alpha, shape=shape)

    @classmethod
    def from_natural_parameters(cls, eta: Array) -> "InverseGamma":
        """Create inverse gamma distribution from natural parameters.

        Args:
            eta: Natural parameters [η₁, η₂] with shape:
                batch_shape + (2,)

        Returns:
            InverseGamma distribution.
        """
        shape = -(eta[..., 0] + 1.0)  # α = - (η₁ + 1)
        rate = -eta[..., 1]  # β = -η₂

        return cls(alpha0=shape, beta0=rate)

    @property
    def kl_divergence_from_prior(self) -> Array:
        """Compute KL divergence KL(post||prior).

        Returns:
            KL divergence KL(post||prior) with shape: batch_shape
        """
        eta_self = self.natural_parameters
        eta_other = jnp.stack([self.nat1_0, self.nat2_0], axis=-1)
        alpha = -(eta_other[..., 0] + 1)
        beta = -eta_other[..., 1]
        other_log_normalizer = jsp.gammaln(alpha) - alpha * jnp.log(beta)
        expected_T = self.expected_sufficient_statistics

        # Sum over natural parameter dimensions
        inner_product = jnp.sum(
            (eta_self - eta_other) * expected_T, axis=tuple(range(-len(self.natural_param_shape), 0))
        )

        return -self.log_normalizer + other_log_normalizer + inner_product

    def mf_expectations(self) -> dict:
        """Return expectations for mean-field coordinate ascent partner."""
        return {
            "expected_precision": self.expected_psi,
        }

    def mf_update(self, stats: tuple, partner_expectations: dict) -> "InverseGamma":
        """Mean-field coordinate ascent update for InverseGamma noise.

        Given partner (weights) expectations, compute the residual and update
        the InverseGamma natural parameters.

        Args:
            stats: Sufficient statistics tuple (SxxT, SxyT, SyyT, N).
            partner_expectations: Dict with 'mean' and 'second_moment' from weights.

        Returns:
            Updated InverseGamma distribution.
        """
        SxxT, SxyT, SyyT, N = stats

        M = partner_expectations["mean"]  # (D, dim)
        E_wwT = partner_expectations["second_moment"]  # (D, dim, dim)

        # SyyT diagonal: (D,) or (D, D)
        SyyT_diag = jnp.diag(SyyT) if SyyT.ndim == 2 else SyyT  # (D,)

        # Cross term: sum_d M_d^T SxyT_d = einsum('di,id->d', M, SxyT)
        cross_term = jnp.einsum("di,id->d", M, SxyT)  # (D,)

        # Trace term: tr(E[w_d w_d^T] @ SxxT) per row
        if SxxT.ndim == 2:
            tr_term = jnp.einsum("dij,ji->d", E_wwT, SxxT)  # (D,)
        else:
            tr_term = jnp.einsum("dij,dji->d", E_wwT, SxxT)  # (D,)

        # Residual: E[sum_t (y_dt - w_d^T z_t)^2]
        residual = SyyT_diag - 2 * cross_term + tr_term  # (D,)

        dnat1 = -N / 2
        dnat2 = -residual / 2

        # For isotropic noise (scalar batch), sum across features
        if self.batch_shape == ():
            D = dnat2.shape[0]
            dnat1 = dnat1 * D if jnp.ndim(dnat1) == 0 else dnat1.sum()
            dnat2 = dnat2.sum()

        return eqx.tree_at(lambda m: (m.dnat1, m.dnat2), self, (dnat1, dnat2))
