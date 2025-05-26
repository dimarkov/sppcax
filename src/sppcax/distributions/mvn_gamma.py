"""Multivariate Normal-Gamma distribution implementation."""

from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import equinox as eqx

from ..types import Array, PRNGKey, Shape
from .exponential_family import ExponentialFamily
from .gamma import InverseGamma, Gamma
from .mvn import MultivariateNormal


class MultivariateNormalInverseGamma(ExponentialFamily):
    """Multivariate Normal-Gamma distribution.

    This distribution combines:
    p(x|σ²) = N(x; μ, σ²Λ⁻¹)  # Multivariate Normal
    p(σ²) = InverseGamma(σ²; α, β)  # Variance scalar

    where:
    - x is a vector (can be batched)
    - μ is the location parameter
    - Λ is a base precision matrix
    - σ² is a scalar variance parameter
    - α, β are Inverse-Gamma distribution parameters
    \sigma^2
    """

    mvn: MultivariateNormal
    inv_gamma: InverseGamma

    def __init__(
        self,
        loc: Array,
        *,
        isotropic_noise,
        mask: Optional[Array] = None,
        alpha0: float = 2.0,
        beta0: float = 1.0,
        scale_tril: Optional[Array] = None,
        covariance: Optional[Array] = None,
        precision: Optional[Array] = None,
    ):
        """Initialize MultivariateNormalGamma distribution.

        Args:
            loc: Location parameter
            mask: Optional boolean mask for active dimensions
            alpha: Shape parameter for Gamma prior
            beta: Rate parameter for Gamma prior
            scale_tril: Optional scale matrix (lower triangular)
            covariance: Optional covariance matrix
            precision: Optional precision matrix

        Note:
            Only one of scale_tril, covariance, or precision should be provided.
            If none are provided, identity matrix is used.
        """
        # Initialize MVN distribution
        self.mvn = MultivariateNormal(
            loc=loc, mask=mask, scale_tril=scale_tril, covariance=covariance, precision=precision
        )

        # Initialize Gamma parameters
        if isotropic_noise:
            self.inv_gamma = InverseGamma(alpha0=alpha0, beta0=beta0)
        else:
            self.inv_gamma = InverseGamma(
                alpha0=alpha0 * jnp.ones(self.mvn.batch_shape), beta0=beta0 * jnp.ones(self.mvn.batch_shape)
            )

        # Set shapes from MVN-Gamma
        super().__init__(batch_shape=self.mvn.batch_shape, event_shape=self.mvn.event_shape)

    def log_prob(self, x: Tuple[Array, Array]) -> Array:
        """Compute log probability.

        Args:
            x: Tuple of (sig_sqr, w) where:
                w: Value of the sample state
                sig_sqr: Value of the sample variance

        Returns:
            Log probability
        """
        sig_sqr, w = x
        sig_sqr = jnp.diagonal(sig_sqr, axis1=-1, axis2=-2) if sig_sqr.ndim == w.ndim else sig_sqr

        # MVN term: p(w|psi)
        mvn = eqx.tree_at(
            lambda x: (x.nat1, x.nat2),
            self.mvn,
            (self.mvn.nat1 / sig_sqr[..., None], self.mvn.nat2 / sig_sqr[..., None, None]),
        )
        mvn_log_prob = mvn.log_prob(w).sum()

        # Gamma term: p(psi)
        inv_gamma_log_prob = self.inv_gamma.log_prob(sig_sqr).sum()

        return mvn_log_prob + inv_gamma_log_prob

    def sample(self, seed: PRNGKey, sample_shape: Shape = ()) -> Tuple[Array, Array]:
        """Sample from the distribution.

        Args:
            seed: PRNG key
            sample_shape: Shape of samples to draw

        Returns:
            Tuple of (sig_sqr, value) samples
        """
        key_g, key_mvn = jr.split(seed)

        # Sample psi ~ Gamma(α, β)
        sig_sqr = self.inv_gamma.sample(key_g, sample_shape=sample_shape)
        sig = jnp.sqrt(sig_sqr)

        # Sample x|σ² ~ MVN(μ, σ²Λ⁻¹)
        # We can sample from base MVN and scale by sqrt(σ²)
        mvn = eqx.tree_at(lambda x: x.nat1, self.mvn, self.mvn.nat1 / sig[..., None])
        value = mvn.sample(key_mvn, sample_shape=sample_shape) * sig[..., None]

        return sig_sqr * jnp.eye(value.shape[-2]), value

    def mode(self):
        r"""Solve for the mode. Recall,
        ..math::
            p(\mu, \sigma^2) \propto
                \mathrm{N}(x \mid \mu, \sigma^2 \Sigma) \times
                \mathrm{IG}(\sigma^2 \mid \alpha, \beta)
        The optimal mean is :math:`x^* = \mu_0`. Substituting this in,
        ..math::
            p(\mu^*, \sigma^2) \propto IG(\sigma^2 \mid \alpha + D/2, \beta)
        where D is dimensionality of x, and the mode of this inverse gamma
        distribution is at
        ..math::
            (\sigma^2)* = \beta / (\alpha + (D + 2)/2)
        """
        dim = self.event_shape[-1]
        return jnp.eye(len(self.beta)) * (self.beta / (self.alpha + (dim + 2) / 2)), self.mean

    @property
    def alpha(self) -> Array:
        return self.inv_gamma.alpha

    @property
    def beta(self) -> Array:
        return self.inv_gamma.beta

    @property
    def precision(self) -> Array:
        return self.mvn.precision

    @property
    def mean(self) -> Array:
        """Get mean of the marginal distribution p(x)."""
        return self.mvn.mean

    @property
    def col_covariance(self) -> Array:
        return self.mvn.covariance

    @property
    def expected_covariance(self) -> Array:
        # mean of the inverse of precision (variance)
        exp_variance = jnp.broadcast_to(self.inv_gamma.mean, self.batch_shape)
        return self.mvn.covariance * exp_variance[..., None, None]

    @property
    def expected_psi(self) -> Array:
        """Compute expected precision E[psi]."""
        return jnp.broadcast_to(self.inv_gamma.alpha / self.inv_gamma.beta, self.batch_shape)

    @property
    def expected_log_psi(self) -> Array:
        """Compute expected log precision E[log(psi)]."""
        return jnp.broadcast_to(jsp.digamma(self.inv_gamma.alpha) - jnp.log(self.inv_gamma.beta), self.batch_shape)

    @property
    def expected_sufficient_statistics_psi(self) -> Array:
        """Compute expected sufficient statistics of psi."""
        gamma = Gamma(self.alpha, self.beta)
        suff_stats = gamma.expected_sufficient_statistics
        return jnp.broadcast_to(suff_stats, self.batch_shape + (2,))


def mvnig_posterior_update(mvnig_prior: MultivariateNormalInverseGamma, sufficient_stats: tuple, props: dict):
    """Update the multivaraite normal inverse gamma (MVNIG) distribution using sufficient statistics

    Returns:
        posterior MVNIG distribution
    """
    # extract parameters of the prior distribution
    mvn_prior = mvnig_prior.mvn

    nat1 = mvn_prior.nat1
    prior_precision = -2.0 * mvn_prior.nat2

    # unpack the sufficient statistics
    SxxT, SxyT, SyyT, N = sufficient_stats

    # compute parameters of the posterior distribution
    nat2_post = -0.5 * (prior_precision + SxxT)
    nat1_post = mvn_prior.apply_mask_vector(nat1 + SxyT.mT)
    Syy = SyyT + mvn_prior.mean @ nat1.mT

    mvn_post = eqx.tree_at(lambda m: (m.nat1, m.nat2), mvnig_prior.mvn, (nat1_post, nat2_post))
    M_pos = mvn_post.mean

    dnat1 = -N / 2
    dnat2 = -(jnp.diag(Syy - M_pos @ SxyT)) / 2

    inv_gamma_post = eqx.tree_at(lambda m: (m.dnat1, m.dnat2), mvnig_prior.inv_gamma, (dnat1, dnat2))
    mvnig_post = eqx.tree_at(lambda m: (m.mvn, m.inv_gamma), mvnig_prior, (mvn_post, inv_gamma_post))

    return mvnig_post
