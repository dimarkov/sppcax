"""Dynamic Factor Analysis parameter containers with variational posteriors."""

from typing import Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..distributions.mvn_gamma import MultivariateNormalInverseGamma
from ..types import Array, Matrix, PRNGKey
from .factor_analysis_params import PFA, PPCA


class DynamicFactorAnalysisParams(eqx.Module):
    """Parameter container for Dynamic Factor Analysis models with variational posteriors."""

    n_samples: int
    n_timesteps: int
    likelihood: Union[PFA, PPCA]
    transition: PFA

    def __init__(
        self,
        n_samples: int,
        n_timesteps: int,
        n_components: int,
        n_features: int,
        n_controls: int = 0,
        control_latents: bool = False,
        update_ard: bool = True,
        isotropic_noise: bool = False,
        *,
        key: PRNGKey,
    ):
        """Initialize DynamicFactorAnalysis model parameters.

        Args:
            n_samples: Number of samples (N)
            n_timesteps: Number of time steps (T)
            n_components: Number of latent components (K)
            n_features: Number of observed features (D)
            n_contorls: Number of control/input variables (U)
            isotropic_noise: If True, use same noise precision R for all features
            control_latents: Apply control to latents (True) or observation (False).
            update_ard: Bool variable regyulating if we are updating the ARD prior or not
            isotropic_noise: Specify type of noise for the likelihood.
            key: A `jax.random.PRNGKey` for initialization.
        """
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps

        key_lkl, key_trns, key_cntrl = jr.split(key, 3)

        latent_control = n_controls if control_latents else 0
        t_model = PFA(
            n_components, n_components, n_controls=latent_control, use_bias=False, update_ard=update_ard, key=key_trns
        )
        self.transition = eqx.tree_at(
            lambda m: (m.q_w_psi.gamma.nat1_0, m.q_w_psi.gamma.nat2_0),
            t_model,
            (1e2 * t_model.q_w_psi.gamma.nat1_0, 1e1 * t_model.q_w_psi.gamma.nat2_0),
        )

        obs_control = n_controls if not control_latents else 0
        if isotropic_noise:
            self.likelihood = PPCA(
                n_components, n_features, n_controls=obs_control, use_bias=True, update_ard=update_ard, key=key_lkl
            )
        else:
            self.likelihood = PFA(
                n_components, n_features, n_controls=obs_control, use_bias=True, update_ard=update_ard, key=key_lkl
            )

    # --- Expected Parameter Properties ---

    @property
    def q_c_r(self) -> MultivariateNormalInverseGamma:
        return self.likelihood.q_w_psi

    @property
    def q_a_q(self) -> MultivariateNormalInverseGamma:
        return self.transition.q_w_psi

    @property
    def expected_A(self) -> Matrix:
        """E[A]"""
        return self.q_a_q.mvn.mean

    @property
    def expected_A_T(self) -> Matrix:
        """E[A^T]"""
        return self.q_a_q.mvn.mean.mT

    @property
    def expected_AAT(self) -> Array:
        """E[AA^T]"""
        EA = self.expected_A
        return self.q_a_q.mvn.covariance + EA[..., None] * EA[..., None, :]

    @property
    def expected_Q_inv(self) -> Matrix:
        """E[Q^-1]. Assumes diagonal Q."""
        return jnp.diag(self.q_a_q.expected_psi)

    @property
    def expected_log_Q_det(self) -> Array:
        """E[log|Q|]. Assumes diagonal Q."""
        # log|Q| = -log|Q^-1| = -sum(log Q_ii^-1)
        return -jnp.sum(self.q_a_q.expected_log_psi)

    @property
    def expected_A_T_Q_inv_A(self) -> Matrix:
        """E[A^T Q^-1 A]. Assumes diagonal Q."""
        Eq_inv_diag = self.q_a_q.expected_psi
        EAAT = self.expected_AAT  # Shape (k, k, k)
        return jnp.sum(Eq_inv_diag[..., None, None] * EAAT, axis=0)

    @property
    def expected_A_T_Q_inv(self) -> Matrix:
        """E[A^T Q^-1]. Assumes diagonal Q."""
        return self.expected_A_T @ self.expected_Q_inv

    @property
    def expected_C(self) -> Matrix:
        """E[C]"""
        return self.q_c_r.mvn.mean

    @property
    def expected_C_T(self) -> Matrix:
        """E[C^T]"""
        return self.q_c_r.mvn.mean.T

    @property
    def expected_CCT(self) -> Array:
        """E[CC^T]"""
        EC = self.expected_C
        return self.q_c_r.mvn.covariance + EC[..., None] * EC[..., None, :]

    @property
    def expected_R_inv(self) -> Matrix:
        """E[R^-1]. Assumes diagonal R."""
        # Handle isotropic case where expected_psi is scalar
        return jnp.diag(self.q_c_r.expected_psi)

    @property
    def expected_log_R_det(self) -> Array:
        """E[log|R|]. Assumes diagonal R."""
        return -jnp.sum(self.q_c_r.expected_log_psi)

    @property
    def expected_C_T_R_inv_C(self) -> Matrix:
        """E[C^T R^-1 C]. Assumes diagonal R."""
        Er_inv_diag = self.q_c_r.expected_psi
        ECCT = self.expected_CCT  # Shape (k, k, k)
        return jnp.sum(Er_inv_diag[..., None, None] * ECCT, axis=0)

    @property
    def expected_C_T_R_inv(self) -> Matrix:
        """E[C^T R^-1]. Assumes diagonal R."""
        return self.q_c_r.mvn.mean.T @ self.expected_R_inv  # Use expected_R_inv property
