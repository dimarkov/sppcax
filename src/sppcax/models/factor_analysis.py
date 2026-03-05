"""Bayesian Factor Analysis and PCA as special cases of Dynamic Factor Analysis.

Factor Analysis (FA) corresponds to DFA with trivial dynamics: F=0, b=0, Q=I.
PCA additionally constrains the emission noise to be isotropic: R = sigma^2 I.

Data is reshaped from (N, D) to (N, 1, D) — N independent batches of T=1 —
so the Kalman smoother runs trivially (one filter step, no scan per batch)
and vmap over batches handles parallelism.
"""

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
from dynamax.utils.distributions import NormalInverseWishart as NIW

from sppcax.distributions.mvn_gamma import MultivariateNormalInverseGamma
from sppcax.types import Array, Matrix, PRNGKey
from sppcax.models.dynamic_factor_analysis import BayesianDynamicFactorAnalysis
from sppcax.models.utils import _make_mvnig_prior


class BayesianFactorAnalysis(BayesianDynamicFactorAnalysis):
    """Bayesian Factor Analysis as a special case of Dynamic Factor Analysis.

    This model is equivalent to DFA with trivial dynamics (F=0, Q=I), meaning
    each latent state z_t ~ N(0, I) independently. The generative model is:

        z_t ~ N(0, I)
        y_t = H z_t + d + noise,  noise ~ N(0, R)

    where R has per-feature noise precision (FA) or isotropic noise (PCA).

    Uses MultivariateNormalInverseGamma (MVNIG) emission prior for element-wise
    sparsification, and optionally an ARD (Automatic Relevance Determination)
    prior for column pruning.

    Args:
        n_components: Number of latent factors (state_dim).
        n_features: Number of observed features (emission_dim).
        input_dim: Dimensionality of control inputs. Defaults to 0.
        has_emissions_bias: Whether to include a bias term d. Defaults to True.
        has_ard: Whether to use ARD prior on emission weight columns. Defaults to True.
        use_bmr: Whether to enable Bayesian Model Reduction. Defaults to False.
        isotropic_noise: If True, constrain R = sigma^2 I (PCA). Defaults to False.
        key: Random key for initialization. Defaults to None.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        input_dim: int = 0,
        has_emissions_bias: bool = True,
        has_ard: bool = True,
        use_bmr: bool = False,
        isotropic_noise: bool = False,
        key: Optional[PRNGKey] = None,
        **kw_priors,
    ):
        # Create MVNIG emission prior if not provided
        if "emission_prior" not in kw_priors:
            kw_priors["emission_prior"] = _make_mvnig_prior(
                n_features, n_components, input_dim, has_bias=has_emissions_bias,
                isotropic_noise=isotropic_noise, key=key,
            )

        # Initial prior with mode (m=0, S=I) for z ~ N(0, I).
        # NIW mode: m = loc, S = scale / (df + dim + 2).
        # So scale = (df + dim + 2) * I gives S = I.
        if "initial_prior" not in kw_priors:
            df = n_components + 2.0
            kw_priors["initial_prior"] = NIW(
                loc=jnp.zeros(n_components),
                mean_concentration=1.0,
                df=df,
                scale=(df + n_components + 2) * jnp.eye(n_components),
            )

        super().__init__(
            state_dim=n_components,
            emission_dim=n_features,
            input_dim=input_dim,
            has_dynamics_bias=False,
            has_emissions_bias=has_emissions_bias,
            is_static=True,
            has_ard=has_ard,
            has_sparsity_prior=use_bmr,
            isotropic_noise=isotropic_noise,
            use_bmr=use_bmr,
            **kw_priors,
        )


class BayesianPCA(BayesianFactorAnalysis):
    """Bayesian PCA as Factor Analysis with isotropic emission noise.

    This model is equivalent to FA with R = sigma^2 I, where sigma^2 is
    a shared noise variance across all features.

    Args:
        n_components: Number of principal components (state_dim).
        n_features: Number of observed features (emission_dim).
        **kwargs: Additional keyword arguments passed to BayesianFactorAnalysis.
    """

    def __init__(self, n_components: int, n_features: int, **kwargs):
        super().__init__(n_components, n_features, isotropic_noise=True, **kwargs)
