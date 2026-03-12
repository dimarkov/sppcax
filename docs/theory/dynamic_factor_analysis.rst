=====================================
Bayesian Dynamic Factor Analysis
=====================================

Introduction
============

The primary model implemented in ``sppcax`` is the Bayesian Dynamic Factor Analysis (DFA) model,
a Linear Gaussian State Space Model (LGSSM) with hierarchical priors for automatic sparsification.
The model places Automatic Relevance Determination (ARD) priors on both the emission matrix
:math:`\mathbf{H}` and the dynamics matrix :math:`\mathbf{F}`, enabling data-driven discovery
of sparse latent representations and temporal dependencies.

Factor Analysis (FA) and Probabilistic PCA (PPCA) are recovered as special cases of DFA when
there are no temporal dynamics — see :ref:`sec-fa-pca-special-cases` below and :doc:`factor_analysis`
for detailed update equations.


.. _sec-dfa-model:

Model Definition
================

The Dynamic Factor Analysis model is defined by:

**State equation:**

.. math::
   :label: state-eq

   \mathbf{z}_t = \mathbf{F}\mathbf{z}_{t-1} + \mathbf{B}\mathbf{u}_t + \mathbf{b} + \boldsymbol{\eta}_t, \quad
   \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})

**Observation equation:**

.. math::
   :label: obs-eq

   \mathbf{x}_t = \mathbf{H}\mathbf{z}_t + \mathbf{D}\mathbf{u}_t + \mathbf{d} + \boldsymbol{\epsilon}_t, \quad
   \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}^{-1})

where:

- :math:`\mathbf{z}_t \in \mathbb{R}^K` is the latent state
- :math:`\mathbf{x}_t \in \mathbb{R}^D` is the observation
- :math:`\mathbf{u}_t \in \mathbb{R}^U` is an optional control input
- :math:`\mathbf{F} \in \mathbb{R}^{K \times K}` is the state dynamics matrix
- :math:`\mathbf{H} \in \mathbb{R}^{D \times K}` is the loading (emission) matrix
- :math:`\mathbf{B} \in \mathbb{R}^{K \times U}` and :math:`\mathbf{D} \in \mathbb{R}^{D \times U}` are input weight matrices
- :math:`\mathbf{b} \in \mathbb{R}^K` and :math:`\mathbf{d} \in \mathbb{R}^D` are bias terms
- :math:`\boldsymbol{\Psi}` is the observation noise precision (diagonal)
- :math:`\mathbf{Q}` is the state noise covariance (fixed to :math:`\mathbf{I}` by default)

.. note::

   The bias terms :math:`\mathbf{b}` and :math:`\mathbf{d}` are **absorbed** into the augmented
   weight matrices by appending a constant input of 1. The augmented emission matrix is
   :math:`\tilde{\mathbf{H}} = [\mathbf{H}, \mathbf{D}, \mathbf{d}]` and the augmented dynamics
   matrix is :math:`\tilde{\mathbf{F}} = [\mathbf{F}, \mathbf{B}, \mathbf{b}]`. The biases are
   learned jointly with the weight matrices under their respective priors, not as separate parameters.

**Initial state:**

.. math::

   \mathbf{z}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)


Prior Distributions
===================

Emission Matrix and Noise
--------------------------

The emission parameters :math:`(\mathbf{H}, \boldsymbol{\Psi})` have a Multivariate Normal-Inverse Gamma
(MVNIG) prior. For each row :math:`d` of the augmented emission matrix
:math:`\tilde{\mathbf{h}}_d = [\mathbf{h}_d, \mathbf{d}^u_d, d_d]` (loadings, input weights, bias):

.. math::

   p(\tilde{\mathbf{h}}_d, \psi_d) = \mathcal{N}(\tilde{\mathbf{h}}_d \mid \mathbf{0}, \psi_d^{-1} \tilde{\boldsymbol{\Sigma}}_d^0) \,
   \text{Gamma}(\psi_d \mid \alpha_0^\psi, \beta_0^\psi)

See :doc:`factor_analysis` for the detailed static-case prior specification.

**Noise precision** has two variants:

1. **Diagonal noise** (FA): :math:`\boldsymbol{\Psi} = \text{diag}(\psi_1, \ldots, \psi_D)` with independent :math:`\psi_d \sim \text{Gamma}(\alpha_0^\psi, \beta_0^\psi)`
2. **Isotropic noise** (PCA): :math:`\boldsymbol{\Psi} = \psi \mathbf{I}` with a shared :math:`\psi \sim \text{Gamma}(\alpha_0^\psi, \beta_0^\psi)`


Dynamics Matrix
----------------

The dynamics parameters :math:`(\mathbf{F}, \mathbf{Q})` have a Multivariate Normal (MVN) prior
with :math:`\mathbf{Q} = \mathbf{I}` fixed. For each row :math:`k` of the augmented dynamics matrix
:math:`\tilde{\mathbf{f}}_k = [\mathbf{f}_k, \mathbf{b}^u_k, b_k]` (dynamics weights, input weights, bias):

.. math::

   p(\tilde{\mathbf{f}}_k) = \mathcal{N}(\tilde{\mathbf{f}}_k \mid \mathbf{0}, \tilde{\boldsymbol{\Sigma}}_k^{0,F})


.. _sec-ard-priors:

ARD Priors
-----------

Automatic Relevance Determination (ARD) places column-wise Gamma priors on both the emission
and dynamics matrices, encouraging removal of unnecessary latent dimensions:

.. math::

   p(\tau_k^H) &= \text{Gamma}(\tau_k^H \mid \alpha_0^\tau, \beta_0^\tau) \\
   p(\tau_k^F) &= \text{Gamma}(\tau_k^F \mid \alpha_0^\tau, \beta_0^\tau)

with defaults :math:`\alpha_0^\tau = \beta_0^\tau = 0.5`. The expected value :math:`\mathbb{E}[\tau_k]`
is incorporated into the prior precision of the :math:`k`-th column of :math:`\mathbf{H}` (or :math:`\mathbf{F}`),
such that large :math:`\tau_k` shrinks the entire column towards zero.

For the emission matrix, the ARD-augmented prior precision for column :math:`k` becomes:

.. math::

   \tilde{\boldsymbol{\Sigma}}_d^{0,-1} \leftarrow \tilde{\boldsymbol{\Sigma}}_d^{0,-1} + \text{diag}(\mathbb{E}[\tau_1^H], \ldots, \mathbb{E}[\tau_K^H], 0, \ldots)

and analogously for the dynamics matrix with :math:`\tau_k^F`.


Initial State
--------------

The initial state distribution has a Normal-Inverse Wishart (NIW) prior:

.. math::

   p(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) = \text{NIW}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0 \mid
   \mathbf{m}_0, \kappa_0, \nu_0, \mathbf{S}_0)


Variational Inference
=====================

We use variational Bayes EM (VB-EM) to approximate the posterior. The factorized approximate posterior is:

.. math::

   p(\mathbf{Z}, \mathbf{H}, \boldsymbol{\Psi}, \mathbf{F}, \boldsymbol{\tau}^H, \boldsymbol{\tau}^F \mid \mathbf{X})
   \approx q(\mathbf{Z})\,q(\mathbf{H}, \boldsymbol{\Psi})\,q(\mathbf{F})\,q(\boldsymbol{\tau}^H)\,q(\boldsymbol{\tau}^F)

VBE-Step
--------

The variational E-step updates :math:`q(\mathbf{Z})` via Kalman smoothing using the expected parameters
from the other variational factors. In the VB setting, the Kalman filter incorporates correction terms
that account for the posterior uncertainty in model parameters :math:`\mathbf{H}` and :math:`\mathbf{F}`.

Specifically, the predicted state covariance is corrected as:

.. math::

   \tilde{\boldsymbol{\Sigma}}_{t|t-1} = (\mathbf{I} + \boldsymbol{\Sigma}_{t|t-1} \mathbf{C}^{xx})^{-1} \boldsymbol{\Sigma}_{t|t-1}

where :math:`\mathbf{C}^{xx} = \sum_d \text{correction from } q(\mathbf{H}, \boldsymbol{\Psi})` captures the
parameter uncertainty. See :doc:`factor_analysis` for the detailed static-case update equations.

VBM-Step
--------

The variational M-step updates each factor in turn:

- :math:`q(\mathbf{H}, \boldsymbol{\Psi})`: MVNIG posterior update using emission sufficient statistics
- :math:`q(\mathbf{F})`: MVN posterior update using dynamics sufficient statistics
  :math:`\mathbf{S}_{zz} = \sum_t \langle \mathbf{z}_{t-1} \mathbf{z}_{t-1}^\top \rangle` and
  :math:`\mathbf{S}_{z'z} = \sum_t \langle \mathbf{z}_t \mathbf{z}_{t-1}^\top \rangle`
- :math:`q(\boldsymbol{\tau}^H)`: Gamma posterior update with natural parameter increments

  .. math::

     \Delta \eta_1^{\tau_k} &= \frac{1}{2} \sum_{d: \lambda_{dk}=1} 1 \\
     \Delta \eta_2^{\tau_k} &= -\frac{1}{2} \sum_d \mathbb{E}_q[\psi_d] \left(\sigma_{dk}^2 + \mu_{dk}^2\right)

- :math:`q(\boldsymbol{\tau}^F)`: analogous Gamma update using dynamics posterior statistics

ELBO
----

The Evidence Lower Bound for the full DFA model is:

.. math::

   \mathcal{L} &= \mathbb{E}_q[\log p(\mathbf{X} \mid \mathbf{Z}, \mathbf{H}, \boldsymbol{\Psi})]
   + \mathbb{E}_q[\log p(\mathbf{Z} \mid \mathbf{F})] \\
   &- \text{KL}(q(\mathbf{H}, \boldsymbol{\Psi}) \| p(\mathbf{H}, \boldsymbol{\Psi} \mid \boldsymbol{\tau}^H))
   - \text{KL}(q(\mathbf{F}) \| p(\mathbf{F} \mid \boldsymbol{\tau}^F)) \\
   &- \text{KL}(q(\boldsymbol{\tau}^H) \| p(\boldsymbol{\tau}^H))
   - \text{KL}(q(\boldsymbol{\tau}^F) \| p(\boldsymbol{\tau}^F)) \\
   &- \text{KL}(q(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \| p(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0))

The expected log-likelihood terms include VB correction terms from the parameter uncertainty.


.. _sec-fa-pca-special-cases:

Factor Analysis and PCA as Special Cases
=========================================

Factor Analysis (FA) and Probabilistic PCA are obtained from DFA by removing the temporal dynamics:

- Set :math:`\mathbf{F} = \mathbf{0}`, :math:`\mathbf{Q} = \mathbf{I}`, :math:`\mathbf{b} = \mathbf{0}`
- Each latent variable :math:`\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` independently
- Data is reshaped from :math:`(\text{N}, \text{D})` to :math:`(\text{N}, 1, \text{D})` — each observation becomes
  an independent single-timestep sequence, so the Kalman smoother reduces to a single filter step

The generative model simplifies to:

.. math::

   \mathbf{x}_n = \tilde{\mathbf{H}}\tilde{\mathbf{z}}_n + \boldsymbol{\epsilon}_n, \quad
   \boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}^{-1})

where :math:`\tilde{\mathbf{z}}_n = [\mathbf{z}_n^\top, 1]^\top` and
:math:`\tilde{\mathbf{H}} = [\mathbf{H}, \mathbf{d}]` is the augmented loading matrix
with the bias absorbed as its last column.

**PCA** additionally constrains the noise to be isotropic: :math:`\boldsymbol{\Psi} = \psi \mathbf{I}`.

.. note::

   In ``sppcax``, the initial distribution :math:`\mathbf{z}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)`
   is **not updated by default** for FA and PCA. The prior is set so that :math:`\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`
   independently for each observation. This can be changed by providing a custom ``initial_prior`` and setting the
   initial distribution to be trainable.

For the detailed VB-EM update equations specific to FA, see :doc:`factor_analysis`.


Training Pipeline
==================

``sppcax`` provides three inference methods — EM, VBEM, and Blocked Gibbs Sampling — all
sharing the same M-step pipeline. Each M-step proceeds as:

1. **Compute posteriors**: Apply ARD to priors, then update posteriors from sufficient statistics
2. **PX-VB rotation** (if enabled): Find rotation :math:`\mathbf{R}` that minimises the expected
   negative log-prior, and remap posteriors (see :doc:`parameter_expansion`)
3. **BMR pruning** (if enabled, after a burn-in period): Prune loading matrix elements
   via Bayesian Model Reduction (see :doc:`model_reduction`)
4. **KL divergence**: Compute KL terms for ELBO
5. **ARD updates**: Update :math:`q(\boldsymbol{\tau}^H)` and :math:`q(\boldsymbol{\tau}^F)` from posteriors
6. **Extract parameters**: Mode (EM), moments (VB-EM), or samples (Gibbs) from posteriors

The methods differ only in the E-step: EM uses a standard Kalman smoother, VBEM uses a
VB-corrected Kalman smoother that accounts for parameter uncertainty, and Gibbs sampling
uses forward filtering backward sampling (FFBS). See :doc:`inference_methods` for details.

For practical examples comparing EM, VB-EM, PX-VB, and BMR settings, see the
:doc:`examples/index` section.


References
==========

1. Luttinen, J., Raiko, T., & Ilin, A. (2014). Linear state-space model with time-varying dynamics.
   In *Machine Learning and Knowledge Discovery in Databases* (ECML PKDD 2014), LNCS 8725, pp. 338-353.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Attias, H. (1999). Inferring parameters and structure of latent variable models by variational Bayes.
   In *Proceedings of the Fifteenth conference on Uncertainty in Artificial Intelligence*.
4. Zhao, J. H., and Philip, L. H. (2009). A note on variational Bayesian factor analysis.
   *Neural Networks*, 22(7), 988-997.
