=========================================
Inference Methods: EM, VBEM, and Gibbs
=========================================

Introduction
============

``sppcax`` implements three inference methods for the Dynamic Factor Analysis model
(see :doc:`dynamic_factor_analysis`): Expectation-Maximization (EM), Variational Bayes EM
(VBEM), and Blocked Gibbs Sampling. All three methods share the **same M-step** — full
Bayesian posterior updates via conjugate computations — and differ only in how the E-step
is computed and how parameters are extracted from the posteriors.

.. list-table:: Comparison of inference methods
   :header-rows: 1
   :widths: 20 35 25 20

   * - Method
     - E-step
     - Parameter extraction
     - Code
   * - **EM**
     - Standard Kalman smoother
     - Mode of posterior
     - ``fit_em()``
   * - **VBEM**
     - VB-corrected Kalman smoother
     - Moments of posterior
     - ``fit_vbem()``
   * - **Gibbs**
     - Forward filtering, backward sampling
     - Sample from posterior
     - ``fit_blocked_gibbs()``


Shared M-Step Pipeline
=======================

All three methods compute sufficient statistics from the E-step, then pass them through
a unified M-step pipeline (``_update_params_from_stats``):

1. **Compute posteriors** from sufficient statistics via conjugate updates
2. **PX-VB rotation** (optional): find rotation :math:`\mathbf{R}` and remap posteriors
   (see :doc:`parameter_expansion`)
3. **BMR pruning** (optional, after burn-in): prune loading matrix elements via
   Bayesian Model Reduction (see :doc:`model_reduction`)
4. **KL divergence**: compute :math:`\text{KL}(q \| p)` terms for the ELBO
5. **ARD updates**: update :math:`q(\boldsymbol{\tau}^H)` and :math:`q(\boldsymbol{\tau}^F)`
6. **Extract parameters** from posteriors — this is the only step that differs:

   - **EM**: extract the **mode** (MAP point estimate)
   - **VBEM**: extract the **moments** (mean and variance)
   - **Gibbs**: draw a **sample** from the posterior


EM (Maximum A Posteriori)
==========================

The EM algorithm finds parameters that maximize the marginal posterior:

.. math::

   \boldsymbol{\theta}^\star = \arg\max_{\boldsymbol{\theta}} \; \log p(\mathbf{X}, \boldsymbol{\theta})

E-Step
-------

The E-step uses a **standard Kalman smoother** with point-estimate parameters to compute
the posterior over latent states :math:`q(\mathbf{Z})`. No parameter uncertainty is
propagated — the filter treats current parameter values as exact.

The smoother produces:

.. math::

   \langle \mathbf{z}_t \rangle, \quad
   \text{Cov}(\mathbf{z}_t), \quad
   \text{Cov}(\mathbf{z}_t, \mathbf{z}_{t-1})

from which the sufficient statistics for the M-step are computed:

.. math::

   \mathbf{S}_{zz} &= \sum_t \langle \mathbf{z}_{t-1} \mathbf{z}_{t-1}^\top \rangle \\
   \mathbf{S}_{z'z} &= \sum_t \langle \mathbf{z}_t \mathbf{z}_{t-1}^\top \rangle \\
   \mathbf{S}_{xz} &= \sum_t \langle \mathbf{z}_t \rangle \mathbf{x}_t^\top

M-Step
-------

The M-step computes the full Bayesian posterior from the sufficient statistics, then extracts
the **mode** as the point estimate for the next iteration. The mode of the MVNIG emission
posterior and the MVN dynamics posterior provides the MAP parameter values.


VBEM (Variational Bayes EM)
=============================

VBEM approximates the full posterior over both latent states and model parameters using
a factorized variational distribution. The key difference from EM is that the E-step
accounts for **parameter uncertainty**.

E-Step
-------

The E-step uses a **VB-corrected Kalman filter/smoother** that incorporates correction
terms from the posterior uncertainty in :math:`\mathbf{H}` and :math:`\mathbf{F}`.

Specifically, the predicted state covariance is corrected as:

.. math::

   \tilde{\boldsymbol{\Sigma}}_{t|t-1} = (\mathbf{I} + \boldsymbol{\Sigma}_{t|t-1} \mathbf{C}^{xx})^{-1} \boldsymbol{\Sigma}_{t|t-1}

where the correction matrix :math:`\mathbf{C}^{xx}` captures the expected second moments
of model parameters beyond their means:

.. math::

   \mathbf{C}^{xx} = \sum_d \bar{\psi}_d \boldsymbol{\Sigma}_d^H + \text{(dynamics corrections)}

This correction inflates the state uncertainty to account for the fact that the model
parameters are not known exactly. Without it, the E-step would underestimate the
posterior variance of the latent states.

The corrected predicted mean is:

.. math::

   \tilde{\boldsymbol{\mu}}_{t|t-1} = (\mathbf{I} + \boldsymbol{\Sigma}_{t|t-1} \mathbf{C}^{xx})^{-1} \boldsymbol{\mu}_{t|t-1}

and a log-likelihood correction term is accumulated for the ELBO.

M-Step
-------

The M-step computes the full Bayesian posterior from the sufficient statistics (identical
conjugate updates as EM), then extracts the **moments** — the posterior mean and covariance.
These moments are used both as the parameters for the next E-step and to compute the
correction matrices :math:`\mathbf{C}^{xx}`.

For detailed static-case update equations, see :doc:`factor_analysis`.


Blocked Gibbs Sampling
========================

Gibbs sampling provides an alternative to variational inference by drawing samples
from the exact posterior distribution. In contrast to VB, which provides a locally
optimal deterministic approximation, Gibbs sampling generates a Markov chain whose
stationary distribution is the true posterior. The blocked variant groups
correlated variables and samples them jointly, reducing autocorrelation and
improving mixing.

E-Step: Forward Filtering Backward Sampling (FFBS)
-----------------------------------------------------

Instead of computing posterior expectations, the E-step **samples** latent states
from their conditional distribution given the current parameter samples.

**Forward pass** (Kalman filter):

.. math::

   p(\mathbf{z}_t | \mathbf{x}_{1:t}, \mathbf{H}, \boldsymbol{\Psi}, \mathbf{F}, \mathbf{Q})
   = \mathcal{N}(\mathbf{z}_t | \boldsymbol{\mu}_{t|t}, \boldsymbol{\Sigma}_{t|t})

using standard Kalman filter recursions with the current parameter samples.

**Backward pass** (sampling):

.. math::

   \mathbf{z}_t | \mathbf{z}_{t+1}, \mathbf{x}_{1:T} \sim
   \mathcal{N}(\boldsymbol{\mu}_{t|t,t+1}, \boldsymbol{\Sigma}_{t|t,t+1})

where:

.. math::

   \mathbf{G}_t &= \boldsymbol{\Sigma}_{t|t} \mathbf{F}^\top \boldsymbol{\Sigma}_{t+1|t}^{-1} \\
   \boldsymbol{\mu}_{t|t,t+1} &= \boldsymbol{\mu}_{t|t} + \mathbf{G}_t
   (\mathbf{z}_{t+1} - \mathbf{F}\boldsymbol{\mu}_{t|t} - \mathbf{b}) \\
   \boldsymbol{\Sigma}_{t|t,t+1} &= \boldsymbol{\Sigma}_{t|t} - \mathbf{G}_t
   \boldsymbol{\Sigma}_{t+1|t} \mathbf{G}_t^\top

The sufficient statistics are then computed directly from the sampled states (not
from expectations), e.g., :math:`\mathbf{S}_{zz} = \sum_t \mathbf{z}_{t-1} \mathbf{z}_{t-1}^\top`.

M-Step: Sampling from Conditional Posteriors
----------------------------------------------

The M-step draws **samples** from the full conditional posteriors of each parameter block.

**Sampling** :math:`(\mathbf{h}_d, \psi_d)` **(row-wise):**

For each row :math:`d` of the loading matrix, the conditional distribution of
:math:`(\mathbf{h}_d, \psi_d)` is a Normal-Gamma:

.. math::

   p(\mathbf{h}_d, \psi_d | \mathbf{Z}, \mathbf{X}, \boldsymbol{\tau}, \text{rest})
   = \mathcal{N}(\mathbf{h}_d | \boldsymbol{\mu}_d, \psi_d^{-1} \boldsymbol{\Sigma}_d)
   \, \text{Gamma}(\psi_d | \alpha_d, \beta_d)

where:

.. math::

   \boldsymbol{\Sigma}_d^{-1} &= \text{diag}(\boldsymbol{\tau}) + \sum_{t=1}^T \mathbf{z}_t \mathbf{z}_t^\top \\
   \boldsymbol{\mu}_d &= \boldsymbol{\Sigma}_d \sum_{t=1}^T \mathbf{z}_t x_{td} \\
   \alpha_d &= \alpha_0^\psi + \frac{T + K}{2} \\
   \beta_d &= \beta_0^\psi + \frac{1}{2} \sum_{t=1}^T (x_{td} - \mathbf{h}_d^\top \mathbf{z}_t)^2
   + \frac{1}{2} \sum_{k=1}^{K} \tau_k h_{dk}^2

Sampling proceeds by first drawing :math:`\psi_d \sim \text{Gamma}(\alpha_d, \beta_d)`,
then :math:`\mathbf{h}_d \sim \mathcal{N}(\boldsymbol{\mu}_d, \psi_d^{-1} \boldsymbol{\Sigma}_d)`.

**Sampling** :math:`\mathbf{F}`:

With the MVN prior used in ``sppcax`` (where :math:`\mathbf{Q} = \mathbf{I}` is fixed):

.. math::

   p(\mathbf{F} | \mathbf{Z}) = \mathcal{N}(\text{vec}(\mathbf{F}) |
   \boldsymbol{\mu}_F^{post}, \boldsymbol{\Sigma}_F^{post})

where:

.. math::

   \boldsymbol{\Sigma}_F^{post-1} &= \boldsymbol{\Sigma}_F^{prior-1} + \mathbf{S}_{zz} \otimes \mathbf{Q}^{-1} \\
   \boldsymbol{\mu}_F^{post} &= \boldsymbol{\Sigma}_F^{post} (\boldsymbol{\Sigma}_F^{prior-1} \boldsymbol{\mu}_F^{prior}
   + (\mathbf{Q}^{-1} \mathbf{S}_{z'z})_{\text{vec}})

**Sampling** :math:`\boldsymbol{\tau}` **(ARD):**

The column-wise ARD precisions have Gamma conditionals:

.. math::

   \hat{\alpha}_k &= \alpha_0^\tau + \frac{D}{2} \\
   \hat{\beta}_k &= \beta_0^\tau + \frac{1}{2} \sum_{d=1}^D \psi_d h_{dk}^2

**Sampling** :math:`\boldsymbol{\Lambda}` **(sparsity structure):**

When using Bayesian Model Reduction for sparsity (see :doc:`model_reduction`),
the sparsity indicators :math:`\lambda_{dk} \in \{0, 1\}` are sampled from
Bernoulli conditionals:

.. math::

   q(\lambda_{dk} = 1) = \sigma\left(-\Delta F_{dk} + \ln \frac{\pi}{1 - \pi}\right)

where :math:`\Delta F_{dk}` is the change in variational free energy when pruning
element :math:`(d, k)`, and :math:`\sigma(\cdot)` is the logistic function.

**Sampling** :math:`\pi` **(sparsity level):**

The inclusion probability :math:`\pi` has a Beta conditional:

.. math::

   a_t &= a_0 + \sum_{d,k} \lambda_{dk} \\
   b_t &= b_0 + \sum_{d,k} (1 - \lambda_{dk})


Algorithm Summaries
====================

.. rst-class:: algorithm

**Algorithm 1: EM for DFA**

| 1. Initialize parameters :math:`\boldsymbol{\theta}^{(0)}`
| 2. **repeat** until convergence:
|    a. **E-step:** Kalman smoother with :math:`\boldsymbol{\theta}^{(m)}` → sufficient statistics
|    b. **M-step:** Posterior update → PX rotation → BMR → ARD → extract **mode**
| 3. **return** MAP parameter estimates

.. rst-class:: algorithm

**Algorithm 2: VBEM for DFA**

| 1. Initialize approximate posteriors
| 2. **repeat** until convergence:
|    a. **VBE-step:** VB-corrected Kalman smoother → sufficient statistics
|    b. **VBM-step:** Posterior update → PX rotation → BMR → ARD → extract **moments**
|    c. Compute ELBO for convergence monitoring
| 3. **return** approximate posteriors

.. rst-class:: algorithm

**Algorithm 3: Blocked Gibbs Sampler for DFA**

| **Input:** :math:`T_{\text{burn-in}}`, :math:`T_{\text{max}}`
| **Output:** posterior samples :math:`\{\boldsymbol{\theta}^{(t)}\}_{t > T_{\text{burn-in}}}`
|
| 1. Initialize :math:`\boldsymbol{\theta}^{(0)}`
| 2. **for** :math:`t = 1, \ldots, T_{\text{max}}` **do**:
|    a. **FFBS:** Sample :math:`\mathbf{Z}^{(t)}` via forward filtering, backward sampling
|    b. Compute sufficient statistics from sampled states
|    c. **M-step:** Posterior update → PX rotation → BMR → ARD → **sample** parameters
|    d. **if** :math:`t > T_{\text{burn-in}}`: store sample :math:`\boldsymbol{\theta}^{(t)}`
| 3. **return** posterior samples


See Also
========

- :doc:`dynamic_factor_analysis` — model definition and priors
- :doc:`factor_analysis` — static-case VB-EM update equations
- :doc:`parameter_expansion` — PX-VB rotation details
- :doc:`model_reduction` — BMR pruning


References
==========

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Luttinen, J., Raiko, T., & Ilin, A. (2014). Linear state-space model with time-varying dynamics.
   In *Machine Learning and Knowledge Discovery in Databases* (ECML PKDD 2014), LNCS 8725, pp. 338-353.
