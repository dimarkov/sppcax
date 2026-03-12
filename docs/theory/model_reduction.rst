============================
Bayesian Model Reduction
============================

Introduction
============

Bayesian Model Reduction (BMR) is a generalisation of the Savage-Dickey ratio allowing an efficient
comparison of a large number of generative models, all differing only in their specification of the
prior distribution. In the context of our Dynamic Factor Analysis implementation (and its FA/PCA
special cases), we use BMR to post-hoc prune parameters of the loading matrix :math:`\mathbf{H}`
and potentially the dynamics matrix :math:`\mathbf{F}`, removing redundant latent dimensions and
providing an efficient approach to learning sparse representations.

BMR is applied within the M-step pipeline **after** posterior computation and PX rotation, using the
ARD-augmented prior (where :math:`\mathbb{E}[\tau_k]` from the previous iteration's ARD posterior
is already incorporated into the prior precision). See :doc:`dynamic_factor_analysis` for the
full pipeline description.

This document provides the mathematical foundation for the BMR algorithm implemented in ``sppcax``.

Principle of Bayesian Model Reduction
=====================================

Bayesian model selections rests on evaluating and comparing model evidence of different generative models :math:`m_i`
given some data :math:`\mathbf{y}`. The model evidence is given by:

.. math::

   p(\mathbf{y}|m_i) = \int p(\mathbf{y}|\boldsymbol{\theta}, m_i) p(\boldsymbol{\theta}|m_i) d\boldsymbol{\theta}

where :math:`\boldsymbol{\theta}` are the model parameters, :math:`p(\mathbf{y}|\boldsymbol{\theta}, m_i)` is the
likelihood, and :math:`p(\boldsymbol{\theta}|m_i)` is the prior.

Instead of fitting each model to the data, we can use BMR and leverage the posterior distribution obtained from a
full model, :math:`m_0`, to analytically derive the evidence for each reduced model, :math:`m_{i>0}`. This is only
possible when the reduced models differ from the full model only in their prior specifications. Specifically, we
call reduced model any model that has a prior of lower entropy compared to the prior of the full model. Importantly,
this condition is satisfied for sparse Factor Analysis, as we are looking for a sparse representation of the loading
matrix, that corresponds to a large number of zero valued elements of the matrix.

Mathematical Formulation
========================

Consider a full model :math:`m_0` with prior :math:`p(\boldsymbol{\theta}|m_F)` and any reduced model :math:`m_{i>0}`
with prior :math:`p(\boldsymbol{\theta}|m_{i>0})`. If we know the posterior distribution of the full model
:math:`p(\boldsymbol{\theta}|\mathbf{y}, m_0)`, we can compute the evidence for reduced models as:

.. math::

   p(\mathbf{y}|m_i) = p(\mathbf{y}|m_0) \int d\boldsymbol{\theta}\frac{p(\boldsymbol{\theta}|m_i)}{p(\boldsymbol{\theta}|m_0)} p(\boldsymbol{\theta}|\mathbf{y}, m_0)

This allows us to compute the ratio of model evidences (Bayes factor) between the reduced and full models as:

.. math::

   \frac{p(\mathbf{y}|m_i)}{p(\mathbf{y}|m_0)} = \int \frac{p(\boldsymbol{\theta}|m_i)}{p(\boldsymbol{\theta}|m_0)} p(\boldsymbol{\theta}|\mathbf{y}, m_0) d\boldsymbol{\theta}

Variational Approximation
=========================

For Bayesian Factor Analysis model (or PCA), we use a variational approximation :math:`q_0(\boldsymbol{\theta})` to the true posterior :math:`p(\boldsymbol{\theta}|\mathbf{y}, m_0)`. Importantly,
we assume the same functional form for the posterior and the prior (see :ref:`sec-prior-dist` for details).
The variational free energy is given by:

.. math::

   F(m_0) = \int d\boldsymbol{\theta} q_0(\boldsymbol{\theta}) \ln \frac{q_0(\boldsymbol{\theta})}{p(\mathbf{y}, \boldsymbol{\theta}|m_0)}

where we assume that the log-joint :math:`\ln p(\mathbf{y}, \boldsymbol{\theta}|m_0)` is obtained during the e-step (see :ref:`e-step`) as:

.. math::
   \ln p(\mathbf{y}, \boldsymbol{\theta}|m_0) \approx \ln p(\pmb{\theta}|m_0) + \int d \pmb{z} q(\pmb{z}) \ln \frac{q(\pmb{z})}{p(\pmb{y}|\pmb{\theta}, \pmb{z})}



For any reduced model, the variational free energy can be computed as:

.. math::

   F(m_i) &= F(m_0) - \ln \int d\boldsymbol{\theta} q_0(\boldsymbol{\theta}) \frac{p(\boldsymbol{\theta}|m_i)}{p(\boldsymbol{\theta}|m_0)} \\
   &= F(m_0) - \Delta F_i

This formula allows us to compute the change in free energy (:math:`\Delta F`) when switching from the full model to a reduced model,
which is what we implemented in the ``compute_delta_f`` function.

Note that if we enumerate different models :math:`m_i` in a way that makes to neghbouring models :math:`m_i` and :math:`m_{i+1}` such that
they differ only in a single parameter, where :math:`m_{i+1}` is a reduced version of :math:`m_i` then we can express the variational free-energy
of model :math:`m_i` as a folowing sequence

.. math::
   F(m_{i-1}) &= F(m_{i}) + \Delta F_{i:i-1} \\
   F(m_0) &= F(m_i) + \sum_{k=1}^i \Delta F_{k:k-1} \\
   \Delta F_{k:k-1} &= \ln \int d \pmb{\theta} q_{k-1}(\theta) \frac{p(\pmb{\theta}|m_k)}{p(\pmb{\theta}|m_{k-1})}


Computing :math:`\Delta F`
===========================

In ``sppcax``, BMR is applied to both the loading matrix :math:`\tilde{\mathbf{H}}` (MVNIG posterior)
and the dynamics matrix :math:`\tilde{\mathbf{F}}` (MVN posterior). The prior precision used in both
cases already contains :math:`\mathbb{E}[\tau_k]` from the ARD posterior of the previous iteration,
so no separate ARD terms appear in the :math:`\Delta F` computation.

We define the following quantities for a given row with pruning mask :math:`\mathbf{g}` (1 = active, 0 = pruned):

.. math::

   \mathbf{G} &= \text{diag}(\mathbf{g}) \\
   \mathbf{L}_{\text{post}} &= \text{chol}(\mathbf{G} \boldsymbol{\Lambda}_{\text{post}} \mathbf{G} + \mathbf{I}_{\text{pruned}}) \\
   \mathbf{L}_{\text{prior}} &= \text{chol}(\mathbf{G} \boldsymbol{\Lambda}_{\text{prior}} \mathbf{G} + \mathbf{I}_{\text{pruned}}) \\
   \tilde{\boldsymbol{\mu}}_{\text{post}} &= \mathbf{L}_{\text{post}}^\top (\mathbf{g} \odot \boldsymbol{\mu}_{\text{post}}) \\
   \tilde{\boldsymbol{\mu}}_{\text{prior}} &= \mathbf{L}_{\text{prior}}^\top (\mathbf{g} \odot \boldsymbol{\mu}_{\text{prior}})

where :math:`\boldsymbol{\Lambda}_{\text{post}}` and :math:`\boldsymbol{\Lambda}_{\text{prior}}` are the
posterior and prior precision matrices, :math:`\boldsymbol{\mu}_{\text{post}}` and :math:`\boldsymbol{\mu}_{\text{prior}}`
are the posterior and prior means, and :math:`\mathbf{I}_{\text{pruned}} = \mathbf{I} - \mathbf{G}` fills in the pruned
dimensions to keep the Cholesky factorisation well-conditioned.


MVNIG Case (Loading Matrix)
----------------------------

The loading matrix has a joint MVNIG posterior (see :ref:`sec-post-dist`):

.. math::

   q(\tilde{\mathbf{h}}_d, \psi_d) = \mathcal{N}(\tilde{\mathbf{h}}_d \mid \boldsymbol{\mu}_d,
   \psi_d^{-1} \boldsymbol{\Sigma}_d) \, \text{Gamma}(\psi_d \mid \alpha_d, \beta_d)

The change in variational free energy for pruning a set of elements in row :math:`d` is:

.. math::

   \Delta F_d = \ln |\mathbf{L}_{\text{post}}| - \ln |\mathbf{L}_{\text{prior}}|
   + \alpha_d \ln \beta_d
   - \alpha_d \ln \left(\beta_d + \tfrac{1}{2} \tilde{\boldsymbol{\mu}}_{\text{post}}^\top \tilde{\boldsymbol{\mu}}_{\text{post}}
   - \tfrac{1}{2} \tilde{\boldsymbol{\mu}}_{\text{prior}}^\top \tilde{\boldsymbol{\mu}}_{\text{prior}} \right)

The :math:`\alpha_d \ln(\cdot)` terms arise from integrating over the shared noise precision
:math:`\psi_d` under the Gamma posterior. When the noise is isotropic (PCA), :math:`\alpha_d`
and :math:`\beta_d` are shared across all dimensions.


MVN Case (Dynamics Matrix)
---------------------------

The dynamics matrix has an MVN posterior without a noise precision parameter (since :math:`\mathbf{Q} = \mathbf{I}`
is fixed):

.. math::

   q(\tilde{\mathbf{f}}_k) = \mathcal{N}(\tilde{\mathbf{f}}_k \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)

The change in variational free energy simplifies to:

.. math::

   \Delta F_k = \ln |\mathbf{L}_{\text{post}}| - \ln |\mathbf{L}_{\text{prior}}|
   - \tfrac{1}{2} \tilde{\boldsymbol{\mu}}_{\text{post}}^\top \tilde{\boldsymbol{\mu}}_{\text{post}}
   + \tfrac{1}{2} \tilde{\boldsymbol{\mu}}_{\text{prior}}^\top \tilde{\boldsymbol{\mu}}_{\text{prior}}

This is a direct quadratic form — the absence of the :math:`\alpha \ln(\beta + \cdots)` terms reflects
the fact that there is no noise variance to integrate over.


Gibbs Sampling with Indian Buffet Process Prior
================================================

To determine the final sparse structure of the loading matrix (and dynamics matrix), we use
Gibbs sampling with a spike-and-slab prior on the sparsity matrix :math:`\boldsymbol{\Lambda}`.
If :math:`\lambda_{dk} = 1`, the element retains its normal prior; if :math:`\lambda_{dk} = 0`,
the prior becomes a delta function at zero, forcing the element to vanish.

Indian Buffet Process Prior
----------------------------

Each column :math:`k` of the sparsity matrix has an independent inclusion probability drawn from a
truncated Indian Buffet Process (IBP) prior:

.. math::

   \pi_k \sim \text{Beta}\!\left(\frac{\alpha_0}{K},\, 1\right)

where :math:`\alpha_0` is a concentration parameter controlling the expected number of active features,
and :math:`K` is the total number of latent dimensions. This prior encourages sparse representations
while allowing the data to determine which dimensions are needed.

Gibbs Sampling Procedure
-------------------------

The posterior probability of element :math:`\lambda_{dk} = 1` is:

.. math::

   q(\lambda_{dk} = 1) &= \frac{p(\mathbf{y}|m_{i-1})\,\pi_k}{p(\mathbf{y}|m_{i-1})\,\pi_k + p(\mathbf{y}|m_i)\,(1-\pi_k)} \\
   &= \sigma\!\left(\Delta F_{d,\, i:i-1} + \ln \frac{\pi_k}{1 - \pi_k}\right)

where :math:`\sigma(\cdot)` is the logistic function and :math:`\Delta F_{d,\, i:i-1}` is the change
in variational free energy between two models differing only in element :math:`(d, k)`.

At each Gibbs iteration :math:`t`:

1. **Sample sparsity columns**: For :math:`k = 1, \ldots, K`:

   .. math::

      \lambda_{dk}^{(t)} \sim \text{Bernoulli}\!\left(\sigma\!\left(\Delta F_{d,\, i:i-1} + \ln \frac{\pi_k}{1 - \pi_k}\right)\right) \cdot \text{mask}_{dk}

   where :math:`\text{mask}_{dk}` is the structural constraint from the model definition.

2. **Update column-wise hyperparameters**:

   .. math::

      \alpha_k &= \frac{\alpha_0}{K} + \sum_d \lambda_{dk}^{(t)} \\
      \beta_k &= 1 + \sum_d (1 - \lambda_{dk}^{(t)})

3. **Update concentration parameter** :math:`\alpha_0` via empirical Bayes:

   .. math::

      \alpha_0 = \frac{-K^2}{\sum_k \left[\psi(\alpha_k) - \psi(\alpha_k + \beta_k)\right]}

   where :math:`\psi(\cdot)` is the digamma function.


Posterior Correction After Pruning
===================================

After the Gibbs sampler converges with final mask :math:`\boldsymbol{\Lambda}`, the posterior
distributions are corrected to account for the pruned elements.

MVNIG (Emissions)
------------------

For the loading matrix posterior:

1. Pruned elements are set to zero in both the mean and natural parameters.

2. The Inverse Gamma :math:`\beta` parameter is corrected:

   .. math::

      \beta_d^{\text{new}} = \beta_d + \frac{1}{2}\left(
      -\boldsymbol{\mu}_{\text{pruned},d}^\top \boldsymbol{\Lambda}_{\text{post},d}\, \boldsymbol{\mu}_{\text{pruned},d}
      + \boldsymbol{\mu}_{\text{pruned},d}^{0\top} \boldsymbol{\Lambda}_{\text{prior},d}\, \boldsymbol{\mu}_{\text{pruned},d}^0
      \right)

   where :math:`\boldsymbol{\mu}_{\text{pruned},d}` denotes the pruned elements of the posterior mean
   and :math:`\boldsymbol{\mu}_{\text{pruned},d}^0` the pruned elements of the prior mean.

3. The Inverse Gamma scale is optimised:

   .. math::

      \text{nat2}_0 = \min\!\left(\frac{\Delta\text{nat2}}{\alpha - 1},\, \frac{-1}{\alpha - 1}\right)

MVN (Dynamics)
---------------

For the dynamics matrix posterior, pruned elements are set to zero via the updated mask.
No noise precision correction is needed since :math:`\mathbf{Q} = \mathbf{I}` is fixed.


See Also
========

For practical examples demonstrating BMR in action:

- :doc:`examples/test_px_em_fa` — FA with BMR for sparse loading matrix discovery
- :doc:`examples/test_px_em_dfa` — DFA with BMR for sparse structure discovery


References
==========

1. Friston, K., Mattout, J., Trujillo-Barreto, N., Ashburner, J., & Penny, W. (2007). Variational free energy and the Laplace approximation. Neuroimage, 34(1), 220-234.
2. Friston, K., & Penny, W. (2011). Post hoc Bayesian model selection. Neuroimage, 56(4), 2089-2099.
3. Friston, K., Parr, T., & Zeidman, P. (2018). Bayesian model reduction. arXiv preprint arXiv:1805.07092.
4. Penny, W., & Ridgway, G. (2013). Efficient posterior probability mapping using Savage-Dickey ratios. PloS one, 8(3), e59655.
5. Zeidman, P., Jafarian, A., Seghier, M. L., Litvak, V., Cagnan, H., Price, C. J., & Friston, K. J. (2019). A guide to group effective connectivity analysis, part 2: Second level analysis with PEB. Neuroimage, 200, 12-25.
6. Marković, D., Friston, K. and Kiebel, S.  (2024). "Bayesian sparsification for deep neural networks with Bayesian model reduction." IEEE Access, 12, 88231 - 88242.
