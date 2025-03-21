============================
Bayesian Model Reduction
============================

Introduction
============

Bayesian Model Reduction (BMR) is a generalisation of the Savage-Dickey ratio allowing an efficient
comparison of a large number of generative models, all differing only in their specification of the
prior distribution. In the context of our Bayesian Factor Analysis implementation, we use BMR to
post-hoc prune parameters of the loading matrix, and removing redundant latent dimensions, providing
an efficient approach to learning sparse representations.

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
we assume the same functional form for the posterior, a Multivariate Normal - Gamma distribution, which we use for the prior, (see :ref:`sec-prior-dist` for details).
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

Bayesian Model Reduction for Factor Analysis
============================================

In the context of Bayesian Factor Analysis, we use BMR to prune unnecessary elements of the loading matrix :math:`\mathbf{W}`. This is achieved by setting the prior precision of selected elements to infinity, effectively forcing those elements to zero.

The reduced model differs from the full model only in the priors for specific elements of the loading matrix. The loading matrix elements are governed by a Normal-Gamma prior, as
specified in :ref:`sec-load-mat`. We will express the reduced prior of model :math:`m_i` (a prior in which additional elements of the loading
matrix are fixed to zero) as:

.. math::

   p_i(\pmb{W}| \pmb{\tau}, \pmb{\psi}) &=  \prod_{k=1}^K \delta(\pmb{\tilde{w}}_k^i) \left(\frac{\tau_k}{2 \pi} \right)^{\frac{D - k - r_k^i + 1}{2}}
   \sqrt{|\pmb{\Psi}_k^i|} \exp\left(-\frac{\tau_k}{2} \pmb{\bar{w}}_{k, i}^T \pmb{\Psi}_k^i \pmb{\bar{w}}_{k, i}\right)\\

where :math:`\pmb{\bar{w}}_k^i` denotes a vector of the remaining (non-zero) elements of the kth column, :math:`\pmb{\tilde{w}}_k^i` denotes a vector
pruned elements, :math:`r_k^i` is the number of pruned elements in kth column, and :math:`\pmb{\Psi}_k^i` is a diagonal matrix containing the remaining
elements of the noise precision matrix.

Given that the approximate posterior :math:`q_0(\pmb{\theta})` is specified as (see :ref:`sec-post-dist`)

.. math::
   q_0(\pmb{\theta}) = \prod_k \text{Gamma}(\tau_k|\alpha^\tau_k, \beta^\tau_k)\prod_d \text{Gamma}(\psi_d|\alpha^\psi_d, \beta^\psi_d) \mathcal{N}(\pmb{w}_d; \pmb{\mu}_d, \psi_d^{-1} \pmb{\Sigma}_d)

(where we ignore :math:`q(\pmb{\mu})` as it plays no role in pruning) we can express the change in expected free energy :math:`\Delta F` as

.. math::
   \Delta F_i &= \ln \int d \pmb{\tau}~q_0(\pmb{\tau}) \int d \pmb{\psi} ~q_0(\pmb{\psi})
   \int d \pmb{\pmb{W}}~q_0 (\pmb{W}| \pmb{\psi}) \prod_{k=1}^K \delta(\pmb{\tilde{w}}_{k}^i)
   \left( \frac{\tau_k}{2 \pi}\right)^{-r_k^i/2} |\pmb{\tilde{\Psi}}_k^i|^{-1/2} \\
   & = \ln \int d \pmb{\tau}~q_0(\pmb{\tau}) \int d \pmb{\psi} ~q_0(\pmb{\psi})
   \prod_{d=1}^D |\pmb{\tilde{\Sigma}}_d^i|^{-1/2}
   \exp \left( - \frac{1}{2} [\pmb{\tilde{\mu}}_d^i]^T \pmb{\tilde{P}}_d^i \pmb{\tilde{\mu}}_d^i  \right)
   \prod_{k=1}^K \tau_k^{- r_k^i / 2}

where we use tilde sign to denote a vector subset corresponding to pruned elements (of the d-th component) in the reduced model :math:`m_i` relative to the full model :math:`m_0`.

Computing :math:`\Delta F`
==========================
We will split the computation of :math:`\Delta F_i` into several components. First, we will use the follwing relation for
the expectation over the inverse square root of :math:`\tau_k`

.. math::
   c_k^i = \int d \tau_k q_0(\tau_k) \tau_k^{- r_k^i / 2} =
   [\beta_k^\tau]^{r_k^i / 2} \frac{\Gamma(\alpha_k^\tau - r_k^i/2)}{\Gamma(\alpha_k^\tau)}

We will use :math:`C_{i} = \prod_{k} c_k^i` to denote the product of corresponding factors.

The expectation over :math:`\pmb{\psi}` results in the following expression for :math:`\Delta F_i`

.. math::
   \Delta F_i &= \ln C_i + \sum_d \ln \int d \psi_d q_0(\psi_d) \frac{1}{\sqrt{|\pmb{\tilde{\Sigma}}_{d}^i|}} \exp \left\{ -\frac{\rho_d}{2} \pmb{\tilde{\mu}}_{di}^T \pmb{\tilde{\Sigma}}_{di}^{-1}\pmb{\tilde{\mu}}_{di} \right\} \\
   &= \ln C_i + \sum_d \ln \frac{1}{\sqrt{|\pmb{\tilde{\Sigma}}_{di}|}}
   \left(\frac{\beta_d^\psi}{\beta_d^\psi + \frac{1}{2} \cdot \pmb{\tilde{\mu}}_{di}^T \pmb{\tilde{\Sigma}}_{di}^{-1} \pmb{\tilde{\mu}}_{di}}\right)^{\alpha_d^\psi}

Similarly, the change in variational free energy of going from model :math:`m_{i-1}` to model :math:`m_i`, which only differ in
a single element of the loading matrix (e.g. at the position :math:`d, k^*`) is obtained as

.. math::
   \Delta F_{i:i-1} &= \Delta F_{i} - \Delta F_{i-1} \\
   &= \ln \frac{c_{k^*}^i}{c_{k^*}^{i-1}} - \frac{1}{2}\ln \sigma_{d,k^*}^2 - \alpha_d^\psi \ln \left( \frac{\beta_d^\psi + \frac{1}{2}[\pmb{\tilde{\mu}}_d^T \pmb{\tilde{\Sigma}}_d^{-1} \pmb{\tilde{\mu}}_d]_i}{\beta_d^\psi + \frac{1}{2}[\pmb{\tilde{\mu}}_d^T \pmb{\tilde{\Sigma}}_d^{-1} \pmb{\tilde{\mu}}_d]_{i-1}}\right)



Gibbs sampling
==============

To determine the final sparse structure of the loading matrix, we utilize the following Gibbs sampling based
appraoch. We assume that a prior probability of an element being pruned or not is given by :math:`\pi \sim \text{Beta}(a_0, b_0)`.
Hence, the sparse structure matrix :math:`\Lambda` is a priory sampled as

.. math::
   \lambda_{dk} \sim \mathcal{Be}(\pi)

In other words, we put a spike-and-slab prior on elements of the sparse structure matrix. If :math:`\lambda_{dk}=1` then we have
the usual normal prior on that element, and if :math:`\lambda_{dk}=0` the prior corresponds to a delta distribution, and that element
is forced to zero.

To obtain posterior samples form :math:`q(\lambda_{dk})` we first utilize the indepence of :math:`\Delta F_i` computations between
components :math:`d \in \{1, \ldots, D\}`. Thus, we can sample in parallel over :math:`\pmb{\lambda}_k` constrained on values
of all other elements, obtained in the previous iteration step :math:`\Lambda_k^{(t-1)}=(\pmb{\lambda}_{1}^{(t-1)}, \ldots, \pmb{\lambda}_{k-1}^{(t-1)}, \pmb{\lambda}_{k+1}^{(t-1)} \ldots, \pmb{\lambda}_{K}^{(t-1)})`.
Hence,

.. math::
   \pmb{\lambda}_k \sim \prod_d q(\lambda_{dk}) = \prod_d \mathcal{Be}\left(\sigma\left(-\Delta F_{d}\left[\Lambda_k^{(t-1)}\right] + \ln \frac{\pi_{t-1}}{1-\pi_{t-1}}\right)\right)

where :math:`\mathcal{Be}(\cdot)` denotes Bernoulli distribution, :math:`\sigma(\cdot)` logistic function, and
:math:`\Delta F_{d}\left[\Lambda_k^{(t-1)}\right]` corresponds to the d-th component of the change in variational free energy between
two models that differ only in element  :math:`dk` of the loading matrix. Practically, with the equation above
we are saying that the posterior probability of :math:`q(\lambda_{dk}=1)` corresponds to

.. math::
   q(\lambda_{dk}=1) &= \frac{p(\pmb{y}|m_{i-1})\pi}{p(\pmb{y}|m_{i-1}) \pi + p(\pmb{y}|m_{i}) (1 - \pi)} \\
   &\approx \frac{e^{-F(m_{i - 1}) + \ln \pi }}{e^{-F(m_{i - 1}) + \ln \pi } + e^{-F(m_{i}) + \ln (1 - \pi) }} \\
   &= \frac{1}{1 + e^{-F(m_{i}) + F(m_{i-1}) + \ln (1 - \pi) - \ln \pi }}  \\
   &= \frac{1}{1 + e^{\Delta F_{d, i:i-1} - \ln \frac{\pi}{1-\pi}}}

Simimilarly, we can generate posterior samples for :math:`q(\pi)` and infer the effective level of sparisty in
the loading matrix by sampling :math:`\pi_t \sim q_t(\pi) = \text{Beta}(a_t, b_t)` where

.. math::
   a_t &= a_0 + \sum_{d, k} \lambda_{dk}^{(t)} \\
   b_t &= b_0 + \sum_{d, k} 1 - \lambda_{dk}^{(t)}

The implementation of the BMR algorithm in ``sppcax`` follows these steps:

1. Sample :math:`\pi_0 \sim p(\pi)` and set :math:`\Lambda^{(0)}` to the basic constrain on the loading matrix as defined in :ref:`sec-prior-dist`.

2. Iterate :math:`t \in {1, \ldots, T}`:

   a. Iterate :math:`k \in \{1, \ldots, K\}`:
       * Sample :math:`\pmb{\lambda}_k^{(t)} \sim \prod_d q_{tk}(\lambda_{dk})`.
   b. Sample :math:`\pi_t \sim q_t(\pi)`.

3. Return posterior with pruned parameters :math:`q(\pmb{\theta}|\Lambda^{(T)})`.

If we assume for simplicity that the final sparse structure matrix :math:`\Lambda^{(T)}` corresponds to a reduced model :math:`m_i`,
then the updated posterior is obtained as follows:

   1. :math:`q_i(\pmb{W})`  is obtained from :math:`q_0(\pmb{W})` by forcing mean and variance of corresponding elements of the loading matrix to zero.
   2. :math:`q_i(\pmb{\psi})` is obtained using following parameter updates:
       .. math::
         \alpha_{d,i}^\psi &= \alpha_d^\psi \\
         \beta_{d, i}^\psi &= \beta_d^\psi + \frac{1}{2} \cdot \pmb{\tilde{\mu}}_{di}^T \pmb{\tilde{\Sigma}}_{di}^{-1} \pmb{\tilde{\mu}}_{di}

   3.  :math:`q_i(\pmb{\tau})` is updated assume

       .. math::
         \alpha_{k, i}^\tau &= \alpha_k^\tau - r_k^i/2 \\
         \beta_{k, i}^\tau &= \beta_k^\tau


References
==========

1. Friston, K., Mattout, J., Trujillo-Barreto, N., Ashburner, J., & Penny, W. (2007). Variational free energy and the Laplace approximation. Neuroimage, 34(1), 220-234.
2. Friston, K., & Penny, W. (2011). Post hoc Bayesian model selection. Neuroimage, 56(4), 2089-2099.
3. Friston, K., Parr, T., & Zeidman, P. (2018). Bayesian model reduction. arXiv preprint arXiv:1805.07092.
4. Penny, W., & Ridgway, G. (2013). Efficient posterior probability mapping using Savage-Dickey ratios. PloS one, 8(3), e59655.
5. Zeidman, P., Jafarian, A., Seghier, M. L., Litvak, V., Cagnan, H., Price, C. J., & Friston, K. J. (2019). A guide to group effective connectivity analysis, part 2: Second level analysis with PEB. Neuroimage, 200, 12-25.
6. Marković, D., Friston, K. and Kiebel, S.  (2024). "Bayesian sparsification for deep neural networks with Bayesian model reduction." IEEE Access, 12, 88231 - 88242.
