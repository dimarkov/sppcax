==============================
Factor Analysis and PCA
==============================

Introduction
============

Factor Analysis (FA) and Probabilistic PCA (PPCA) are special cases of the
Dynamic Factor Analysis model (see :doc:`dynamic_factor_analysis`). They are obtained
when there are no temporal dynamics, i.e., :math:`\mathbf{F} = \mathbf{0}`,
:math:`\mathbf{Q} = \mathbf{I}`. Each observation :math:`\mathbf{x}_n` is treated as
an independent single-timestep sequence: data is reshaped from :math:`(N, D)` to
:math:`(N, 1, D)` so that the Kalman smoother reduces to a single filter step per
observation.

This document provides the detailed VB-EM update equations for the static case.
The two variants differ only in the noise model:

1. **Probabilistic Principal Component Analysis (PPCA)**:
   - Uses isotropic noise (same precision for all dimensions)
   - Equivalent to PCA in the limit of infinite precision

2. **Factor Analysis (FA)**:
   - Uses diagonal noise (different precision for each dimension)
   - More flexible in modeling different noise levels across dimensions


Model Definition
================

The Bayesian Factor Analysis model assumes that observed data :math:`\mathbf{X} \in \mathbb{R}^{N \times D}` is generated from a lower-dimensional latent space :math:`\mathbf{Z} \in \mathbb{R}^{N \times K}` where :math:`K < D`:

.. math::

   \mathbf{x}_n = \tilde{\mathbf{W}}\tilde{\mathbf{z}}_n + \boldsymbol{\epsilon}_n

where :math:`\tilde{\mathbf{z}}_n = [\mathbf{z}_n^\top, 1]^\top` is the latent variable
augmented with a constant 1, and :math:`\tilde{\mathbf{W}} = [\mathbf{W}, \boldsymbol{\mu}] \in \mathbb{R}^{D \times (K+1)}`
is the augmented loading matrix whose last column is the bias vector :math:`\boldsymbol{\mu}`.
Equivalently, the model can be written as:

.. math::

   \mathbf{x}_n = \mathbf{W}\mathbf{z}_n + \boldsymbol{\mu} + \boldsymbol{\epsilon}_n

where:
 - :math:`\mathbf{x}_n \in \mathbb{R}^D` is the :math:`n`-th observation
 - :math:`\mathbf{W} \in \mathbb{R}^{D \times K}` is the loading matrix
 - :math:`\mathbf{z}_n \in \mathbb{R}^K` is the latent variable for observation :math:`n`
 - :math:`\boldsymbol{\mu} \in \mathbb{R}^D` is the bias (last column of :math:`\tilde{\mathbf{W}}`)
 - :math:`\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}^{-1})` is the noise term with :math:`\boldsymbol{\Psi}` being a diagonal precision matrix

.. note::

   The bias :math:`\boldsymbol{\mu}` is **not** a separate parameter — it is absorbed into the
   augmented loading matrix :math:`\tilde{\mathbf{W}}` and learned jointly under a single MVNIG
   prior. In the code, a constant input of 1 is appended to the input vector so that the bias
   emerges as the last column of the weight matrix.

In probabilistic notation:

.. math::

   p(\mathbf{x}_n | \mathbf{z}_n, \tilde{\mathbf{W}}, \boldsymbol{\Psi}) = \mathcal{N}(\mathbf{x}_n | \tilde{\mathbf{W}}\tilde{\mathbf{z}}_n, \boldsymbol{\Psi}^{-1})


.. _sec-prior-dist:

Prior Distributions
===================

In our Bayesian approach, we place the following prior distributions on the model parameters.
For the full DFA prior specification including ARD priors, see :doc:`dynamic_factor_analysis`.

Latent Variables
----------------

.. math::

   p(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n | \mathbf{0}, \mathbf{I})


.. _sec-load-mat:

Loading Matrix and Noise Precision
------------------------------------

The loading matrix and noise precision have a joint Multivariate Normal-Inverse Gamma (MVNIG)
prior. For each row :math:`d` of the augmented loading matrix
:math:`\tilde{\mathbf{w}}_d = [\mathbf{w}_d, \mu_d]` (loadings and bias):

.. math::

   p(\tilde{\mathbf{w}}_d, \psi_d) = \mathcal{N}(\tilde{\mathbf{w}}_d \mid \mathbf{0},
   \psi_d^{-1} \tilde{\boldsymbol{\Sigma}}_d^0) \,
   \text{Gamma}(\psi_d \mid \alpha_0^\psi, \beta_0^\psi)

Column-wise ARD priors :math:`\tau_k \sim \text{Gamma}(\alpha_0^\tau, \beta_0^\tau)` are placed
on the loading matrix columns, with :math:`\mathbb{E}[\tau_k]` incorporated into the prior
precision :math:`\tilde{\boldsymbol{\Sigma}}_d^{0,-1}` (see :ref:`sec-ard-priors`).

For the noise precision, we have two variants:

1. **Isotropic Noise** (PPCA variant):
   :math:`\boldsymbol{\Psi} = \psi \mathbf{I}` with a shared :math:`\psi \sim \text{Gamma}(\alpha_0^\psi, \beta_0^\psi)`

2. **Diagonal Noise** (FA variant):
   :math:`\boldsymbol{\Psi} = \text{diag}(\psi_1, \ldots, \psi_D)` with independent
   :math:`\psi_d \sim \text{Gamma}(\alpha_0^\psi, \beta_0^\psi)`

.. note::

   In ``sppcax``, the initial distribution :math:`\mathbf{z}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)`
   is **not updated by default** for FA and PCA. The prior is set so that
   :math:`\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` independently for each observation.
   This can be changed by providing a custom ``initial_prior`` and setting the initial distribution
   to be trainable.

.. _sec-post-dist:

Variational Inference
=====================

We use variational inference to approximate the posterior distributions of the latent variables
and model parameters. Since the bias :math:`\boldsymbol{\mu}` is part of the augmented loading
matrix :math:`\tilde{\mathbf{W}}`, it does not require a separate variational factor. The
variational approximation assumes the following factorization:

.. math::

  p(\mathbf{Z}, \tilde{\mathbf{W}}, \boldsymbol{\tau}, \boldsymbol{\psi}|\mathbf{X}) \approx q(\mathbf{Z})\,q(\tilde{\mathbf{W}}, \boldsymbol{\psi})\,q(\boldsymbol{\tau})

where:
  - :math:`q(\mathbf{Z}) = \prod_{n=1}^N q(\mathbf{z}_n)` with :math:`q(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n)`
  - :math:`q(\tilde{\mathbf{W}}, \boldsymbol{\psi})` is a Multivariate Normal-Inverse Gamma distribution with

   .. math::
      q(\tilde{\mathbf{W}}, \boldsymbol{\psi}) = \prod_d \mathcal{N}(\tilde{\mathbf{w}}_d|\tilde{\boldsymbol{\mu}}_d, \psi_d^{-1}\tilde{\boldsymbol{\Sigma}}_d)
      \text{Gamma}(\psi_d|\alpha^\psi_d, \beta^\psi_d)

  - :math:`q(\boldsymbol{\tau})` is a product of Gamma distributions, hence :math:`q(\boldsymbol{\tau}) = \prod_k \text{Gamma}(\tau_k|\alpha_k, \beta_k)`

Evidence Lower Bound (ELBO)
===========================

The variational inference optimizes the Evidence Lower Bound (ELBO), which is defined as:

.. math::

   \mathcal{L} = \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z}, \tilde{\mathbf{W}}, \boldsymbol{\tau}, \boldsymbol{\Psi})] - \mathbb{E}_{q}[\log q(\mathbf{Z}, \tilde{\mathbf{W}}, \boldsymbol{\tau}, \boldsymbol{\Psi})]

This can be expanded as:

.. math::

   \mathcal{L} &= \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \tilde{\mathbf{W}}, \boldsymbol{\Psi})] \\
   &+ \mathbb{E}_{q}[\log p(\mathbf{Z})] - \mathbb{E}_{q}[\log q(\mathbf{Z})] \\
   &+ \mathbb{E}_{q}[\log p(\tilde{\mathbf{W}} | \boldsymbol{\psi}, \boldsymbol{\tau})] + \mathbb{E}_{q}[\log p(\boldsymbol{\tau})] + \mathbb{E}_{q}[\log p(\boldsymbol{\psi})] \\
   &- \mathbb{E}_{q}[\log q(\tilde{\mathbf{W}}, \boldsymbol{\psi})] - \mathbb{E}_{q}[\log q(\boldsymbol{\tau})]

The first term is the expected log-likelihood, and the remaining terms are the negative KL divergences between the approximate posteriors and the corresponding priors.

Update Equations
================

The variational inference procedure alternates between two steps:

.. _e-step:

VBE-step:
---------

Here we update the posterior over latent variables, :math:`q(\mathbf{Z})`. For each
observation :math:`n`, define the augmented latent vector
:math:`\tilde{\mathbf{z}}_n = [\mathbf{z}_n^\top, 1]^\top`. The posterior distribution
over :math:`\mathbf{z}_n` is:

.. math::

   q(\mathbf{z}_n) &= \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n) \\
   \boldsymbol{\Sigma}_n &= (\mathbf{I} + \mathbb{E}_q[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}])^{-1} \\
   \boldsymbol{\mu}_n &= \boldsymbol{\Sigma}_n \, \mathbb{E}_q\left[\mathbf{W}^T \boldsymbol{\Psi}\right] (\mathbf{x}_n - \bar{\boldsymbol{\mu}})

where :math:`\bar{\boldsymbol{\mu}}` is the expected bias (last column of
:math:`\tilde{\boldsymbol{\mu}}_d`) and:

  - :math:`\mathbb{E}_q[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}] = \mathbf{M}^T \bar{\boldsymbol{\Psi}}\mathbf{M} + \sum_d \bar{\psi}_d \boldsymbol{\Sigma}_d^{[:K,:K]}` is the expected precision of the latent space, where :math:`\mathbf{M}` denotes the first :math:`K` columns of the posterior mean :math:`\tilde{\boldsymbol{\mu}}_d`
  - :math:`\mathbb{E}_q[\mathbf{W}^T \boldsymbol{\Psi}] (\mathbf{x}_n - \bar{\boldsymbol{\mu}}) = \mathbf{M}^T \bar{\boldsymbol{\Psi}} (\mathbf{x}_n - \bar{\boldsymbol{\mu}})` is the precision-weighted expected residual

VBM-step:
---------

We update the parameters of the joint posterior :math:`q(\tilde{\mathbf{W}}, \boldsymbol{\psi})`
of the augmented loading matrix and noise precision. The augmented sufficient statistics use
:math:`\tilde{\mathbf{z}}_n = [\mathbf{z}_n^\top, 1]^\top`:

.. math::
   \tilde{\mathbf{P}}_d &= \text{diag}(\boldsymbol{\tau}, 0) + \sum_n \left\langle \tilde{\mathbf{z}}_n \tilde{\mathbf{z}}_n^T \right\rangle \\
   \tilde{\mathbf{P}}_d \tilde{\boldsymbol{\mu}}_d &= \sum_n \left\langle \tilde{\mathbf{z}}_n \right\rangle x_{n,d}

where :math:`\tilde{\mathbf{P}}_d = \tilde{\boldsymbol{\Sigma}}_d^{-1}` is the
:math:`(K+1) \times (K+1)` precision matrix of the augmented posterior.

The update equations for :math:`q(\boldsymbol{\psi})` depend on the model variant.
For the FA variant (diagonal noise):

.. math::

   \alpha^\psi_d &= \alpha^\psi_0 + \delta \alpha_d^\psi \\
   \delta \alpha_d^\psi &= \frac{N + K + 1}{2} \\
   \beta^\psi_d &= \beta^\psi_0 + \delta \beta^\psi_d \\
   \delta \beta^\psi_d &= \frac{1}{2}\sum_n \left[(x_{n,d} - \tilde{\boldsymbol{\mu}}_d^T \langle\tilde{\mathbf{z}}_n\rangle)^2 + \tilde{\boldsymbol{\mu}}_d^T \tilde{\boldsymbol{\Sigma}}_n \tilde{\boldsymbol{\mu}}_d\right] \\
   &+ \frac{1}{2}\sum_{k=1}^{K} \bar{\tau}_k (\tilde{\sigma}_{dk}^2 + \tilde{\mu}_{dk}^2)

where :math:`\tilde{\boldsymbol{\Sigma}}_n` is the :math:`(K+1) \times (K+1)` augmented
second-moment correction (with the last row/column accounting for the constant 1).

The PPCA variant is then obtained as :math:`\alpha^{\psi} = \alpha_0^{\psi} + \sum_d \delta \alpha_d^\psi`, and
:math:`\beta^\psi=\beta_0^\psi + \sum_d \delta \beta_d^{\psi}`.

In the final step of the VBM, we update :math:`q(\boldsymbol{\tau})` as

.. math::

   \alpha^{\tau}_k &= \alpha_0^{\tau} + \frac{D}{2}\\
   \beta^{\tau}_k &= \beta_0^{\tau} + \frac{1}{2} \sum_{d=1}^D \bar{\psi}_d \left[\tilde{\sigma}^2_{d,k} + \tilde{\mu}^2_{d,k} \right]

Handling Missing Data
=====================

The implementation allows for missing data in the observations. This is handled by using a mask matrix :math:`\mathbf{M} \in \{0, 1\}^{N \times D}` where :math:`m_{nd} = 1` if the element :math:`x_{nd}` is observed, and :math:`m_{nd} = 0` if it is missing.

The expected log-likelihood term in the ELBO is then modified to only include observed elements:

.. math::

   \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \tilde{\mathbf{W}}, \boldsymbol{\Psi})] = \sum_{n=1}^N \sum_{d=1}^D m_{nd} \mathbb{E}_{q}[\log p(x_{nd} | \mathbf{z}_n, \tilde{\mathbf{w}}_d, \psi_d)]


References
==========

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
2. Attias, H. (1999). Inferring parameters and structure of latent variable models by variational Bayes. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence.
3. Zhao, J. H., and Philip, L. H. (2009). A note on variational Bayesian factor analysis. Neural Networks, 22(7), 988-997.
