==============================
Bayesian Factor Analysis
==============================

Introduction
============

Factor Analysis (FA) is a statistical method used to describe variability among observed
variables in terms of a potentially lower number of latent variables called factors.
In the Bayesian approach to factor analysis, we place prior distributions both on the parameters
of the model, and factors and use Bayesian inference to estimate the posterior distributions.

This document provides the mathematical foundations for the Bayesian Factor Analysis model implemented in ``sppcax``.

Model Definition
================

The Bayesian Factor Analysis model assumes that observed data :math:`\mathbf{X} \in \mathbb{R}^{N \times D}` is generated from a lower-dimensional latent space :math:`\mathbf{Z} \in \mathbb{R}^{N \times K}` where :math:`K < D`:

.. math::

   \mathbf{x}_n = \mathbf{W}\mathbf{z}_n + \pmb{\mu} + \boldsymbol{\epsilon}_n

where:
 - :math:`\mathbf{x}_n \in \mathbb{R}^D` is the :math:`n`-th observation
 - :math:`\mathbf{W} \in \mathbb{R}^{D \times K}` is the loading matrix
 - :math:`\mathbf{z}_n \in \mathbb{R}^K` is the latent variable for observation :math:`n`
 - :math:`\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}^{-1})` is the noise term with :math:`\boldsymbol{\Psi}` being a diagonal precision matrix

In probabilistic notation, the model can be written as:

.. math::

   p(\mathbf{x}_n | \mathbf{z}_n, \mathbf{W}, \boldsymbol{\Psi}) = \mathcal{N}(\mathbf{x}_n | \mathbf{W}\pmb{z}_n + \pmb{\mu}, \boldsymbol{\Psi}^{-1})


Noise Precision
---------------

For the noise precision, we have two variants:

1. **Isotropic Noise** (Probabilistic PCA (PPCA) variant):

   .. math::

      p(\psi) = \text{Gamma}(\psi | \alpha^\psi_0, \beta^\psi_0)

   where :math:`\psi` is a scalar precision applied to all dimensions, and for this case :math:`\Psi = \psi I`.

2. **Diagonal Noise** (Probabilistic FA (PFA) variant):

   .. math::

      p(\psi_d) = \text{Gamma}(\psi_d | \alpha^\psi_0, \beta^\psi_0)

   where :math:`\psi_d` is the precision for dimension :math:`d`, and for this case :math:`\Psi=\text{diag}(\pmb{\psi})`.


.. _sec-prior-dist:

Prior Distributions
===================

In our Bayesian approach, we place the following prior distributions on the model parameters:

Latent Variables
----------------

.. math::

   p(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n | \mathbf{0}, \mathbf{I})

Loading Matrix
--------------

We use a hierarchical prior on the loading matrix :math:`\mathbf{W}`. For each row :math:`\pmb{w}_{d}` of the loading matrix:

.. math::

   p(\pmb{W}| \pmb{\tau}, \pmb{\psi}) &=  \prod_{k=1}^K \left(\frac{\tau_k}{2 \pi} \right)^{\frac{D - k + 1}{2}} \left[\prod_{j=k}^D \psi_j^{1/2} \right]\exp\left(-\frac{\tau_k}{2} \pmb{\bar{w}}_k^T \pmb{\Psi}_k\pmb{\bar{w}}_k\right)\\
   p(\tau_{k}) &= \text{Gamma}(\tau_{k} | \alpha^\tau_0, \beta^\tau_0) \\
   p(\psi_d) &= \text{Gamma}(\psi_d|\alpha^{\psi}_0, \beta^{\psi}_0)

where :math:`\pmb{\bar{w}}_k = (w_{kk}, \ldots, w_{dk} )`, and :math:`\pmb{\Psi}_k=\text{diag}(\psi_k, \ldots, \psi_d)`.
This hierarchical prior is encourages sparsity in the loading matrix via automatic relevance determination, and leads
to a tighter ELBO bound as requires less factorization in the approximate posterior.

Note that we use a specifically constrained parameterisation of the loading matrix as follows

.. math::

   W = \pmatrix{w_{11} & 0 & 0 & \ldots & 0 \\
    w_{21} & w_{22} & 0 & \ldots & 0 \\
    \vdots & \vdots \\
    w_{K-1,1} & w_{K-1,2} & w_{K-1,3} & \ldots & 0 \\
    w_{K,1} & w_{K,2} & w_{K,3} & \ldots & w_{K,K} \\
    \vdots & \vdots \\
    w_{D,1} & w_{D,2} & w_{D,3} & \ldots & w_{D,D}}

which helps with identifiability. In particular, in the case of Bayesian Factor Analysis the
following relation has to be satisfied :math:`(D - K)^2 \leq D + K` to ensure identifiability. In contrast, in Bayesian PCA,
the number of latent variables can be set as large as :math:`D - 1`.

.. _sec-post-dist:

Variational Inference
=====================

We use variational inference to approximate the posterior distributions of the latent variables and model parameters. The variational approximation assumes the following factorization:

.. math::

  p(\mathbf{Z}, \mathbf{W}, \pmb{\mu}, \boldsymbol{\tau}, \boldsymbol{\psi}|\pmb{X}) \approx q(\mathbf{Z})q(\pmb{\mu})q(\mathbf{W}, \boldsymbol{\psi})q(\boldsymbol{\tau}) q(\pmb{\mu})

where:
  - :math:`q(\mathbf{Z}) = \prod_{n=1}^N q(\mathbf{z}_n)` with :math:`q(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \pmb{\Sigma}_n)`
  - :math:`q(\mathbf{W}, \boldsymbol{\psi})` is a Multivariate Normal-Gamma distribution with

   .. math::
      q(\mathbf{W}, \boldsymbol{\psi}) = \prod_d \mathcal{N}(\pmb{\tilde{w}}_d|\pmb{\tilde{\mu}}_d, \psi_d^{-1}\pmb{\tilde{\Sigma}}_d)
      \text{Gamma}(\psi_d|\alpha^\psi_d, \beta^\psi_d)

  - :math:`q(\pmb{\tau})` is a product of Gamma distributions, hence :math:`q(\pmb{\tau}) = \prod_k \text{Gamma}(\tau_k|\alpha_d, \beta_d)`
  - :math:`q(\pmb{\mu})=\mathcal{N}(\pmb{\mu}|\pmb{m}, \pmb{\Sigma})` is a multivariate normal distribution.

Note that the dimensionality of :math:`\pmb{\tilde{w}}_d` is :math:`min(d, K)` as we are working with
a lower triangular matrix for :math:`\pmb{W}`.

Evidence Lower Bound (ELBO)
===========================

The variational inference optimizes the Evidence Lower Bound (ELBO), which is defined as:

.. math::

   \mathcal{L} = \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z}, \mathbf{W}, \pmb{\mu}, \boldsymbol{\tau}, \boldsymbol{\Psi})] - \mathbb{E}_{q}[\log q(\mathbf{Z}, \mathbf{W}, \pmb{\mu}, \boldsymbol{\tau}, \boldsymbol{\Psi})]

This can be expanded as:

.. math::

   \mathcal{L} &= \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \mathbf{W}, \boldsymbol{\Psi})] \\
   &+ \mathbb{E}_{q}[\log p(\mathbf{Z})] - \mathbb{E}_{q}[\log q(\mathbf{Z})] \\
   &+ \mathbb{E}_{q}[\log p(\mathbf{W} | \boldsymbol{\psi}, \pmb{\tau})] + \mathbb{E}_{q}[\log p(\boldsymbol{\tau})] + \mathbb{E}_{q}[\log p(\boldsymbol{\psi})] \\
   &- \mathbb{E}_{q}[\log q(\mathbf{W}, \boldsymbol{\psi})] - \mathbb{E}_{q}[\log q(\boldsymbol{\tau})] \\
   &+  \mathbb{E}_{q}[\log p(\boldsymbol{\mu})] - \mathbb{E}_{q}[\log q(\boldsymbol{\mu})]

The first term is the expected log-likelihood, and the remaining terms are the negative KL divergences between the approximate posteriors and the corresponding priors.

Update Equations
================

The variational inference procedure alternates between two steps:

.. _e-step:

VBE-step:
---------

Here we update the posterior over latent variables, :math:`q(\mathbf{Z})`. For each observation :math:`n`, the posterior distribution over the latent variable :math:`\mathbf{z}_n` is:

.. math::

   q(\mathbf{z}_n) &= \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n) \\
   \boldsymbol{\Sigma}_n &= (\mathbf{I} + \mathbb{E}_q[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}])^{-1} \\
   \boldsymbol{\mu}_n &= \boldsymbol{\Sigma}_n \mathbb{E}_q\left[\mathbf{W}^T \boldsymbol{\Psi} (\mathbf{x}_n - \boldsymbol{\mu}) \right]

where:

- :math:`\mathbb{E}_q[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}] = \pmb{M}^T \bar{\pmb{\Psi}}\pmb{M} + \sum_d \pmb{\Sigma}_d` is the expected precision of the latent space
- :math:`\mathbb{E}_q\left[\mathbf{W}^T \boldsymbol{\Psi} (\mathbf{x}_n - \boldsymbol{\mu}) \right]=\pmb{M}^T \pmb{\bar{\Psi}} (\mathbf{x}_n - \pmb{m})` is the precision weighted expected error

VBM-step:
---------

The updates for the loading matrix and noise precision involve computing the natural gradient of the ELBO with respect to the natural parameters of the distributions.

For the loading matrix, the update involves:

.. math::

   \mathbb{E}[\mathbf{W}] = \mathbb{E}[\mathbf{X} \mathbf{Z}^T] (\mathbb{E}[\mathbf{Z} \mathbf{Z}^T])^{-1}

Where expectations are taken with respect to the current variational distributions.

The update equations for :math:`q(\pmb{\psi})` are depend on the model variant.
For the noise precision, in the PPCA variant (isotropic noise):

.. math::

   \alpha^\psi &= \alpha^\psi_0 + \frac{n D + \sum_d min(d, K)}{2} \\
   \beta^\psi & = \beta^\psi_0 + \frac{\sum_n \pmb{x}_n^2 - \pmb{m})}{2}
   \mathbb{E}[\psi] = \frac{\alpha_0 + ND/2}{\beta_0 + \frac{1}{2}\sum_{n=1}^N \mathbb{E}[||\mathbf{x}_n - \mathbf{W}\mathbf{z}_n||^2]}

For the PFA variant (diagonal noise):

.. math::

   \alpha^\psi_d &= \alpha^\psi_0 + \frac{N + min(d, K)}{2} \\
   \beta^\psi_d &= \beta^\psi_0 + \frac{1}{2}\sum_n \left[(x_{n,d}^c - \pmb{\mu}_d^T \pmb{\mu}_n)^2
   + \pmb{\mu}_d^T \pmb{\Sigma}_n \pmb{\mu}_d\right] + \frac{N}{2}[\sigma^2_m]_{d} + \frac{1}{2}\sum_{k=1}^d \bar{\tau}_k (\sigma_{dk}^2 + \mu_{dk}^2)

where :math:`\mathbf{w}_d` is the :math:`d`-th row of the loading matrix.

Handling Missing Data
=====================

The implementation allows for missing data in the observations. This is handled by using a mask matrix :math:`\mathbf{M} \in \{0, 1\}^{N \times D}` where :math:`m_{nd} = 1` if the element :math:`x_{nd}` is observed, and :math:`m_{nd} = 0` if it is missing.

The expected log-likelihood term in the ELBO is then modified to only include observed elements:

.. math::

   \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \mathbf{W}, \boldsymbol{\Psi})] = \sum_{n=1}^N \sum_{d=1}^D m_{nd} \mathbb{E}_{q}[\log p(x_{nd} | \mathbf{z}_n, \mathbf{w}_d, \psi_d)]

The E-step update equations are also modified to account for the mask, ensuring that missing values do not influence the posterior distributions.

Probabilistic PCA vs. Factor Analysis
=====================================

The implementation provides two variants of the model:

1. **Probabilistic Principal Component Analysis (PPCA)**:
   - Uses isotropic noise (same precision for all dimensions)
   - Equivalent to PCA in the limit of infinite precision

2. **Factor Analysis (FA)**:
   - Uses diagonal noise (different precision for each dimension)
   - More flexible in modeling different noise levels across dimensions

References
==========

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
2. Attias, H. (1999). Inferring parameters and structure of latent variable models by variational Bayes. In Proceedings of the Fifteenth conference on Uncertainty in artificial intelligence.
3. Zhao, J. H., and Philip, L. H. (2009). A note on variational Bayesian factor analysis. Neural Networks, 22(7), 988-997.
