==============================
Bayesian Factor Analysis
==============================

Introduction
============

Factor Analysis (FA) is a statistical method used to describe variability among observed variables in terms of a potentially lower number of unobserved variables called factors. In the Bayesian approach to factor analysis, we place prior distributions on the parameters of the model and use Bayesian inference to estimate the posterior distributions.

This document provides the mathematical foundations for the Bayesian Factor Analysis model implemented in ``sppcax``.

Model Definition
================

The Bayesian Factor Analysis model assumes that observed data :math:`\mathbf{X} \in \mathbb{R}^{N \times D}` is generated from a lower-dimensional latent space :math:`\mathbf{Z} \in \mathbb{R}^{N \times K}` where :math:`K < D`:

.. math::

   \mathbf{x}_n = \mathbf{W}\mathbf{z}_n + \boldsymbol{\epsilon}_n

where:

- :math:`\mathbf{x}_n \in \mathbb{R}^D` is the :math:`n`-th observation
- :math:`\mathbf{W} \in \mathbb{R}^{D \times K}` is the loading matrix
- :math:`\mathbf{z}_n \in \mathbb{R}^K` is the latent variable for observation :math:`n`
- :math:`\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}^{-1})` is the noise term with :math:`\boldsymbol{\Psi}` being a diagonal precision matrix

In probabilistic notation, the model can be written as:

.. math::

   p(\mathbf{x}_n | \mathbf{z}_n, \mathbf{W}, \boldsymbol{\Psi}) = \mathcal{N}(\mathbf{x}_n | \mathbf{W}\mathbf{z}_n, \boldsymbol{\Psi}^{-1})


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

We use a hierarchical prior on the loading matrix :math:`\mathbf{W}`. For each element :math:`w_{dk}` of the loading matrix:

.. math::

   p(w_{dk} | \tau_{dk}) &= \mathcal{N}(w_{dk} | 0, \tau_{dk}^{-1}) \\
   p(\tau_{dk}) &= \text{Gamma}(\tau_{dk} | \alpha_0, \beta_0)

where :math:`\tau_{dk}` is the precision of the loading matrix element. This hierarchical prior is known as the Normal-Gamma prior and encourages sparsity in the loading matrix, particularly when using Bayesian Model Reduction.

Noise Precision
---------------

For the noise precision, we have two variants:

1. **Isotropic Noise** (PPCA variant):

   .. math::

      p(\psi) = \text{Gamma}(\psi | \alpha_0, \beta_0)

   where :math:`\psi` is a scalar precision applied to all dimensions.

2. **Diagonal Noise** (FA variant):

   .. math::

      p(\psi_d) = \text{Gamma}(\psi_d | \alpha_0, \beta_0)

   where :math:`\psi_d` is the precision for dimension :math:`d`.

.. _sec-post-dist:
Appoximate Posterior Distributions
===================

Variational Inference
=====================

We use variational inference to approximate the posterior distributions of the latent variables and model parameters. The variational approximation assumes the following factorization:

.. math::

   q(\mathbf{Z}, \mathbf{W}, \boldsymbol{\tau}, \boldsymbol{\Psi}) = q(\mathbf{Z})q(\mathbf{W}, \boldsymbol{\tau})q(\boldsymbol{\Psi})

where:

- :math:`q(\mathbf{Z}) = \prod_{n=1}^N q(\mathbf{z}_n)` with :math:`q(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n)`
- :math:`q(\mathbf{W}, \boldsymbol{\tau})` is a Multivariate Normal-Gamma distribution
- :math:`q(\boldsymbol{\Psi})` is a Gamma distribution

Evidence Lower Bound (ELBO)
===========================

The variational inference optimizes the Evidence Lower Bound (ELBO), which is defined as:

.. math::

   \mathcal{L} = \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z}, \mathbf{W}, \boldsymbol{\tau}, \boldsymbol{\Psi})] - \mathbb{E}_{q}[\log q(\mathbf{Z}, \mathbf{W}, \boldsymbol{\tau}, \boldsymbol{\Psi})]

This can be expanded as:

.. math::

   \mathcal{L} &= \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \mathbf{W}, \boldsymbol{\Psi})] \\
   &+ \mathbb{E}_{q}[\log p(\mathbf{Z})] - \mathbb{E}_{q}[\log q(\mathbf{Z})] \\
   &+ \mathbb{E}_{q}[\log p(\mathbf{W} | \boldsymbol{\tau})] + \mathbb{E}_{q}[\log p(\boldsymbol{\tau})] - \mathbb{E}_{q}[\log q(\mathbf{W}, \boldsymbol{\tau})] \\
   &+ \mathbb{E}_{q}[\log p(\boldsymbol{\Psi})] - \mathbb{E}_{q}[\log q(\boldsymbol{\Psi})]

The first term is the expected log-likelihood, and the remaining terms are the negative KL divergences between the approximate posteriors and the corresponding priors.

Update Equations
================

The variational inference procedure alternates between two steps:

.. _e-step:
E-step:
--------

Here we update the posterior over latent variables, :math:`q(\mathbf{Z})`. For each observation :math:`n`, the posterior distribution over the latent variable :math:`\mathbf{z}_n` is:

.. math::

   q(\mathbf{z}_n) &= \mathcal{N}(\mathbf{z}_n | \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n) \\
   \boldsymbol{\Sigma}_n &= (\mathbf{I} + \mathbb{E}[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}])^{-1} \\
   \boldsymbol{\mu}_n &= \boldsymbol{\Sigma}_n \mathbb{E}[\mathbf{W}^T \boldsymbol{\Psi}] (\mathbf{x}_n - \boldsymbol{\mu})

where:
- :math:`\mathbb{E}[\mathbf{W}^T \boldsymbol{\Psi} \mathbf{W}]` is the expected precision of the latent space
- :math:`\boldsymbol{\mu}` is the data mean

M-step: Update :math:`q(\mathbf{W}, \boldsymbol{\tau})` and :math:`q(\boldsymbol{\Psi})`
----------------------------------------------------------------------------------------

The updates for the loading matrix and noise precision involve computing the natural gradient of the ELBO with respect to the natural parameters of the distributions.

For the loading matrix, the update involves:

.. math::

   \mathbb{E}[\mathbf{W}] = \mathbb{E}[\mathbf{X} \mathbf{Z}^T] (\mathbb{E}[\mathbf{Z} \mathbf{Z}^T])^{-1}

Where expectations are taken with respect to the current variational distributions.

For the noise precision, in the PPCA variant (isotropic noise):

.. math::

   \mathbb{E}[\psi] = \frac{\alpha_0 + ND/2}{\beta_0 + \frac{1}{2}\sum_{n=1}^N \mathbb{E}[||\mathbf{x}_n - \mathbf{W}\mathbf{z}_n||^2]}

For the FA variant (diagonal noise):

.. math::

   \mathbb{E}[\psi_d] = \frac{\alpha_0 + N/2}{\beta_0 + \frac{1}{2}\sum_{n=1}^N \mathbb{E}[(x_{nd} - \mathbf{w}_d^T\mathbf{z}_n)^2]}

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
3. Paisley, J., Blei, D. M., & Jordan, M. I. (2012). Variational Bayesian inference with stochastic search. In International Conference on Machine Learning.
4. Ilin, A., & Raiko, T. (2010). Practical approaches to principal component analysis in the presence of missing values. Journal of Machine Learning Research, 11, 1957-2000.
5. Zhao, J. H., Yu, P. L., & Jiang, Q. (2008). ML estimation for factor analysis: EM or non-EM? Statistics and Computing, 18(2), 109-123.
