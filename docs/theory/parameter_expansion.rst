=====================================================
Parameter Expansion for Dynamic Factor Analysis
=====================================================

Introduction
============

A persistent difficulty with variational Bayesian inference for factor models is
slow convergence caused by strong posterior coupling between the loading matrix
and the latent variables. This coupling arises from the rotational invariance of
the factor model: for any non-singular matrix :math:`\mathbf{R}`, the
transformation :math:`\mathbf{H} \to \mathbf{H}\mathbf{R}^{-1}`,
:math:`\mathbf{z}_t \to \mathbf{R}\mathbf{z}_t` leaves the likelihood
unchanged. As a consequence, sequential VB updates zigzag slowly along
ridge-lines of equal likelihood.

Parameter expansion (PX) alleviates this problem by introducing auxiliary
parameters that decouple the variational factors, enabling faster convergence.
The **Parameter Expanded Variational Bayes (PX-VB)** framework [1]_ optimizes
auxiliary parameters within the variational framework, and the result is remapped
to the original parameterization via a reduction step.

This document derives the PX-VB approach for the Dynamic Factor Analysis (DFA)
model implemented in ``sppcax``. For the DFA model definition, prior distributions,
and ARD priors, see :doc:`dynamic_factor_analysis`. For the static-case VB-EM update
equations, see :doc:`factor_analysis`.


Parameter Expanded VBEM (PX-VBEM)
==================================

The PX-VB framework [1]_ introduces auxiliary parameters :math:`\boldsymbol{\alpha}`
that expand the original model :math:`p(\tilde{\mathbf{x}}, D)` into
:math:`p_{\boldsymbol{\alpha}}(\mathbf{x}, D)`, where the original model is
recovered at :math:`\boldsymbol{\alpha} = \boldsymbol{\alpha}_0`. Each PX-VB
iteration consists of three steps:

1. Sequential VB updates with :math:`\boldsymbol{\alpha} = \boldsymbol{\alpha}_0`
2. Minimize :math:`\text{KL}(q(\mathbf{x}) \| p_{\boldsymbol{\alpha}}(\mathbf{x}, D))`
   over :math:`\boldsymbol{\alpha}`
3. Reduce the expanded model back to the original via reparameterization


.. _sec-rotation-expansion:

Rotation-Based Parameter Expansion for LGSSM
---------------------------------------------

For state-space models, a natural parameter expansion is the rotation of the
latent subspace [3]_. For any non-singular :math:`\mathbf{R} \in \mathbb{R}^{K \times K}`:

.. math::
   :label: rotation-obs

   \mathbf{x}_t = \mathbf{H}\mathbf{z}_t = (\mathbf{H}\mathbf{R}^{-1})(\mathbf{R}\mathbf{z}_t)

The rotation must be compensated in the dynamics:

.. math::
   :label: rotation-dyn

   \mathbf{R}\mathbf{z}_t = \mathbf{R}\mathbf{F}\mathbf{R}^{-1}(\mathbf{R}\mathbf{z}_{t-1}) + \text{noise}

Thus the expanded model uses transformed variables
:math:`\tilde{\mathbf{z}}_t = \mathbf{R}\mathbf{z}_t`,
:math:`\tilde{\mathbf{H}} = \mathbf{H}\mathbf{R}^{-1}`, and
:math:`\tilde{\mathbf{F}} = \mathbf{R}\mathbf{F}\mathbf{R}^{-1}`.


Standard VB-EM Updates
-----------------------

The factorized approximate posterior is:

.. math::

   p(\mathbf{Z}, \mathbf{H}, \boldsymbol{\Psi}, \mathbf{F}, \mathbf{Q}, \boldsymbol{\tau} | \mathbf{X})
   \approx q(\mathbf{Z})\,q(\mathbf{H}, \boldsymbol{\Psi})\,q(\mathbf{F}, \mathbf{Q})\,q(\boldsymbol{\tau})

The **VBE-step** updates :math:`q(\mathbf{Z})` via Kalman smoothing using the
expected parameters from the other factors.

The **VBM-step** updates each factor in turn:

- :math:`q(\mathbf{H}, \boldsymbol{\Psi})`: Normal-Gamma (MVNIG) posterior update
  using sufficient statistics from the E-step (see :doc:`factor_analysis` for
  the static update equations)
- :math:`q(\mathbf{F}, \mathbf{Q})`: MNIW or MVN posterior update using
  state transition sufficient statistics
- :math:`q(\boldsymbol{\tau})`: Gamma posterior update (ARD hyperparameters)


PX-VB Rotation Step
---------------------

After the standard VB-EM updates, we find the optimal rotation :math:`\mathbf{R}`
that minimizes the KL divergence. The rotation is applied to the posterior
distributions as follows [3]_:

**Rotating** :math:`q(\mathbf{Z})`:

.. math::

   q(\tilde{\mathbf{Z}}) = q(\mathbf{R}\mathbf{Z})

where the sufficient statistics transform as:

.. math::

   \langle \tilde{\mathbf{z}}_t \rangle &= \mathbf{R} \langle \mathbf{z}_t \rangle \\
   \langle \tilde{\mathbf{z}}_t \tilde{\mathbf{z}}_t^\top \rangle &=
   \mathbf{R} \langle \mathbf{z}_t \mathbf{z}_t^\top \rangle \mathbf{R}^\top

**Rotating** :math:`q(\mathbf{H})`:

.. math::
   :label: rotate-h

   \tilde{\boldsymbol{\mu}}_d^H &= \mathbf{R}^{-\top} \boldsymbol{\mu}_d^H \\
   \tilde{\boldsymbol{\Sigma}}_d^H &= \mathbf{R}^{-\top} \boldsymbol{\Sigma}_d^H \mathbf{R}^{-1}

**Rotating** :math:`q(\mathbf{F})`:

.. math::
   :label: rotate-f

   \tilde{\boldsymbol{\mu}}^F &= \mathbf{R} \boldsymbol{\mu}^F \mathbf{R}^{-1} \\
   \tilde{\boldsymbol{\Sigma}}^F &\text{ transforms accordingly}


Finding the Optimal Rotation
-----------------------------

The optimal rotation :math:`\mathbf{R}` is found by minimizing the expected
negative log-prior of the rotated model parameters:

.. math::
   :label: px-loss

   \mathcal{L}(\mathbf{R}) = \mathbb{E}_q\left[-\ln p(\tilde{\mathbf{H}}, \tilde{\mathbf{F}}, \tilde{\mathbf{z}}_0)\right]

This objective decomposes into four terms:

.. math::

   \mathcal{L}(\mathbf{R}) = \mathcal{L}_{\text{init}} + \mathcal{L}_{\text{emission}} + \mathcal{L}_{\text{dyn-prior}} + \mathcal{L}_{\text{dyn-lik}}

where:

- :math:`\mathcal{L}_{\text{init}}`: expected negative log-prior of the rotated initial state
  :math:`\tilde{\mathbf{z}}_0 = \mathbf{R}^{-1}\mathbf{z}_0`

- :math:`\mathcal{L}_{\text{emission}}`: expected negative log-prior of the rotated emission
  matrix. The rotation uses a block-diagonal structure
  :math:`\mathbf{R}_{\text{block}} = \text{blkdiag}(\mathbf{R}, \mathbf{I})` so that only the
  first :math:`K` columns of the augmented emission matrix are rotated, preserving the input
  weights and bias columns:

  .. math::

     \mathcal{L}_{\text{emission}} = \frac{1}{2} \sum_d \text{tr}\left(\boldsymbol{\Lambda}_d \, \mathbf{R}_{\text{block}}^\top \mathbb{E}[\tilde{\mathbf{h}}_d \tilde{\mathbf{h}}_d^\top] \mathbf{R}_{\text{block}}\right) - D \ln|\mathbf{R}|

- :math:`\mathcal{L}_{\text{dyn-prior}}`: expected negative log-prior of the rotated dynamics
  :math:`\tilde{\mathbf{F}} = \mathbf{R}^{-1}\mathbf{F}\mathbf{R}_{\text{block}}`

- :math:`\mathcal{L}_{\text{dyn-lik}}`: expected negative log-likelihood of the dynamics residuals
  under the rotated state noise

For the **static case** (FA/PCA), only :math:`\mathcal{L}_{\text{init}}` and
:math:`\mathcal{L}_{\text{emission}}` are needed since there are no dynamics.

**Numerical optimization:** In ``sppcax``, the rotation :math:`\mathbf{R}` is found by
minimizing Eq. :eq:`px-loss` using gradient descent with Anderson acceleration [7]_.
Starting from :math:`\mathbf{R} = \mathbf{I}`, the optimizer runs for a fixed number
of steps (default: 32, learning rate: 0.001). Anderson acceleration with memory :math:`m=1`
is applied after the first gradient step to improve convergence. If the final loss exceeds
the initial loss, the optimizer falls back to the identity rotation.


Static Case Simplification
----------------------------

For static Factor Analysis (:math:`\mathbf{F} = \mathbf{0}`,
:math:`\mathbf{Q} = \mathbf{I}`), the rotation only affects :math:`\mathbf{H}`
and :math:`\mathbf{Z}`. The dynamics terms vanish, and the loss simplifies to:

.. math::

   \mathcal{L}(\mathbf{R}) = \mathcal{L}_{\text{init}} + \mathcal{L}_{\text{emission}}


PX-VBEM Algorithm Summary
---------------------------

.. rst-class:: algorithm

**Algorithm 1: PX-VBEM for Dynamic Factor Analysis**

| **Input:** observations :math:`\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_T\}`, prior hyperparameters
| **Output:** approximate posterior :math:`q(\mathbf{Z})q(\mathbf{H}, \boldsymbol{\Psi})q(\mathbf{F}, \mathbf{Q})q(\boldsymbol{\tau})`
|
| 1. Initialize all approximate posteriors
| 2. **repeat** until convergence:
|    a. **VBE-step:** update :math:`q(\mathbf{Z})` via Kalman smoother using expected parameters
|    b. **VBM-step:**
|       - Update :math:`q(\mathbf{H}, \boldsymbol{\Psi})` using emission sufficient statistics
|       - Update :math:`q(\mathbf{F}, \mathbf{Q})` using dynamics sufficient statistics
|       - Update :math:`q(\boldsymbol{\tau})` using ARD update equations
|    c. **PX-step (rotation):**
|       - Minimize :math:`\mathcal{L}(\mathbf{R})` (Eq. :eq:`px-loss`) via gradient descent with Anderson acceleration
|       - Remap :math:`q(\mathbf{Z}) \to q(\mathbf{R}\mathbf{Z})` (sufficient statistics)
|       - Remap :math:`q(\mathbf{H}) \to q(\mathbf{H}\mathbf{R}^{-1})` via Eq. :eq:`rotate-h`
|       - Remap :math:`q(\mathbf{F}) \to q(\mathbf{R}\mathbf{F}\mathbf{R}^{-1})` via Eq. :eq:`rotate-f`
|    d. Compute ELBO for convergence monitoring
| 3. **return** approximate posterior


See Also
========

For practical examples comparing EM, VB-EM, and PX-VB:

- :doc:`examples/test_px_em_fa` — Parameter expansion for Factor Analysis
- :doc:`examples/test_px_em_dfa` — Parameter expansion for Dynamic Factor Analysis


References
==========

.. [1] Qi, Y., & Jaakkola, T. S. (2006). Parameter expanded variational Bayesian
   methods. In *Advances in Neural Information Processing Systems 19*, pp. 1097-1104.

.. [3] Luttinen, J., Raiko, T., & Ilin, A. (2014). Linear state-space model with
   time-varying dynamics. In *Machine Learning and Knowledge Discovery in Databases*
   (ECML PKDD 2014), LNCS 8725, pp. 338-353.

.. [7] Luttinen, J., & Ilin, A. (2010). Transformations in variational Bayesian
   factor analysis to speed up learning. *Neurocomputing*, 73, 1093-1102.
