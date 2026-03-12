===============
Theory Overview
===============

This section provides the mathematical foundations for the algorithms implemented in ``sppcax``.
The primary model is Dynamic Factor Analysis (DFA), a Linear Gaussian State Space Model with
ARD priors for automatic sparsification. Factor Analysis (FA) and Probabilistic PCA are
obtained as special cases when there are no temporal dynamics.

.. toctree::
   :maxdepth: 2

   dynamic_factor_analysis
   inference_methods
   factor_analysis
   parameter_expansion
   model_reduction
   examples/index
