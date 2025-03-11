=====
Usage
=====

This section provides examples of how to use the ``sppcax`` package for Bayesian Factor Analysis and Model Reduction.

Basic Example
=============

Here's a simple example of using the Bayesian Factor Analysis model:

.. code-block:: python

    import jax.numpy as jnp
    import jax.random as jr
    from sppcax.models.factor_analysis import PPCA

    # Generate random data
    key = jr.PRNGKey(0)
    n_samples, n_features, n_components = 100, 20, 5

    # Create a model with 5 components
    model = PPCA(n_components=n_components,
                 n_features=n_features,
                 random_state=key)

    # Generate some synthetic data
    X = jnp.ones((n_samples, n_features))

    # Fit the model
    model, elbos = model.fit(X, n_iter=50)

    # Transform data to latent space
    latent_representation = model.transform(X)

    # Reconstruct the data
    reconstructed_X = model.inverse_transform(latent_representation)

Factor Analysis vs. PPCA
========================

``sppcax`` provides two main variants of Bayesian Factor Analysis:

1. **Probabilistic PCA (PPCA)**: Uses isotropic noise (same precision for all features)

   .. code-block:: python

       from sppcax.models.factor_analysis import PPCA

       model = PPCA(n_components=5, n_features=20)

2. **Factor Analysis (FA)**: Uses diagonal noise (different precision for each feature)

   .. code-block:: python

       from sppcax.models.factor_analysis import FactorAnalysis

       model = FactorAnalysis(n_components=5, n_features=20)

Handling Missing Data
=====================

Both models can handle missing data by providing a mask:

.. code-block:: python

    import jax.numpy as jnp
    from sppcax.models.factor_analysis import PPCA

    # Data with some missing values (marked as False in the mask)
    data = jnp.ones((100, 20))
    mask = jnp.ones((100, 20), dtype=bool)

    # Set some values as missing
    mask = mask.at[10:20, 5:10].set(False)

    # Create a model with the mask
    model = PPCA(n_components=5,
                 n_features=20,
                 data_mask=mask)

    # Fit the model
    model, elbos = model.fit(data)

    # Transform can use the mask to handle missing values in new data
    latent = model.transform(data, use_data_mask=True)

Bayesian Model Reduction
========================

The Bayesian Model Reduction (BMR) algorithm can be used to prune unnecessary parameters in the loading matrix:

.. code-block:: python

    import jax.numpy as jnp
    from sppcax.models.factor_analysis import PPCA
    from sppcax.bmr.delta_f import compute_delta_f

    # Fit a model
    model = PPCA(n_components=5, n_features=20)
    data = jnp.ones((100, 20))
    model, _ = model.fit(data)

    # Compute delta F for each parameter in the loading matrix
    delta_f_values = compute_delta_f(
        posterior=model.W_dist,  # Posterior distribution
        prior=model.W_dist.__class__(  # Prior distribution
            loc=jnp.zeros_like(model.W_dist.mvn.mean),
            mask=model.W_dist.mask,
            alpha=model.W_dist.gamma.alpha0,
            beta=model.W_dist.gamma.beta0
        )
    )

    # Parameters with large positive delta F are candidates for pruning
    # This would typically be followed by updating the model with the pruned parameters

Advanced Usage
==============

For more advanced usage, including custom priors, mixed likelihood models, and integration with other JAX-based libraries, please refer to the API documentation.
