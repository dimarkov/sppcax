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
    from sppcax.models import PPCA, fit, transform, inverse_transform

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
    key, _key = jr.split(key)
    model, elbos = fit(model, X, n_iter=50, key=_key)

    # Transform data to latent space
    qz = transform(model, X)

    # Reconstruct the data
    reconstructed_X = inverse_transform(model, qz).mean

Factor Analysis vs. PPCA
========================

``sppcax`` provides two main variants of Bayesian Factor Analysis:

1. **Probabilistic PCA (PPCA)**: Uses isotropic noise (same precision for all features)

   .. code-block:: python

       from sppcax.models import PPCA

       model = PPCA(n_components=5, n_features=20)

2. **Factor Analysis (FA)**: Uses diagonal noise (different precision for each feature)

   .. code-block:: python

       from sppcax.models import PFA

       model = PFA(n_components=5, n_features=20)

Handling Missing Data
=====================

Both models can handle missing data by providing a mask:

.. code-block:: python

    import jax.numpy as jnp
    from sppcax.models import PPCA, fit, transform

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
    model, elbos = fit(model, data)

    # Transform can use the mask to handle missing values in new data
    latent = transform(model, data, use_data_mask=True)

Bayesian Model Reduction
========================

The Bayesian Model Reduction (BMR) algorithm can be used to prune unnecessary parameters in the loading matrix:

.. code-block:: python

    import jax.numpy as jnp
    from sppcax.models import PFA
    from sppcax.bmr.delta_f import compute_delta_f

    # Fit a model
    model = PPCA(
        n_components=5,
        n_features=20,
        optimize_with_bmr=True,
        bmr_e_step=True,
        bmr_m_step=True,
        bmr_e_step_opts=('max_iter', 2, 'pi', 0.2)
    )
    # optimize_with_bmr controls Empirical Bayes like hyperparameter optimization
    # bmr_e_step controls BMR during VBE-step, where the posterior over latents is pruned
    # bmr_m_step controls BMR during VBM-step, where the posterior over loading matrix elements is pruned.
    data = jnp.ones((100, 20))
    key, _key = jr.split(key)
    fitted_model, elbos = fit(model, data, n_iter=256, bmr_frequency=16, key=_key)
    # bmr_frequency specifies the frequency of BMR pruning during the VBM-step, here
    # the pruning is performed every 16 steps.

Advanced Usage
==============

For more advanced usage please refer to jupyter notebooks provided in examples directory.
