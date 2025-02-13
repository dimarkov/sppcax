"""Example of using PPCA with synthetic data."""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from sppcax.models import PPCA


def generate_synthetic_data(key: jr.PRNGKey, n_samples: int = 1000):
    """Generate synthetic data with known latent structure.

    Args:
        key: Random number generator key.
        n_samples: Number of samples to generate.

    Returns:
        X: Generated data matrix.
        W: True loading matrix.
        Z: True latent variables.
    """
    # True parameters
    n_features, n_components = 10, 2
    key1, key2, key3 = jr.split(key, 3)

    # Generate true loading matrix with structure
    W = jnp.zeros((n_features, n_components))
    W = W.at[:5, 0].set(jr.normal(key1, (5,)))  # First 5 features load on component 1
    W = W.at[5:, 1].set(jr.normal(key2, (5,)))  # Last 5 features load on component 2

    # Generate latent variables
    Z = jr.normal(key3, (n_samples, n_components))

    # Generate observations with noise
    noise = 0.1 * jr.normal(key3, (n_samples, n_features))
    X = Z @ W.T + noise

    return X, W, Z


def main():
    """Run PPCA example."""
    # Generate synthetic data
    key = jr.PRNGKey(0)
    X, W_true, Z_true = generate_synthetic_data(key)
    n_samples, n_features = X.shape
    n_components = W_true.shape[1]

    # Fit PPCA model
    model = PPCA(n_components=n_components, n_features=n_features, random_state=key)
    Z_est = model.fit_transform(X)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot true latent space
    plt.subplot(131)
    plt.scatter(Z_true[:, 0], Z_true[:, 1], alpha=0.5)
    plt.title("True Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Plot estimated latent space
    plt.subplot(132)
    plt.scatter(Z_est[:, 0], Z_est[:, 1], alpha=0.5)
    plt.title("Estimated Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Plot loading matrix comparison
    plt.subplot(133)
    plt.imshow(jnp.abs(jnp.corrcoef(W_true.T, model.W_.T)), cmap="coolwarm")
    plt.title("Loading Matrix Correlation")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Print model statistics
    X_rec = model.inverse_transform(Z_est)
    mse = jnp.mean(jnp.square(X - X_rec))
    print(f"Reconstruction MSE: {mse:.4f}")

    # Compute explained variance
    X_centered = X - model.mean_
    total_var = jnp.var(X_centered, axis=0).sum()
    explained_var = jnp.sum(jnp.square(model.W_))
    print(f"Explained variance ratio: {explained_var/total_var:.4f}")


if __name__ == "__main__":
    main()
