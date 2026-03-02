"""Example of using BayesianPCA (unified model) with synthetic data."""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from sppcax.models import BayesianPCA


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
    """Run BayesianPCA example."""
    key = jr.PRNGKey(0)
    X, W_true, Z_true = generate_synthetic_data(key)
    n_samples, n_features = X.shape
    n_components = W_true.shape[1]

    # Create and fit BayesianPCA model (PCA = FA with isotropic noise = DFA with F=0, Q=I, R=sigma^2*I)
    model = BayesianPCA(n_components=n_components, n_features=n_features, key=key)
    params, props = model.initialize(key)
    params, elbos = model.fit_em(params, props, X, key=key, num_iters=50, verbose=False)

    # Transform: project data to latent space
    qz = model.transform(params, X)

    # Inverse transform: reconstruct observations
    X_recon = model.inverse_transform(params, qz)

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.scatter(Z_true[:, 0], Z_true[:, 1], alpha=0.5)
    plt.title("True Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.subplot(132)
    plt.scatter(qz.mean[:, 0], qz.mean[:, 1], alpha=0.5)
    plt.title("Estimated Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.subplot(133)
    W_est = params.emissions.weights  # (D, K)
    plt.imshow(jnp.abs(jnp.corrcoef(W_true.T, W_est.T)), cmap="coolwarm")
    plt.title("Loading Matrix Correlation")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Print model statistics
    mse = jnp.mean(jnp.square(X - X_recon.mean))
    print(f"Reconstruction MSE: {mse:.4f}")
    print(f"Final ELBO: {elbos[-1]:.2f}")


if __name__ == "__main__":
    main()
