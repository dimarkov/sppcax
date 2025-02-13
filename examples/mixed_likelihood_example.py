"""Example of using Mixed Likelihood VBPCA."""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from sppcax.models import MixedLikelihoodVBPCA


def generate_synthetic_data(n_samples=500, n_components=2, seed=0):
    """Generate synthetic dataset with mixed types.

    Creates a dataset with:
    - 2 continuous variables
    - 2 binary variables
    - 2 count variables (Poisson)
    """
    key = jr.PRNGKey(seed)

    # Generate latent variables
    key, subkey = jr.split(key)
    Z = jr.normal(subkey, (n_samples, n_components))

    # Generate loading matrices
    key, subkey = jr.split(key)
    W_continuous = jr.normal(subkey, (2, n_components))
    key, subkey = jr.split(key)
    W_binary = jr.normal(subkey, (2, n_components))
    key, subkey = jr.split(key)
    W_poisson = jr.normal(subkey, (2, n_components))

    # Generate continuous observations
    X_continuous = jnp.dot(Z, W_continuous.T)

    # Generate binary observations
    logits_binary = jnp.dot(Z, W_binary.T)
    probs_binary = 1 / (1 + jnp.exp(-logits_binary))
    X_binary = (probs_binary > 0.5).astype(jnp.float32)

    # Generate count observations
    log_rates = jnp.dot(Z, W_poisson.T)
    rates = jnp.exp(log_rates)
    key, subkey = jr.split(key)
    X_poisson = jr.poisson(subkey, rates)

    # Combine all features
    X = jnp.concatenate([X_continuous, X_binary, X_poisson], axis=1)

    return X, Z


def main():
    # Generate synthetic data
    X, Z_true = generate_synthetic_data()

    # Define feature types
    feature_types = {
        "normal": [0, 1],  # First two features are continuous
        "bernoulli": [2, 3],  # Next two are binary
        "poisson": [4, 5],  # Last two are count data
    }

    # Create and fit model
    model = MixedLikelihoodVBPCA(n_components=2, feature_types=feature_types, random_state=jr.PRNGKey(0))

    model.fit(X)

    # Transform data to latent space
    Z_inferred = model.transform(X)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot true latent space
    plt.subplot(131)
    plt.scatter(Z_true[:, 0], Z_true[:, 1], alpha=0.5)
    plt.title("True Latent Space")
    plt.xlabel("Z1")
    plt.ylabel("Z2")

    # Plot inferred latent space
    plt.subplot(132)
    plt.scatter(Z_inferred[:, 0], Z_inferred[:, 1], alpha=0.5)
    plt.title("Inferred Latent Space")
    plt.xlabel("Z1")
    plt.ylabel("Z2")

    # Plot reconstruction quality
    X_rec = model.inverse_transform(Z_inferred)

    plt.subplot(133)
    # Plot continuous reconstruction
    plt.scatter(X[:, 0], X_rec["normal"][:, 0], alpha=0.5, label="Continuous")
    # Plot binary reconstruction
    plt.scatter(X[:, 2], X_rec["bernoulli"][:, 0], alpha=0.5, label="Binary")
    # Plot count reconstruction
    plt.scatter(X[:, 4], X_rec["poisson"][:, 0], alpha=0.5, label="Count")
    plt.plot(
        [0, max(X[:, 4].max(), X_rec["poisson"][:, 0].max())],
        [0, max(X[:, 4].max(), X_rec["poisson"][:, 0].max())],
        "k--",
    )  # Diagonal line
    plt.title("Reconstruction Quality")
    plt.xlabel("True Values")
    plt.ylabel("Reconstructed Values")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
