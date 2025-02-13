"""Base model class."""

from typing import Any

import equinox as eqx

from ..types import Array


class Model(eqx.Module):
    """Base class for all models."""

    def __init__(self, **kwargs: Any):
        """Initialize model parameters."""
        pass

    def fit(self, X: Array, **kwargs: Any) -> "Model":
        """Fit the model to data.

        Args:
            X: Input data matrix of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments.

        Returns:
            self: The fitted model.
        """
        raise NotImplementedError

    def transform(self, X: Array) -> Array:
        """Apply dimensionality reduction to X.

        Args:
            X: Input data matrix of shape (n_samples, n_features).

        Returns:
            X_new: Transformed data matrix.
        """
        raise NotImplementedError

    def fit_transform(self, X: Array, **kwargs: Any) -> Array:
        """Fit the model and apply dimensionality reduction to X.

        Args:
            X: Input data matrix of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments.

        Returns:
            X_new: Transformed data matrix.
        """
        return self.fit(X, **kwargs).transform(X)

    def inverse_transform(self, X: Array) -> Array:
        """Transform data back to its original space.

        Args:
            X: Input data matrix in transformed space.

        Returns:
            X_original: Data matrix in original space.
        """
        raise NotImplementedError

    def score(self, X: Array) -> Array:
        """Return the log likelihood of X under the model.

        Args:
            X: Input data matrix of shape (n_samples, n_features).

        Returns:
            log_likelihood: Log likelihood of X.
        """
        raise NotImplementedError
