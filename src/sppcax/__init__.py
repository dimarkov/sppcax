"""Sparse Probabilistic Principal Component Analysis with Bayesian Model Reduction."""

from . import distributions, models
from .models import fit, transform, inverse_transform

__version__ = "0.1.0"

__all__ = ["distributions", "models", "fit", "transform", "inverse_transform"]
