"""Sparse Probabilistic Principal Component Analysis with Bayesian Model Reduction."""

from . import distributions, models
from .models import (
    BayesianDynamicFactorAnalysis,
    BayesianFactorAnalysis,
    BayesianPCA,
)

__version__ = "0.1.0"

__all__ = [
    "distributions",
    "models",
    "BayesianDynamicFactorAnalysis",
    "BayesianFactorAnalysis",
    "BayesianPCA",
]
