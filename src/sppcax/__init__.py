"""Sparse Probabilistic Principal Component Analysis with Bayesian Model Reduction."""

from . import distributions, models
from .models import PPCA

__all__ = [
    "distributions",
    "models",
    "PPCA",
]
