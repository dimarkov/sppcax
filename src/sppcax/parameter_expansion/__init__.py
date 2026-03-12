"""Parameter expansion (PX-VB) routines for accelerating variational inference."""

from .rotation import (
    compute_px_rotation_numerical,
    rotate_distribution,
)

__all__ = [
    "compute_px_rotation_numerical",
    "rotate_distribution",
]
