"""Type definitions for sppcax."""

from typing import Tuple, TypeVar

from jaxtyping import Array, Float, PRNGKeyArray

# Type aliases
Scalar = Float[Array, ""]  # scalar
Vector = Float[Array, "dim"]  # vector
Matrix = Float[Array, "rows cols"]  # matrix
Tensor3D = Float[Array, "dim1 dim2 dim3"]  # 3D tensor

# Random key type
PRNGKey = PRNGKeyArray

# Shape type
Shape = Tuple[int, ...]

# Generic type for self references
T = TypeVar("T")
