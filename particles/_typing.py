"""Shared numerical types and checks for optional dependencies."""

from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]
Scalar: TypeAlias = float | np.floating[Any]
ScalarOrArray: TypeAlias = Scalar | Array

_T = TypeVar("_T")


def _require(value: _T | None, name: str) -> _T:
    """Narrow an optional value when an operation actually needs it."""
    if value is None:
        raise ValueError(f"{name} must be set before this operation")
    return value
