"""Static obstacle types for collision broadphase / narrowphase.

Each type exposes aabb(margin) -> (mn, mx) for shared AABB broadphase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class SphereObs:
    """Ball obstacle: center c (3,), radius r."""

    kind: ClassVar[str] = "sphere"
    c: np.ndarray
    r: float
    name: str = ""

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned box of the ball, expanded by margin.

        Args:
            margin: extra pad on the ball radius

        Returns:
            mn, mx: shape (3,) each
        """
        c = np.asarray(self.c, float)
        e = float(self.r) + float(margin)
        return c - e, c + e


@dataclass
class CapsuleObs:
    """Capsule obstacle: segment ab with radius r."""

    kind: ClassVar[str] = "capsule"
    a: np.ndarray
    b: np.ndarray
    r: float
    name: str = ""

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """AABB of the two endpoint balls (contains the capsule), + margin.

        Args:
            margin: extra pad on the capsule radius

        Returns:
            mn, mx: shape (3,) each
        """
        a = np.asarray(self.a, float)
        b = np.asarray(self.b, float)
        e = float(self.r) + float(margin)
        return np.minimum(a, b) - e, np.maximum(a, b) + e


@dataclass
class AabbObs:
    """Solid axis-aligned box [mn, mx]."""

    kind: ClassVar[str] = "aabb"
    mn: np.ndarray
    mx: np.ndarray
    name: str = ""

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Box corners, optionally expanded by margin.

        Args:
            margin: pad applied on all sides

        Returns:
            mn, mx: shape (3,) each
        """
        mn = np.asarray(self.mn, float)
        mx = np.asarray(self.mx, float)
        e = float(margin)
        return mn - e, mx + e
