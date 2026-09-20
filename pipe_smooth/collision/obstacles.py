"""Static obstacles for collision broadphase / GJK.

Each type: aabb(margin) -> (mn, mx) and to_coal() -> (shape, Transform3s).
Native coal Sphere / Capsule / Box. Four-point hulls live in gjk.Hull4.
"""

# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import coal

_EPS = 1e-12


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

    def to_coal(self) -> tuple[coal.Sphere, coal.Transform3s]:
        """Native coal.Sphere at c."""
        T = coal.Transform3s.Identity()
        T.setTranslation(np.asarray(self.c, float))
        return coal.Sphere(float(self.r)), T


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

    def to_coal(self) -> tuple[coal.Capsule, coal.Transform3s]:
        """Native coal.Capsule; local +z aligned to b-a (length ||b-a||)."""
        a = np.asarray(self.a, float)
        b = np.asarray(self.b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        T = coal.Transform3s.Identity()
        T.setTranslation(0.5 * (a + b))
        if L < _EPS:
            return coal.Sphere(float(self.r)), T
        T.setQuatRotation(coal.Quaternion.FromTwoVectors(np.array([0.0, 0.0, 1.0]), d / L))
        return coal.Capsule(float(self.r), L), T


@dataclass
class AabbObs:
    """Solid axis-aligned box [mn, mx]."""

    kind: ClassVar[str] = "aabb"
    mn: np.ndarray
    mx: np.ndarray
    name: str = ""

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """AABB of the box, expanded by margin.

        Args:
            margin: extra pad on the box sides

        Returns:
            mn, mx: shape (3,) each
        """
        mn = np.asarray(self.mn, float)
        mx = np.asarray(self.mx, float)
        e = float(margin)
        return mn - e, mx + e

    def to_coal(self):
        """Native coal.Box centered at 0.5*(mn+mx)."""
        mn = np.asarray(self.mn, float)
        mx = np.asarray(self.mx, float)
        size = np.maximum(mx - mn, _EPS)
        T = coal.Transform3s.Identity()
        T.setTranslation(0.5 * (mn + mx))
        return coal.Box(float(size[0]), float(size[1]), float(size[2])), T
