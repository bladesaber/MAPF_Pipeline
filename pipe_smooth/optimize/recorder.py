"""Optimization history + best-so-far; optional full state snapshots for viz."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Recorder:
    """Track J (and optional x) over steps; keep best by objective.

    ``record_state=True`` stores a copy of ``x`` each step so you can unpack /
    refresh graphs later and visualize how geometry evolves.
    ``out_dir``: directory for ``save`` / default dumps (created on init).
    """

    out_dir: str | Path = "runs"
    record_state: bool = False
    phase: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    best_J: float = float("inf")
    best_x: np.ndarray | None = None
    best_step: int = -1
    best_phase: str = ""

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        step: int,
        J: float,
        x: Any | None = None,
        *,
        phase: str | None = None,
        **extra,
    ) -> None:
        """Append one step. Pass ``x`` when ``record_state`` (or always for best)."""
        ph = self.phase if phase is None else phase
        Jf = float(J)
        row: dict[str, Any] = {"step": int(step), "phase": ph, "J": Jf, **extra}
        x_np = None if x is None else np.asarray(x, dtype=np.float64).copy()
        if self.record_state and x_np is not None:
            row["x"] = x_np
        self.history.append(row)

        if Jf < self.best_J and x_np is not None:
            self.best_J = Jf
            self.best_x = x_np
            self.best_step = int(step)
            self.best_phase = ph

    def J_curve(self) -> np.ndarray:
        return np.asarray([h["J"] for h in self.history], dtype=np.float64)

    def states(self) -> list[np.ndarray]:
        """All recorded ``x`` snapshots (empty if ``record_state`` was False)."""
        return [h["x"] for h in self.history if "x" in h]

    def save(self, name: str = "recorder.pkl") -> Path:
        path = Path(name)
        if not path.is_absolute():
            path = self.out_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "Recorder":
        with Path(path).open("rb") as f:
            return pickle.load(f)
