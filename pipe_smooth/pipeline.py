"""Stage-1 pipeline (§8.3 / §8.5): load → tree → junction → Hermite → soft → hard → accept."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import yaml

from pipe_smooth.collision import (
    AabbObs,
    CapsuleObs,
    SphereObs,
    find_broadphase_pairs,
    find_narrowphase_pairs,
)
from pipe_smooth.costs.union import find_union_pairs
from pipe_smooth.geometry.assemble import OptimizeGraph, assemble_hermite_graph
from pipe_smooth.optimize.adam import adam_annealed
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.optimize.pack import PackLayout, layout, unpack
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.optimize.sqp import elastic_qp
from pipe_smooth.topology.junction import build_junctions_all
from pipe_smooth.topology.tree import build_trees

try:
    from pipe_smooth.collision import find_hull_narrowphase_pairs
except ImportError:  # coal optional
    find_hull_narrowphase_pairs = None  # type: ignore[assignment]

Obstacle = SphereObs | CapsuleObs | AabbObs


def load_prepared(path: str | Path) -> dict:
    """Load a prepared env YAML (from ``example_load_env`` / ``data/``)."""
    return yaml.safe_load(Path(path).read_text())


def obstacles_from_env(env: dict) -> list[Obstacle]:
    """Parse optional ``obstacles`` block; empty if absent (prepared MVP dumps)."""
    raw = env.get("obstacles") or {}
    out: list[Obstacle] = []
    for i, s in enumerate(raw.get("spheres") or []):
        out.append(
            SphereObs(
                c=np.asarray(s["c"], float),
                r=float(s["r"]),
                name=str(s.get("name", f"sphere{i}")),
            )
        )
    for i, c in enumerate(raw.get("capsules") or []):
        out.append(
            CapsuleObs(
                a=np.asarray(c["a"], float),
                b=np.asarray(c["b"], float),
                r=float(c["r"]),
                name=str(c.get("name", f"capsule{i}")),
            )
        )
    for i, b in enumerate(raw.get("aabbs") or []):
        out.append(
            AabbObs(
                mn=np.asarray(b["mn"], float),
                mx=np.asarray(b["mx"], float),
                name=str(b.get("name", f"aabb{i}")),
            )
        )
    return out


def build_graphs(
    env: dict,
    length_max: float = 8.0,
    *,
    tau0: float = 2.0,
    alpha: float = 2.0,
) -> list[OptimizeGraph]:
    """§6.1–6.3 / 6.5: trees → junctions → Hermite graphs + refresh samples."""
    trees = build_trees(env)
    domain = env.get("grid") or {}
    parts = build_junctions_all(trees, domain=domain, tau0=tau0, alpha=alpha)
    graphs = [assemble_hermite_graph(p, length_max=length_max) for p in parts]
    for g in graphs:
        g.refresh(backend="numpy")
    return graphs


def _refresh_capsule(graphs, obstacles, margin: float):
    broad_obs, broad_grp = find_broadphase_pairs(graphs, obstacles, margin=margin)
    return find_narrowphase_pairs(
        graphs, obstacles, broad_obs, broad_grp, margin=margin
    )


def _refresh_hull(graphs, obstacles, margin: float):
    if find_hull_narrowphase_pairs is None:
        raise ImportError("hull narrowphase needs coal (pipe_smooth.collision.gjk)")
    broad_obs, broad_grp = find_broadphase_pairs(graphs, obstacles, margin=margin)
    return find_hull_narrowphase_pairs(
        graphs, obstacles, broad_obs, broad_grp, margin=margin
    )


def _refresh_union(graphs: Sequence[OptimizeGraph]):
    return {g.id: find_union_pairs(g) for g in graphs}


def accept_clearance(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    kind: Literal["capsule", "hull"] = "hull",
    margin: float = 0.0,
    tol: float = 0.0,
) -> tuple[bool, float]:
    """Dense accept (§8.3): True iff max(d_req - d_geom) <= tol on active pairs."""
    if kind == "hull":
        narrow_obs, narrow_grp = _refresh_hull(graphs, obstacles, margin)
    else:
        narrow_obs, narrow_grp = _refresh_capsule(graphs, obstacles, margin)
    viol = 0.0
    for p in list(narrow_obs) + list(narrow_grp):
        viol = max(viol, float(p.d_req) - float(p.d_geom))
    return viol <= tol, viol


def run_stage1(
    env: dict,
    *,
    weights: LossWeights | None = None,
    obstacles: Sequence[Obstacle] | None = None,
    margin: float = 0.0,
    length_max: float = 8.0,
    soft_steps: int = 200,
    hard_steps: int = 20,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    soft: Literal["adam_annealed", "none"] = "adam_annealed",
    hard: Literal["elastic_qp", "none"] = "elastic_qp",
    verbose: bool = False,
) -> dict[str, Any]:
    """Full Stage-1 schedule (§8.3): soft capsule AD → hard hull SQP → accept."""
    graphs = build_graphs(env, length_max=length_max)
    obs = list(obstacles if obstacles is not None else obstacles_from_env(env))
    lay = layout(graphs)
    w = weights or LossWeights()
    rec = Recorder(out_dir=out_dir, record_state=record_state, phase="stage1")

    def refresh_soft(gs):
        return _refresh_capsule(gs, obs, margin)

    def refresh_hard(gs):
        return _refresh_hull(gs, obs, margin)

    def refresh_union(gs):
        return _refresh_union(gs)

    hist_soft: list[float] = []
    hist_hard: list[float] = []

    if soft == "adam_annealed":
        rec.phase = "adam_annealed"
        _, hist_soft = adam_annealed(
            graphs,
            lay,
            w,
            steps=soft_steps,
            clear_kind="capsule",
            refresh_clear=refresh_soft,
            refresh_union=refresh_union if w.lam != 0.0 else None,
            recorder=rec,
            verbose=verbose,
        )

    if hard == "elastic_qp":
        rec.phase = "elastic_qp"
        _, hist_hard = elastic_qp(
            graphs,
            lay,
            w,
            steps=hard_steps,
            clear_kind="hull",
            refresh_clear=refresh_hard,
            refresh_union=refresh_union if w.lam != 0.0 else None,
            recorder=rec,
            verbose=verbose,
        )

    if rec.best_x is not None:
        unpack(rec.best_x, graphs, lay, backend="numpy")
        for g in graphs:
            g.refresh(backend="numpy")

    ok, viol = accept_clearance(graphs, obs, kind="hull", margin=margin)
    rec.save("recorder.pkl")
    lay.export(Path(rec.out_dir) / "layout.json")
    for g in graphs:
        g.export(Path(rec.out_dir) / f"graph_{g.id}.json")

    return {
        "graphs": graphs,
        "layout": lay,
        "recorder": rec,
        "history_soft": hist_soft,
        "history_hard": hist_hard,
        "accepted": ok,
        "max_viol": viol,
        "obstacles": obs,
    }


def main() -> None:
    import argparse

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="pipe_smooth Stage-1 pipeline (§8)")
    parser.add_argument(
        "env",
        nargs="?",
        default=str(root / "data" / "instance_0.yaml"),
        help="prepared env YAML",
    )
    parser.add_argument("-o", "--out-dir", default=str(root / "runs" / "stage1"))
    parser.add_argument("--soft-steps", type=int, default=50)
    parser.add_argument("--hard-steps", type=int, default=10)
    parser.add_argument("--record-state", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    env = load_prepared(args.env)
    out = run_stage1(
        env,
        soft_steps=args.soft_steps,
        hard_steps=args.hard_steps,
        out_dir=args.out_dir,
        record_state=args.record_state,
        verbose=args.verbose,
    )
    print(
        f"accepted={out['accepted']}  max_viol={out['max_viol']:.6g}  "
        f"groups={len(out['graphs'])}  out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
