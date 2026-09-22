"""Stage-1 pipeline (§8.5): load → tree → junction → Hermite → one optimizer → accept."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
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
from pipe_smooth.optimize.adam import adam, adam_annealed
from pipe_smooth.optimize.ipopt import interior_nlp
from pipe_smooth.optimize.langevin import langevin
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

METHODS = ("adam", "adam_annealed", "langevin", "elastic_qp", "interior_nlp")
Method = Literal[*METHODS]
ClearKind = Literal["capsule", "hull"]


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
    out_dir: str | Path | None = None,
) -> list[OptimizeGraph]:
    """§6.1–6.3 / 6.5: trees → junctions → Hermite graphs + refresh samples.

    If ``out_dir`` is set, dump ``tree_*.json``, ``partition_*.json``, and
    ``graph_*_init.json`` (pre-optimize) for debug / reload.
    """
    trees = build_trees(env)
    domain = env.get("grid") or {}
    parts = build_junctions_all(trees, domain=domain, tau0=tau0, alpha=alpha)
    graphs = [assemble_hermite_graph(p, length_max=length_max) for p in parts]
    for g in graphs:
        g.refresh(backend="numpy")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for t in trees:
            t.export(out / f"tree_{t.id}.json")
        for p in parts:
            p.export(out / f"partition_{p.id}.json")
        for g in graphs:
            g.export(out / f"graph_{g.id}_init.json")

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
) -> tuple[bool, int]:
    """Dense accept (§8.3): True iff narrowphase returns no active pairs.

    Narrowphase only keeps pairs with ``d_geom < d_req``; empty ⇒ clearance OK.
    """
    if kind == "hull":
        narrow_obs, narrow_grp = _refresh_hull(graphs, obstacles, margin)
    else:
        narrow_obs, narrow_grp = _refresh_capsule(graphs, obstacles, margin)
    n_hit = len(narrow_obs) + len(narrow_grp)
    return n_hit == 0, n_hit


def run_stage1(
    env: dict,
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    weights: LossWeights | None = None,
    obstacles: Sequence[Obstacle] | None = None,
    margin: float = 0.0,
    length_max: float = 8.0,
    steps: int = 200,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build graphs, run **one** optimizer, then accept (§8.5)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    graphs = build_graphs(env, length_max=length_max, out_dir=out_dir)
    obs = list(obstacles if obstacles is not None else obstacles_from_env(env))
    out = Path(out_dir)
    lay = layout(graphs)
    lay.export(out / "layout.json")
    w = weights or LossWeights()
    (out / "weights.json").write_text(json.dumps(asdict(w), indent=2))
    rec = Recorder(out_dir=out, record_state=record_state, phase=method)

    def refresh_clear(gs):
        if clear_kind == "hull":
            return _refresh_hull(gs, obs, margin)
        return _refresh_capsule(gs, obs, margin)

    union_cb = (lambda gs: _refresh_union(gs)) if w.lam != 0.0 else None
    kw: dict[str, Any] = dict(
        steps=steps,
        clear_kind=clear_kind,
        refresh_clear=refresh_clear,
        refresh_union=union_cb,
        recorder=rec,
        verbose=verbose,
    )

    if method == "adam":
        _, history = adam(graphs, lay, w, **kw)
    elif method == "adam_annealed":
        _, history = adam_annealed(graphs, lay, w, **kw)
    elif method == "langevin":
        _, history = langevin(graphs, lay, w, **kw)
    elif method == "elastic_qp":
        _, history = elastic_qp(graphs, lay, w, **kw)
    else:
        _, history = interior_nlp(graphs, lay, w, **kw)

    if rec.best_x is not None:
        unpack(rec.best_x, graphs, lay, backend="numpy")
        for g in graphs:
            g.refresh(backend="numpy")

    ok, n_hit = accept_clearance(graphs, obs, kind=clear_kind, margin=margin)
    rec.save("recorder.pkl")
    for g in graphs:
        g.export(Path(rec.out_dir) / f"graph_{g.id}.json")

    return {
        "graphs": graphs,
        "layout": lay,
        "recorder": rec,
        "history": history,
        "method": method,
        "clear_kind": clear_kind,
        "weights": w,
        "accepted": ok,
        "n_hit": n_hit,
        "obstacles": obs,
    }


def main() -> None:
    import argparse

    root = Path(__file__).resolve().parent
    defaults = LossWeights()
    parser = argparse.ArgumentParser(description="pipe_smooth Stage-1 pipeline (§8)")
    parser.add_argument(
        "env",
        nargs="?",
        default=str(root / "data" / "instance_0.yaml"),
        help="prepared env YAML",
    )
    parser.add_argument("-o", "--out-dir", default=str(root / "runs" / "stage1"))
    parser.add_argument("-m", "--method", default="adam_annealed", choices=list(METHODS))
    parser.add_argument(
        "--clear-kind",
        default="capsule",
        choices=["capsule", "hull"],
        help="narrowphase proxy for clear / accept",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="YAML/JSON with LossWeights fields (overrides defaults; CLI flags win)",
    )
    for f in fields(LossWeights):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=float,
            default=None,
            dest=f.name,
            help=f"LossWeights.{f.name} (default {getattr(defaults, f.name)})",
        )
    parser.add_argument("--record-state", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    w_kw = asdict(defaults)
    if args.weights:
        loaded = yaml.safe_load(Path(args.weights).read_text()) or {}
        w_kw.update({k: float(v) for k, v in loaded.items() if k in w_kw})
    for f in fields(LossWeights):
        val = getattr(args, f.name)
        if val is not None:
            w_kw[f.name] = float(val)
    weights = LossWeights(**w_kw)

    env = yaml.safe_load(Path(args.env).read_text())
    out = run_stage1(
        env,
        method=args.method,
        clear_kind=args.clear_kind,
        weights=weights,
        steps=args.steps,
        out_dir=args.out_dir,
        record_state=args.record_state,
        verbose=args.verbose,
    )
    print(
        f"method={out['method']}  clear={out['clear_kind']}  "
        f"accepted={out['accepted']}  n_hit={out['n_hit']}  "
        f"groups={len(out['graphs'])}  out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
