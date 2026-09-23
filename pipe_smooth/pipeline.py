"""Stage-1 pipeline (§8): load → tree → junction → Hermite → one optimizer → report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import yaml

from pipe_smooth.collision import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.costs.union import find_union_pairs
from pipe_smooth.geometry.assemble import OptimizeGraph, assemble_hermite_graph
from pipe_smooth.optimize.adam import adam, adam_annealed
from pipe_smooth.optimize.ipopt import interior_nlp
from pipe_smooth.optimize.langevin import langevin
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.optimize.pack import layout
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.optimize.sqp import elastic_qp
from pipe_smooth.performance import performance_report, refresh_capsule, refresh_hull
from pipe_smooth.topology.junction import build_junctions_all
from pipe_smooth.topology.tree import build_trees

Obstacle = SphereObs | CapsuleObs | AabbObs

METHODS = ("adam", "adam_annealed", "langevin", "elastic_qp", "interior_nlp")
Method = Literal[*METHODS]
ClearKind = Literal["capsule", "hull", "ignore"]


def load_obstacles(env: dict, clear_kind: ClearKind = "capsule") -> list[Obstacle]:
    """Parse ``obstacles`` from env; ``clear_kind='ignore'`` → empty (debug)."""
    if clear_kind == "ignore":
        return []
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
) -> dict[str, Any]:
    """§6.1–6.3 / 6.5: trees → junctions → Hermite graphs + refresh samples."""
    trees = build_trees(env)
    domain = env.get("grid") or {}
    parts = build_junctions_all(trees, domain=domain, tau0=tau0, alpha=alpha)
    graphs = [assemble_hermite_graph(p, length_max=length_max) for p in parts]
    for g in graphs:
        g.refresh(backend="numpy")
    return {"trees": trees, "parts": parts, "graphs": graphs}


def _refresh_union(graphs: Sequence[OptimizeGraph]):
    return {g.id: find_union_pairs(g) for g in graphs}


def run_optimize(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    weights: LossWeights | None = None,
    margin: float = 0.0,
    steps: int = 200,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
    **opt_kwargs: Any,
) -> dict[str, Any]:
    """One optimizer pass from **current** graph state (solver unpacks best_x).

    ``opt_kwargs`` are forwarded to the selected solver (e.g. ``lr`` for adam).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    lay = layout(list(graphs))
    w = weights or LossWeights()
    rec = Recorder(out_dir=out_dir, record_state=record_state, phase=method)
    obs = list(obstacles)

    def refresh_clear(gs):
        if clear_kind == "ignore":
            return [], []
        if clear_kind == "hull":
            return refresh_hull(gs, obs, margin)
        return refresh_capsule(gs, obs, margin)

    union_cb = (lambda gs: _refresh_union(gs)) if w.lam != 0.0 else None
    kw: dict[str, Any] = dict(
        steps=steps,
        clear_kind="capsule" if clear_kind == "ignore" else clear_kind,
        refresh_clear=refresh_clear,
        refresh_union=union_cb,
        recorder=rec,
        verbose=verbose,
    )
    kw.update(opt_kwargs)

    t0 = time.perf_counter()
    if method == "adam":
        adam(graphs, lay, w, **kw)
    elif method == "adam_annealed":
        adam_annealed(graphs, lay, w, **kw)
    elif method == "langevin":
        langevin(graphs, lay, w, **kw)
    elif method == "elastic_qp":
        elastic_qp(graphs, lay, w, **kw)
    else:
        interior_nlp(graphs, lay, w, **kw)
    wall_time = time.perf_counter() - t0

    return {
        "layout": lay,
        "recorder": rec,
        "weights": w,
        "wall_time": wall_time,
        "method": method,
        "clear_kind": clear_kind,
    }


def run_pipeline(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    weights: LossWeights | None = None,
    margin: float = 0.0,
    steps: int = 200,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
    **opt_kwargs: Any,
) -> dict[str, Any]:
    """Optimize current graphs → report (+ dump under ``out_dir``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store_cfg: dict[str, Any] = {"out_dir": str(out)}
    obs = list(obstacles)

    w = weights or LossWeights()
    weights_path = out / "weights.json"
    weights_path.write_text(json.dumps(asdict(w), indent=2))
    store_cfg["weights"] = str(weights_path)

    opt = run_optimize(
        graphs,
        obs,
        method=method,
        clear_kind=clear_kind,
        weights=w,
        margin=margin,
        steps=steps,
        out_dir=out,
        record_state=record_state,
        verbose=verbose,
        **opt_kwargs,
    )
    lay, rec = opt["layout"], opt["recorder"]
    lay_path = out / "layout.json"
    lay.export(lay_path)
    store_cfg["layout"] = str(lay_path)

    report = performance_report(
        graphs,
        obs,
        rec,
        clear_kind=clear_kind,
        margin=margin,
        wall_time=opt["wall_time"],
    )
    store_cfg["recorder"] = str(rec.save("recorder.pkl"))
    store_cfg["graph"] = {}
    for g in graphs:
        gpath = out / f"graph_{g.id}.json"
        g.export(gpath)
        store_cfg["graph"][g.id] = str(gpath)

    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    store_cfg["report"] = str(report_path)
    (out / "store_cfg.json").write_text(json.dumps(store_cfg, indent=2))

    return {
        "method": method,
        "clear_kind": clear_kind,
        "report": report,
        "out_dir": str(out),
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
    parser.add_argument("-m", "--method", default="adam", choices=list(METHODS))
    parser.add_argument(
        "--clear-kind",
        default="capsule",
        choices=["capsule", "hull", "ignore"],
        help="narrowphase proxy; 'ignore' skips all clearance (debug)",
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
    built = build_graphs(env)
    graphs = built["graphs"]
    obs = load_obstacles(env, args.clear_kind)
    out = run_pipeline(
        graphs,
        obs,
        method=args.method,
        clear_kind=args.clear_kind,
        weights=weights,
        steps=args.steps,
        out_dir=args.out_dir,
        record_state=args.record_state,
        verbose=args.verbose,
    )
    c = out["report"]["clearance"]
    p = out["report"]["path"]["aggregate"]
    s = out["report"]["solve"]
    print(
        f"{out['method']}  {out['clear_kind']}  "
        f"ok={c['ok']}  n_hit={c['n_hit']}  "
        f"L={p['L']:.4g}  κmax={p['kappa_max']:.4g}  "
        f"t={s['wall_time']:.3g}s  → {out['out_dir']}"
    )


if __name__ == "__main__":
    main()
