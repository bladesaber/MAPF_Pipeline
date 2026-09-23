"""Stage-1 pipeline (§8): load → tree → junction → Hermite → one optimizer → report."""

from __future__ import annotations

import json
import time
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
from pipe_smooth.costs.path import kappa_menger
from pipe_smooth.costs.union import find_union_pairs
from pipe_smooth.costs.volume import tube_volume
from pipe_smooth.geometry.assemble import OptimizeGraph, assemble_hermite_graph
from pipe_smooth.optimize.adam import adam, adam_annealed
from pipe_smooth.optimize.ipopt import interior_nlp
from pipe_smooth.optimize.langevin import langevin
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.optimize.pack import layout, unpack
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
ClearKind = Literal["capsule", "hull", "ignore"]


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
    out_dir: str | Path,
    store_cfg: dict[str, Any],
) -> list[OptimizeGraph]:
    """§6.1–6.3 / 6.5: trees → junctions → Hermite graphs + refresh samples.

    Dump ``tree_*.json``, ``partition_*.json``, ``graph_*_init.json`` under
    ``out_dir``; paths recorded in ``store_cfg``.
    """
    trees = build_trees(env)
    domain = env.get("grid") or {}
    parts = build_junctions_all(trees, domain=domain, tau0=tau0, alpha=alpha)
    graphs = [assemble_hermite_graph(p, length_max=length_max) for p in parts]
    for g in graphs:
        g.refresh(backend="numpy")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store_cfg.setdefault("tree", {})
    store_cfg.setdefault("partition", {})
    store_cfg.setdefault("graph_init", {})
    for t in trees:
        path = out / f"tree_{t.id}.json"
        t.export(path)
        store_cfg["tree"][t.id] = str(path)
    for p in parts:
        path = out / f"partition_{p.id}.json"
        p.export(path)
        store_cfg["partition"][p.id] = str(path)
    for g in graphs:
        path = out / f"graph_{g.id}_init.json"
        g.export(path)
        store_cfg["graph_init"][g.id] = str(path)

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


def clearance_report(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    kind: Literal["capsule", "hull", "ignore"] = "hull",
    margin: float = 0.0,
) -> dict[str, Any]:
    """Hard clearance report (§8.1.1). Same proxy as the loop; strip soft ``margin``.

    Soft loop uses ``d_req = radii + margin``. Feasibility uses
    ``d_req_hard = d_req - margin``; ``n_hit`` / ``ok`` count only hard violations
    (``d_req_hard - d_geom > 0``). ``max_shortfall`` is that max (0 if none).
    ``hard_obs`` / ``hard_grp`` are JSON-friendly ``{pair, shortfall}`` via ``.export()``.
    ``kind='ignore'`` skips detection (debug).
    """
    if kind == "ignore":
        return {
            "kind": kind,
            "margin": float(margin),
            "n_hit": 0,
            "n_hit_obs": 0,
            "n_hit_grp": 0,
            "max_shortfall": 0.0,
            "ok": True,
            "hard_obs": [],
            "hard_grp": [],
        }

    if kind == "hull":
        narrow_obs, narrow_grp = _refresh_hull(graphs, obstacles, margin)
    else:
        narrow_obs, narrow_grp = _refresh_capsule(graphs, obstacles, margin)

    def hard_pairs(pairs):
        out = []
        for p in pairs:
            d_req = getattr(p, "d_req", None)
            if d_req is None:
                d_req = p.hits[0].d_req
            s = (float(d_req) - float(margin)) - float(p.d_geom)
            if s > 0.0:
                out.append({"pair": p.export(), "shortfall": float(s)})
        return out

    hard_obs, hard_grp = hard_pairs(narrow_obs), hard_pairs(narrow_grp)
    hard = hard_obs + hard_grp
    max_sf = max((h["shortfall"] for h in hard), default=0.0)
    return {
        "kind": kind,
        "margin": float(margin),
        "n_hit": len(hard),
        "n_hit_obs": len(hard_obs),
        "n_hit_grp": len(hard_grp),
        "max_shortfall": max_sf,
        "ok": len(hard) == 0,
        "hard_obs": hard_obs,
        "hard_grp": hard_grp,
    }


def path_report(
    graphs: Sequence[OptimizeGraph],
    *,
    voxel: float = 0.25,
) -> dict[str, Any]:
    """Usage-path metrics (§8.1.2): per-usage → group → aggregate.

    ``L`` / ``kappa_max`` from each usage polyline. ``V`` is voxel-union of
    sample spheres (segments once) at group / aggregate.
    """
    groups: dict[str, Any] = {}
    for g in graphs:
        usages: dict[str, Any] = {}
        seen_seg: set[str] = set()
        g_xyz, g_r = [], []
        for path in g.usage_paths:
            pts, rads = [], []
            for sid, rev in zip(path.segments, path.reverse):
                _, xyz, radius = g.segments[sid].samples(backend="numpy")
                if sid not in seen_seg:
                    seen_seg.add(sid)
                    g_xyz.append(xyz)
                    g_r.append(radius)
                if rev:
                    xyz, radius = xyz[::-1], radius[::-1]
                if pts:
                    xyz, radius = xyz[1:], radius[1:]
                pts.append(xyz)
                rads.append(radius)
            if not pts:
                raise ValueError(f"path has no samples")
            xyz = np.concatenate(pts, axis=0)
            radius = np.concatenate(rads, axis=0)
            kappa = np.asarray(
                kappa_menger(xyz, t_start=path.v_start, t_end=path.v_end)
            )
            key = f"{path.usage[0]}->{path.usage[1]}"
            usages[key] = {
                "L": float(np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1).sum()),
                "kappa_max": float(np.max(kappa * radius)) if kappa.size else 0.0,
                "kappa": kappa.tolist(),
            }
        g_xyz = np.concatenate(g_xyz, axis=0) if g_xyz else np.zeros((0, 3))
        g_r = np.concatenate(g_r, axis=0) if g_r else np.zeros((0,))
        groups[g.id] = {
            "usages": usages,
            "L": sum(u["L"] for u in usages.values()),
            "kappa_max": max((u["kappa_max"] for u in usages.values()), default=0.0),
            "V": tube_volume(g_xyz, g_r, voxel=voxel),
        }
    agg = {
        "L": sum(g["L"] for g in groups.values()),
        "kappa_max": max((g["kappa_max"] for g in groups.values()), default=0.0),
        "V": sum(g["V"] for g in groups.values()),
    }
    return {"groups": groups, "aggregate": agg}


def solve_report(
    recorder: Recorder,
    *,
    wall_time: float | None = None,
    late_frac: float = 0.25,
) -> dict[str, Any]:
    """Solve-process metrics (§8.1.3): time, steps, J drop / late stability."""
    J = recorder.J_curve()
    n = int(J.size)
    if n == 0:
        raise ValueError("recorder has no J values")
    J0, JT = float(J[0]), float(J[-1])
    i0 = max(0, int((1.0 - late_frac) * n))
    late = J[i0:]
    dJ = np.diff(J)
    return {
        "wall_time": wall_time,
        "steps": n,
        "J0": J0,
        "JT": JT,
        "best_J": float(recorder.best_J),
        "rel_drop": (J0 - JT) / max(abs(J0), 1e-12),
        "late_std": float(np.std(late)),
        "uphill_frac": float(np.mean(dJ > 0.0)) if dJ.size else 0.0,
    }


def run_stage1(
    env: dict,
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    weights: LossWeights | None = None,
    margin: float = 0.0,
    length_max: float = 8.0,
    steps: int = 200,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build graphs, run **one** optimizer, then §8.1 reports."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    out = Path(out_dir)
    store_cfg: dict[str, Any] = {"out_dir": str(out)}

    graphs = build_graphs(
        env, length_max=length_max, out_dir=out, store_cfg=store_cfg
    )
    obs = [] if clear_kind == "ignore" else obstacles_from_env(env)
    lay = layout(graphs)
    layout_path = out / "layout.json"
    lay.export(layout_path)
    store_cfg["layout"] = str(layout_path)

    w = weights or LossWeights()
    weights_path = out / "weights.json"
    weights_path.write_text(json.dumps(asdict(w), indent=2))
    store_cfg["weights"] = str(weights_path)

    rec = Recorder(out_dir=out, record_state=record_state, phase=method)

    def refresh_clear(gs):
        if clear_kind == "ignore":
            return [], []
        if clear_kind == "hull":
            return _refresh_hull(gs, obs, margin)
        return _refresh_capsule(gs, obs, margin)

    union_cb = (lambda gs: _refresh_union(gs)) if w.lam != 0.0 else None
    # loss clear_kind is capsule|hull only; ignore → empty active set, capsule unused
    kw: dict[str, Any] = dict(
        steps=steps,
        clear_kind="capsule" if clear_kind == "ignore" else clear_kind,
        refresh_clear=refresh_clear,
        refresh_union=union_cb,
        recorder=rec,
        verbose=verbose,
    )

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

    if rec.best_x is not None:
        unpack(rec.best_x, graphs, lay, backend="numpy")
        for g in graphs:
            g.refresh(backend="numpy")

    clear = clearance_report(graphs, obs, kind=clear_kind, margin=margin)
    path = path_report(graphs)
    solve = solve_report(rec, wall_time=wall_time)
    store_cfg["recorder"] = str(rec.save("recorder.pkl"))
    store_cfg["graph"] = {}
    for g in graphs:
        gpath = Path(rec.out_dir) / f"graph_{g.id}.json"
        g.export(gpath)
        store_cfg["graph"][g.id] = str(gpath)

    report = {
        "clearance": clear,
        "path": path,
        "solve": solve,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    store_cfg["report"] = str(report_path)
    (out / "store_cfg.json").write_text(json.dumps(store_cfg, indent=2))

    return {
        "graphs": graphs,
        "layout": lay,
        "recorder": rec,
        "method": method,
        "clear_kind": clear_kind,
        "weights": w,
        "clearance": clear,
        "path": path,
        "solve": solve,
        "accepted": clear["ok"],
        "n_hit": clear["n_hit"],
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
    c, p, s = out["clearance"], out["path"]["aggregate"], out["solve"]
    print(
        f"method={out['method']}  clear={out['clear_kind']}  "
        f"ok={c['ok']}  n_hit={c['n_hit']}  max_shortfall={c['max_shortfall']:.4g}  "
        f"L={p['L']:.4g}  κmax={p['kappa_max']:.4g}  V={p['V']:.4g}  "
        f"t={s['wall_time']:.3g}s  rel_drop={s['rel_drop']:.4g}  "
        f"groups={len(out['graphs'])}  out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
