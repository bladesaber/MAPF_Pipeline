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
UnionKind = Literal["chamfer", "ignore"]

# Solver kwargs (vis_bench ``_OPT_SPECS`` values). Step-size key overridden in two-stage.
OPT_DEFAULTS: dict[str, dict[str, Any]] = {
    "adam": {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "adam_annealed": {
        "eta_max": 1e-2,
        "delta": 1e-2,
        "T0": 1e-3,
        "anneal": "linear",
        "rho": 0.95,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "key": 0,
    },
    "langevin": {
        "eta_max": 1e-2,
        "delta": 1e-2,
        "T0": 1e-3,
        "anneal": "linear",
        "rho": 0.95,
        "eps": 1e-8,
        "key": 0,
    },
    "elastic_qp": {
        "trust": 0.1,
        "eps_band": 0.2,
        "upper_only": True,
        "beta_v": 0.4,
        "w_slack": 1e3,
    },
    "interior_nlp": {
        "max_iter": 5,
        "eps_band": 0.2,
        "upper_only": True,
        "beta_v": 0.4,
    },
}
# Which kw is the outer step / trust size for coarse→refine.
# adam_annealed / langevin: stage size → δ; η_max is synced to δ (Langevin EM rate = trust).
OPT_STEP_KEY: dict[str, str] = {
    "adam": "lr",
    "adam_annealed": "delta",
    "langevin": "delta",
    "elastic_qp": "trust",
}


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


def load_obstacles_file(
    path: str | Path,
    clear_kind: ClearKind = "capsule",
) -> list[Obstacle]:
    """Load obstacles from YAML/JSON.

    Accepts a full env (``obstacles`` key) or a bare
    ``{spheres,capsules,aabbs}`` file (vis_bench ``obstacles.json``).
    """
    path = Path(path)
    text = path.read_text()
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"obstacles file must be a mapping: {path}")
    if "obstacles" not in data and any(k in data for k in ("spheres", "capsules", "aabbs")):
        data = {"obstacles": data}
    return load_obstacles(data, clear_kind)


def dump_obstacles(obstacles: Sequence[Obstacle], path: str | Path) -> None:
    """Write ``{spheres,capsules,aabbs}`` JSON (same schema as ``load_obstacles_file``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spheres, capsules, aabbs = [], [], []
    for i, o in enumerate(obstacles):
        if isinstance(o, SphereObs):
            spheres.append(
                {"c": np.asarray(o.c, float).tolist(), "r": float(o.r), "name": o.name or f"sphere{i}"}
            )
        elif isinstance(o, CapsuleObs):
            capsules.append(
                {
                    "a": np.asarray(o.a, float).tolist(),
                    "b": np.asarray(o.b, float).tolist(),
                    "r": float(o.r),
                    "name": o.name or f"capsule{i}",
                }
            )
        elif isinstance(o, AabbObs):
            aabbs.append(
                {
                    "mn": np.asarray(o.mn, float).tolist(),
                    "mx": np.asarray(o.mx, float).tolist(),
                    "name": o.name or f"aabb{i}",
                }
            )
        else:
            raise TypeError(f"unknown obstacle type: {type(o)!r}")
    path.write_text(
        json.dumps({"spheres": spheres, "capsules": capsules, "aabbs": aabbs}, indent=2)
    )


def load_graphs(path: str | Path) -> list[OptimizeGraph]:
    """Load ``OptimizeGraph`` from a JSON file or a folder of ``graph_*.json``.

    Calls ``refresh`` on each graph.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("graph_*.json"))
        if not files:
            raise FileNotFoundError(f"no graph_*.json in {path}")
    elif path.suffix.lower() == ".json":
        files = [path]
    else:
        raise ValueError(f"graph input must be .json or a directory, got {path}")
    graphs = [OptimizeGraph.load(p) for p in files]
    for g in graphs:
        g.refresh(backend="numpy", store=True)
    return graphs


def resolve_case(
    path: str | Path,
    *,
    clear_kind: ClearKind = "capsule",
    obstacles_path: str | Path | None = None,
) -> tuple[list[OptimizeGraph], list[Obstacle]]:
    """Load graphs + obstacles from YAML env, graph JSON, or a case directory.

    - ``*.yaml`` / ``*.yml``: ``build_graphs`` + obstacles from env
    - ``graph_*.json`` / OptimizeGraph JSON: load graph; obstacles from
      ``obstacles_path`` or sibling ``obstacles.json``
    - directory: all ``graph_*.json``; obstacles from ``obstacles_path`` or
      ``<dir>/obstacles.json``
    ``obstacles_path`` always wins when set.
    """
    path = Path(path)
    obs_override = Path(obstacles_path) if obstacles_path is not None else None

    if path.is_dir() or path.suffix.lower() == ".json":
        graphs = load_graphs(path)
        if obs_override is not None:
            obs = load_obstacles_file(obs_override, clear_kind)
        else:
            sibling = (path if path.is_dir() else path.parent) / "obstacles.json"
            obs = (
                load_obstacles_file(sibling, clear_kind) if sibling.exists() else []
            )
        return graphs, obs

    if path.suffix.lower() in (".yaml", ".yml"):
        env = yaml.safe_load(path.read_text()) or {}
        graphs = build_graphs(env)["graphs"]
        if obs_override is not None:
            obs = load_obstacles_file(obs_override, clear_kind)
        else:
            obs = load_obstacles(env, clear_kind)
        return graphs, obs

    raise ValueError(
        f"unsupported input {path!s}: use env YAML, graph JSON, or a directory "
        f"with graph_*.json"
    )


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
        g.refresh(backend="numpy", store=True)
    return {"trees": trees, "parts": parts, "graphs": graphs}


def _refresh_union(graphs: Sequence[OptimizeGraph]):
    return {g.id: find_union_pairs(g) for g in graphs}


def run_optimize(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    union_kind: UnionKind = "chamfer",
    weights: LossWeights | None = None,
    margin: float = 0.0,
    steps: int = 200,
    p_max_obs: int = 128,
    p_max_grp: int = 128,
    p_max_union: int = 64,
    refresh_every: int = 10,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
    recorder: Recorder | None = None,
    phase: str | None = None,
    **opt_kwargs: Any,
) -> dict[str, Any]:
    """One optimizer pass from **current** graph state (solver unpacks best_x).

    ``p_max_*`` seed packed clear/union buffer lengths (see ``PackLayout``).
    ``refresh_every``: re-run clear/union detection every N outer steps.
    ``opt_kwargs`` are forwarded to the selected solver (e.g. ``lr`` for adam).
    ``union_kind='ignore'`` skips junction Chamfer refresh (even if ``lam≠0``).
    Pass a shared ``recorder`` (and ``phase``) to accumulate history across stages.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if union_kind not in ("chamfer", "ignore"):
        raise ValueError(f"union_kind must be 'chamfer' or 'ignore', got {union_kind!r}")

    lay = layout(
        list(graphs),
        p_max_obs=p_max_obs,
        p_max_grp=p_max_grp,
        p_max_union=p_max_union,
    )    
    w = weights or LossWeights()
    rec = recorder or Recorder(out_dir=out_dir, record_state=record_state)
    rec.phase = phase or method
    obs = list(obstacles)

    def refresh_clear(gs):
        if clear_kind == "ignore":
            return [], []
        if clear_kind == "hull":
            return refresh_hull(gs, obs, margin)
        return refresh_capsule(gs, obs, margin)

    if union_kind == "ignore" or w.lam == 0.0:
        union_cb = None
    else:
        union_cb = lambda gs: _refresh_union(gs)
    kw: dict[str, Any] = dict(
        steps=steps,
        clear_kind="capsule" if clear_kind == "ignore" else clear_kind,
        refresh_clear=refresh_clear,
        refresh_union=union_cb,
        refresh_every=refresh_every,
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
        "union_kind": union_kind,
    }


def run_pipeline(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    method: Method = "adam_annealed",  # type: ignore[assignment]
    clear_kind: ClearKind = "capsule",
    union_kind: UnionKind = "chamfer",
    weights: LossWeights | None = None,
    margin: float | None = None,
    margin_ratio: float = 1.5,
    steps: int | None = None,
    p_max_obs: int = 128,
    p_max_grp: int = 128,
    p_max_union: int = 64,
    refresh_every: int = 10,
    two_stage: bool = False,
    coarse_steps: int = 50,
    refine_steps: int = 50,
    coarse_step_size: float = 0.1,
    refine_step_size: float = 0.01,
    out_dir: str | Path = "runs/stage1",
    record_state: bool = False,
    verbose: bool = False,
    **opt_kwargs: Any,
) -> dict[str, Any]:
    """Optimize current graphs → report (+ dump under ``out_dir``).

    ``two_stage``: coarse then refine. Per stage: ``coarse_steps`` / ``refine_steps``
    and step size (``lr`` / ``delta`` / ``trust`` ← ``*_step_size``;
    adam_annealed/langevin also set ``eta_max = delta``).
    Single-pass (``two_stage=False``): ``steps`` if given (vis_bench), else ``refine_steps``.
    Clearance soft pad: ``margin`` if given, else ``margin_ratio * step_size``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store_cfg: dict[str, Any] = {"out_dir": str(out)}
    obs = list(obstacles)

    w = weights or LossWeights()
    weights_path = out / "weights.json"
    weights_path.write_text(json.dumps(asdict(w), indent=2))
    store_cfg["weights"] = str(weights_path)

    base_kw = {**OPT_DEFAULTS.get(method, {}), **opt_kwargs}
    step_key = OPT_STEP_KEY.get(method)
    n_coarse = int(coarse_steps)
    n_refine = int(refine_steps)
    if two_stage and step_key is not None:
        stage_steps = [
            (coarse_step_size, n_coarse, "coarse"),
            (refine_step_size, n_refine, "refine"),
        ]
    else:
        n_single = int(steps if steps is not None else n_refine)
        stage_steps = [(None, n_single, method)]

    wall_time = 0.0
    opt: dict[str, Any] | None = None
    last_margin = 0.0
    rec = Recorder(out_dir=out, record_state=record_state)
    for step_sz, n_iters, phase in stage_steps:
        kw = dict(base_kw)
        if step_key is not None and step_sz is not None:
            kw[step_key] = step_sz
        # Langevin-family: η_max = δ so flat-region EM step matches trust length.
        if method in ("adam_annealed", "langevin"):
            d = float(step_sz if step_sz is not None else kw.get("delta", 1e-2))
            kw["delta"] = d
            kw["eta_max"] = d
        if margin is not None:
            m = float(margin)
        else:
            s = step_sz if step_sz is not None else (kw.get(step_key) if step_key else 0.0)
            m = float(margin_ratio) * float(s or 0.0)
        last_margin = m
        opt = run_optimize(
            graphs,
            obs,
            method=method,
            clear_kind=clear_kind,
            union_kind=union_kind,
            weights=w,
            margin=m,
            steps=n_iters,
            p_max_obs=p_max_obs,
            p_max_grp=p_max_grp,
            p_max_union=p_max_union,
            refresh_every=refresh_every,
            out_dir=out,
            record_state=record_state,
            verbose=verbose,
            recorder=rec,
            phase=phase,
            **kw,
        )
        wall_time += float(opt["wall_time"])

    assert opt is not None
    (out / "opt.json").write_text(
        json.dumps(
            {
                "method": method,
                "clear_kind": clear_kind,
                "union_kind": union_kind,
                "two_stage": two_stage,
                "coarse_steps": n_coarse,
                "refine_steps": n_refine,
                "coarse_step_size": coarse_step_size,
                "refine_step_size": refine_step_size,
                "margin_ratio": margin_ratio,
                "margin": margin,
                "last_margin": last_margin,
                "steps": steps,
                "p_max_obs": p_max_obs,
                "p_max_grp": p_max_grp,
                "p_max_union": p_max_union,
                "refresh_every": refresh_every,
                "kwargs": base_kw,
            },
            indent=2,
        )
    )
    lay, rec = opt["layout"], opt["recorder"]
    lay_path = out / "layout.json"
    lay.export(lay_path)
    store_cfg["layout"] = str(lay_path)
    store_cfg["opt"] = str(out / "opt.json")

    report = performance_report(
        graphs,
        obs,
        rec,
        clear_kind=clear_kind,
        margin=last_margin,
        wall_time=wall_time,
    )
    store_cfg["recorder"] = str(rec.save("recorder.pkl"))
    obs_path = out / "obstacles.json"
    dump_obstacles(obs, obs_path)
    store_cfg["obstacles"] = str(obs_path)
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
        "union_kind": union_kind,
        "report": report,
        "out_dir": str(out),
        "two_stage": two_stage,
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
        help="env YAML, OptimizeGraph JSON, or directory with graph_*.json",
    )
    parser.add_argument(
        "--obstacles",
        type=str,
        default=None,
        help="obstacles YAML/JSON (overrides env / sibling obstacles.json)",
    )
    parser.add_argument("-o", "--out-dir", default=str(root / "runs" / "stage1"))
    parser.add_argument("-m", "--method", default="adam", choices=list(METHODS))
    parser.add_argument(
        "--clear-kind",
        default="capsule",
        choices=["capsule", "hull", "ignore"],
        help="narrowphase proxy; 'ignore' skips all clearance (debug)",
    )
    parser.add_argument(
        "--union-kind",
        default="chamfer",
        choices=["chamfer", "ignore"],
        help="junction union; 'ignore' skips Chamfer refresh (debug)",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=1.5,
        dest="margin_ratio",
        help="clear soft pad = ratio * stage step_size; want >1 (default 1.5)",
    )
    parser.add_argument(
        "--p-max-obs",
        type=int,
        default=128,
        dest="p_max_obs",
        help="packed obstacle-clear hit budget (default 128)",
    )
    parser.add_argument(
        "--p-max-grp",
        type=int,
        default=128,
        dest="p_max_grp",
        help="packed cross-group-clear hit budget (default 128)",
    )
    parser.add_argument(
        "--p-max-union",
        type=int,
        default=64,
        dest="p_max_union",
        help="packed junction-union Chamfer sample budget (default 64)",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=10,
        dest="refresh_every",
        help="re-detect clear/union active sets every N steps (default 10)",
    )
    parser.add_argument(
        "--two-stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="coarse step then refine (default on; --no-two-stage for single pass)",
    )
    parser.add_argument(
        "--coarse-step",
        type=int,
        default=50,
        dest="coarse_steps",
        help="two-stage coarse iteration count (default 50)",
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=50,
        dest="refine_steps",
        help="two-stage refine iteration count; also used if --no-two-stage (default 50)",
    )
    parser.add_argument(
        "--coarse-step-size",
        type=float,
        default=0.1,
        dest="coarse_step_size",
        help="two-stage coarse lr/delta/trust (default 0.1)",
    )
    parser.add_argument(
        "--refine-step-size",
        type=float,
        default=0.01,
        dest="refine_step_size",
        help="two-stage refine lr/delta/trust (default 0.01)",
    )
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

    graphs, obs = resolve_case(
        args.env,
        clear_kind=args.clear_kind,
        obstacles_path=args.obstacles,
    )
    out = run_pipeline(
        graphs,
        obs,
        method=args.method,
        clear_kind=args.clear_kind,
        union_kind=args.union_kind,
        weights=weights,
        margin_ratio=args.margin_ratio,
        p_max_obs=args.p_max_obs,
        p_max_grp=args.p_max_grp,
        p_max_union=args.p_max_union,
        refresh_every=args.refresh_every,
        two_stage=args.two_stage,
        coarse_steps=args.coarse_steps,
        refine_steps=args.refine_steps,
        coarse_step_size=args.coarse_step_size,
        refine_step_size=args.refine_step_size,
        out_dir=args.out_dir,
        record_state=args.record_state,
        verbose=args.verbose,
    )
    c = out["report"]["clearance"]
    p = out["report"]["path"]["aggregate"]
    s = out["report"]["solve"]
    print(
        f"{out['method']}  clear={out['clear_kind']}  union={out['union_kind']}  "
        f"ok={c['ok']}  n_hit={c['n_hit']}  "
        f"L={p['L']:.4g}  kr_max={p['kr_max']:.4g}  "
        f"t={s['wall_time']:.3g}s  → {out['out_dir']}"
    )


if __name__ == "__main__":
    main()
