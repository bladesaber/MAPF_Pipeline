#!/usr/bin/env python3
"""Env-normalized multi-factor analysis (ols) and controlled one-factor compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices  # type: ignore


def load_id_names(models: Path, heaps: Path, heuristics: Path) -> dict[str, str]:
    """Map model_id/heap_id/heuristic_id -> display name from preset JSON."""
    out: dict[str, str] = {}
    for path, id_key, name_key in (
        (models, "model_id", "model"),
        (heaps, "heap_id", "heap"),
        (heuristics, "heuristic_id", "heuristic"),
    ):
        data = json.loads(path.read_text())
        for entry in data.values():
            out[str(entry[id_key])] = str(entry[name_key])
    return out


def env_ratio(g: pd.DataFrame, keys: list[str], metric: str, direction: str) -> pd.Series:
    if direction == "lower":
        best = g.groupby(keys, dropna=False)[metric].transform("min")
        return g[metric] / best
    best = g.groupby(keys, dropna=False)[metric].transform("max")
    return best / g[metric]


def ols(df: pd.DataFrame, env: list[str], indep: list[str], metric: str, direction: str):
    """Fit OLS on log(env-normalized ratio). Returns (results, aggregated frame)."""
    g = df.groupby(env + indep, dropna=False)[metric].mean().reset_index()
    for c in indep:
        g[c] = g[c].astype(str)
    g["ratio"] = env_ratio(g, env, metric, direction)
    g["y"] = np.log(g["ratio"].astype(float))

    terms = [f"C({f}, Sum)" for f in indep if g[f].nunique() > 1]
    formula = "y ~ 1" if not terms else "y ~ " + " + ".join(terms)
    y, x = dmatrices(formula, data=g, return_type="dataframe")
    return sm.OLS(y, x).fit(), g


def conclude_ols(res, g: pd.DataFrame, indep: list[str], names: dict[str, str]) -> None:
    print("\n=== OLS conclusion (lower weight = better on log-ratio) ===")
    print(f"R^2 = {res.rsquared:.4f}  (additive main effects only; interactions ignored)")
    for f in indep:
        lev = sorted(g[f].unique())
        if len(lev) == 1:
            print(f"\n{f}: only one level → {names.get(lev[0], lev[0])} ({lev[0]})")
            continue
        w = {lv: float(res.params[f"C({f}, Sum)[S.{lv}]"]) for lv in lev[:-1]}
        w[lev[-1]] = -sum(w.values())
        ranked = sorted(w.items(), key=lambda kv: kv[1])
        print(f"\n{f} best → worst:")
        for i, (lv, coef) in enumerate(ranked, 1):
            key = f"C({f}, Sum)[S.{lv}]"
            p = float(res.pvalues[key]) if key in res.pvalues else float("nan")
            p_s = f"p={p:.3g}" if p == p else "p=n/a (omitted Sum level)"
            print(f"  {i}. {names.get(lv, lv):20s}  {lv:14s}  weight={coef:+.4f}  {p_s}")
        print(
            f"  → prefer {names.get(ranked[0][0], ranked[0][0])}; "
            f"avoid {names.get(ranked[-1][0], ranked[-1][0])}"
        )


def one_factor(
    df: pd.DataFrame,
    env: list[str],
    indep: list[str],
    factor: str,
    metric: str,
    direction: str,
    names: dict[str, str],
) -> None:
    """Vary one indep factor; hold env + other indep fixed. Print ranking conclusion."""
    if factor not in indep:
        raise SystemExit(f"--factor {factor!r} must be one of --indep {indep}")
    others = [c for c in indep if c != factor]
    keys = env + others
    g = df.groupby(keys + [factor], dropna=False)[metric].mean().reset_index()
    g[factor] = g[factor].astype(str)
    g["ratio"] = env_ratio(g, keys, metric, direction)

    # matched slices: all levels present
    n_levels = g[factor].nunique()
    counts = g.groupby(keys)[factor].transform("count")
    matched = g[counts == n_levels].copy()
    if matched.empty:
        raise SystemExit(f"one_factor[{factor}]: no fully matched slices")

    rows = []
    for lv, sub in matched.groupby(factor):
        rows.append(
            {
                "level": lv,
                "name": names.get(lv, lv),  # type: ignore
                "n": len(sub),
                "mean_metric": float(sub[metric].mean()),
                "gm_ratio": float(np.exp(np.log(sub["ratio"].astype(float)).mean())),
                "win_rate": float((sub["ratio"] <= 1.0 + 1e-15).mean()),
            }
        )
    tab = pd.DataFrame(rows).sort_values("gm_ratio")
    tab["rank"] = range(1, len(tab) + 1)

    print(f"\n=== one_factor: vary {factor} (hold {keys}) ===")
    print(f"matched slices: {matched.groupby(keys).ngroups}  levels: {n_levels}")
    print(tab.to_string(index=False))
    print(
        f"→ prefer {tab.iloc[0]['name']}; avoid {tab.iloc[-1]['name']}  "
        f"(by gm_ratio; win_rate = fraction of slices where level is best)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--env", nargs="+", required=True)
    ap.add_argument("--indep", nargs="+", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--metric-direction", choices=("lower", "higher"), required=True)
    ap.add_argument("--method", choices=("ols", "one_factor"), default="ols")
    ap.add_argument(
        "--factor",
        default=None,
        help="for one_factor: which --indep column to vary (default: each indep once)",
    )
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--heaps", type=Path, required=True)
    ap.add_argument("--heuristics", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_file)
    need = args.env + args.indep + [args.metric]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"missing columns: {miss}")

    names = load_id_names(args.models, args.heaps, args.heuristics)

    if args.method == "ols":
        res, g = ols(df, args.env, args.indep, args.metric, args.metric_direction)
        print(res.summary())
        conclude_ols(res, g, args.indep, names)
    elif args.method == "one_factor":
        factors = [args.factor] if args.factor else list(args.indep)
        for f in factors:
            one_factor(
                df, args.env, args.indep, f, args.metric, args.metric_direction, names
            )
    else:
        raise SystemExit(f"unknown method {args.method}")


if __name__ == "__main__":
    main()
