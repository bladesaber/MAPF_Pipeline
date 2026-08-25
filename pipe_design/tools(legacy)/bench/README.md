# tools/bench

Low-level **search** performance (pandas).

Aligned with `pipe_design/ROADMAP.md`: low-level planners (A\*/bi-dir/…).
Not for high-level / multi-group **design** (`DesignInstance` — future).

| Script | Role |
|--------|------|
| `compare_search.py` | Run all tasks under a batch tree → `all_reports.csv` |
| `factor_compare.py` | Factor analysis: `ols` (main effects) or `one_factor` (controlled slice) |

## Factors vs environment

| Role | CLI | Typical fields |
|------|-----|----------------|
| Environment | `--env` | `map_id`, `query_id` |
| Independent | `--indep` | `model_id`, `heap_id`, `heuristic_id` |
| Metric | `--metric` + `--metric-direction` | e.g. `total_time_ms` + `lower` |
| Presets | `--models` / `--heaps` / `--heuristics` | JSON id → name for conclusions |

Within each compare key, metric becomes a performance **ratio** (best = 1, worse > 1).

## Methods

| `--method` | What it does | When to use |
|------------|--------------|-------------|
| `ols` | Env-normalize → `log(ratio) ~ indep` via `statsmodels.OLS` (Sum contrast) → `summary()` + named ranking | Overall main-effect ranking across all indep together |
| `one_factor` | Vary **one** `--factor`; hold `--env` + other `--indep` fixed; report `gm_ratio` / `win_rate` | Fair single-factor compare (heap vs heap under same heuristic, etc.) |

**`one_factor` columns**

| Column | Meaning |
|--------|---------|
| `mean_metric` | Average raw metric on matched slices |
| `gm_ratio` | Geometric mean of per-slice ratio (best = 1); lower better |
| `win_rate` | Fraction of matched slices where this level is best |
| `rank` | By `gm_ratio` |

Omit `--factor` to run `one_factor` once for **each** `--indep` column.

## Usage

```bash
# 1) run pipe_run_low on every instance, collect reports
python3 pipe_design/tools/bench/compare_search.py \
  --batch-dir /path/to/batch --run
# → <batch-dir>/bench/all_reports.csv
# or: --out-file /path/to/all_reports.csv

# 2a) OLS main effects
python3 pipe_design/tools/bench/factor_compare.py \
  --input-file /path/to/benchmarks.csv \
  --env map_id query_id \
  --indep model_id heap_id heuristic_id \
  --metric total_time_ms \
  --metric-direction lower \
  --method ols \
  --models /path/to/models.json \
  --heaps /path/to/heaps.json \
  --heuristics /path/to/heuristics.json

# 2b) one-factor: vary heap only (fix model + heuristic + env)
python3 pipe_design/tools/bench/factor_compare.py \
  --input-file /path/to/benchmarks.csv \
  --env map_id query_id \
  --indep model_id heap_id heuristic_id \
  --metric total_time_ms \
  --metric-direction lower \
  --method one_factor \
  --factor heap_id \
  --models /path/to/models.json \
  --heaps /path/to/heaps.json \
  --heuristics /path/to/heuristics.json

# 2c) one-factor for every indep column
python3 pipe_design/tools/bench/factor_compare.py \
  --input-file /path/to/benchmarks.csv \
  --env map_id query_id \
  --indep model_id heap_id heuristic_id \
  --metric total_time_ms \
  --metric-direction lower \
  --method one_factor \
  --models /path/to/models.json \
  --heaps /path/to/heaps.json \
  --heuristics /path/to/heuristics.json
```

Depends on `pandas`, `numpy`, `statsmodels`.
