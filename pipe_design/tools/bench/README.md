# tools/bench

Low / middle-level **search** performance (pandas).

Aligned with `pipe_design/ROADMAP.md`: low-level (A\*/…) + middle-level wrappers.
Not for high-level / multi-group **design** (`DesignInstance` — future).

| Script | Role |
|--------|------|
| `compare_search.py` | Run all tasks under a batch tree → `all_reports.csv` |
| `factor_compare.py` | One-factor comparison table from that CSV |

## Factors vs environment

| Role | Fields |
|------|--------|
| Factors | `model_id`, `heap_id`, `heuristic_id` |
| Environment | `map_id`, `query_id` |

**Fair comparison:** vary **one** factor; keep environment + the other two factors the same.

## Usage

```bash
# 1) run pipe_run_low on every instance, collect reports
python3 pipe_design/tools/bench/compare_search.py \
  --batch-dir /path/to/batch --run
# → <batch-dir>/bench/all_reports.csv
# or: --out-file /path/to/all_reports.csv

# 2) compare one factor
python3 pipe_design/tools/bench/factor_compare.py \
  --csv /path/to/batch/bench/all_reports.csv \
  --factor heap_id \
  --metric total_time_ms
# → compare_heap_id_total_time_ms.csv
```

Depends on `pandas`.
