# tools/gen

Python: raw benchmarks → `SearchInstance` JSON for C++ runners (`pipe_run_low`, …).
Schema: `tools/formats/SearchInstance.h`. Reports stay `SearchReport` — do not invent formats here.

```text
gen/
  movingai_grids.py         # one map → one instance
  batch_movingai_grids.py   # model × heap × heuristic → calls movingai_grids
  search_presets/           # default models.json, heaps.json, heuristics.json
```

Raw maps: `tools/data/movingai/…` (not committed). Output: caller `--output-dir`.

---

## Invariant

**One map → one instance → one directory.**

```text
<instance-dir>/
  instance.json
  # C++ may write reports.jsonl, closed_set/, posthoc/ here
```

`SearchInstance.output_dir` equals that instance directory.

---

## `movingai_grids.py`

MovingAI `.map` + `.scen` → instances. Format: https://movingai.com/benchmarks/formats.html

| | |
|--|--|
| Input | `data/movingai/grids/<suite>/{map,scen}/` (`arena.map` ↔ `arena.map.scen`) |
| Output | `<output-dir>/<map>/instance.json` |
| `output_dir` field | `<output-dir>/<map>/` |

`--output-dir` is required. Search config and optional `--model-id` / `--heap-id` / `--heuristic-id` via CLI (written into `SearchInstance`).

```bash
python3 pipe_design/tools/gen/movingai_grids.py \
  --suite dao --map arena --limit 10 \
  --output-dir /path/to/runs
```

Reusable: `load_map`, `load_scen`, `build_instance`, `convert_one`.

---

## `batch_movingai_grids.py`

Cartesian product of three JSON files. Each flag is a **file path**; omit → default under `search_presets/`:

| Flag | Default |
|------|---------|
| `--models` | `search_presets/models.json` |
| `--heaps` | `search_presets/heaps.json` |
| `--heuristics` | `search_presets/heuristics.json` |

Entries need `model_id` / `heap_id` / `heuristic_id`. Layout:

```text
<output-dir>/<model_id>__<heap_id>__<heuristic_id>/<map>/instance.json
```

```bash
python3 pipe_design/tools/gen/batch_movingai_grids.py \
  --suite dao --map arena --limit 10 \
  --output-dir /path/to/batch

python3 pipe_design/tools/gen/batch_movingai_grids.py \
  --suite dao --map arena \
  --models /path/to/my_models.json \
  --heaps /path/to/my_heaps.json \
  --output-dir /path/to/batch
```

Does not run the C++ runner. For run + compare, see `tools/bench/compare_search.py`.

---

## Extending

1. New map pack → `tools/data/<source>/…`.
2. New loader → `gen/<source>_….py` → same `SearchInstance` schema; keep one map / one instance dir.
3. New batch combo → add entries (with ids) to a presets JSON, or pass a custom file via `--models` / `--heaps` / `--heuristics`.
4. New planner field → update schema + C++ load first; gen only fills JSON.
