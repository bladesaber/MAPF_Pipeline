# pipe_smooth

Stage-1 centerline refine after discrete routing (`pipe_design`). Hermite graphs, soft/hard clearance, junction union; metrics in `report.json`.

Method details: `ROADMAP.md`.

---

## Generate instances — `gen_robust.sh`

Needs `pipe_design`’s `main_gen` (or `MAIN_GEN=/path/to/main_gen`).

```bash
cd pipe_smooth
./gen_robust.sh [OUT_DIR] [NUM] [RADIUS] [SHELL_MARGIN]
# default → data/bench/{easy,hard}/instance_*.yaml  (50 each, r=1.5)
```

| level | layout |
|-------|--------|
| easy  | 3 groups × 3 terminals |
| hard  | 8 groups × 8 terminals |

Generator: `plane_slice` + `pillar_field`. Optional env: `SIZE`, `CORRIDOR_MARGIN`, `MAX_FILL_ROUNDS`.  
Prepared sets used for experiments: `data/easy`, `data/hard`.

---

## Batch optimize — `run_instances.sh`

Parallel Stage-1 over `instance_*.yaml` in **folders × methods × clear-kinds**.  
Writes `runs/<folder>/<method>/<clear_kind>/instance_*/`.

```bash
cd pipe_smooth
./run_instances.sh data/hard -j 8 -m all --clear-kind all \
  --two-stage --coarse-step 500 --refine-step 200 \
  --coarse-step-size 0.1 --refine-step-size 0.01 \
  --union-kind chamfer --margin-ratio 1.5 \
  --p-max-obs 1024 --p-max-grp 1024 --p-max-union 512
```

`interior_nlp` has an inner IPOPT loop (`max_iter`, default 20). Bash rescales outer
iters to `ceil(N / IPOPT_MAX_ITER)` so total work ≈ adam-style budget
(e.g. coarse 500 → 25 outer × 20 inner). Override with env `IPOPT_MAX_ITER`.

| flag | role |
|------|------|
| `DATA_DIR …` | one or more instance folders (cwd or under `pipe_smooth/`) |
| `-j` | parallel workers |
| `-m` | method, comma-list, or `all` (`adam` / `adam_annealed` / `langevin` / `elastic_qp` / `interior_nlp`) |
| `--clear-kind` | `capsule` / `hull` / `ignore`, comma-list, or `all` |
| `--coarse-step` / `--refine-step` | iters per stage (`--refine-step` also used if `--no-two-stage`); for `interior_nlp` these are adam-equivalent budgets (rescaled, see above) |
| `--coarse-step-size` / `--refine-step-size` | stage step size (see table below) |
| `--margin-ratio` | clear soft pad = ratio × stage step size (want > 1) |
| `--union-kind` | `chamfer` / `ignore` (skip junction Chamfer refresh) |
| `--p-max-obs/grp/union` | packed hit budgets |
| `--refresh-every` | re-detect clear/union active sets every N steps |
| `--two-stage` / `--no-two-stage` | |

**Stage step size → solver kw**

| method | `*_step_size` sets |
|--------|--------------------|
| `adam` | `lr` |
| `adam_annealed` / `langevin` | `delta` and `eta_max` (kept equal; Langevin EM rate = trust length) |
| `elastic_qp` | `trust` |
| `interior_nlp` | `delta` (box `x±delta` each outer solve) |

Single case (same flags):

```bash
python -m pipe_smooth.pipeline data/easy/instance_0.yaml -o runs/stage1 \
  -m adam --two-stage --coarse-step 500 --refine-step 200 \
  --coarse-step-size 0.1 --refine-step-size 0.01 \
  --clear-kind capsule --union-kind chamfer --margin-ratio 1.5 \
  --p-max-obs 1024 --p-max-grp 1024 --p-max-union 512
```

Also accepts `graph_*.json` / case dirs + `--obstacles`.

**Run folder contents:** `opt.json` (method, clear/union, step sizes, `p_max_*`, …), `weights.json`, `layout.json`, `obstacles.json`, `graph_*.json`, `report.json`, `recorder.pkl`, `store_cfg.json`.  
`report.json` → `solve.J0` / `JT` use the full two-stage recorder history (coarse + refine).

---

## Viewers

```bash
# Optimize bench (synthetic cases, weights UI, Start/Replay)
python -m pipe_smooth.vis_bench.app [-o SAVE_DIR]

# Env YAML → trees / junctions / Hermite (+ obstacles, Clip bar)
python -m pipe_smooth.vis_instance_topology [ENV.yaml] [-r 0.1]

# Exported JSON dumps (tree_*/partitioned_*/graph_*); File → Load obstacles
python -m pipe_smooth.vis_topology [file_or_dir ...] [-r 0.1]
```

| tool | input | use |
|------|--------|-----|
| `vis_bench` | built-in examples / saved cases under `runs/` | tune weights, run one optimize |
| `vis_instance_topology` | prepared env YAML | inspect tree → junction → Hermite; View → Obstacles; Clip |
| `vis_topology` | `tree_*.json` / `partitioned_*.json` / `graph_*.json` | inspect dumps from a run folder |
