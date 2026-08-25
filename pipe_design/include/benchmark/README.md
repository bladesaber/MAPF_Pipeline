# benchmark

Constructive **YAML** instances for multi-group pipe routing: corridors (+ witness
solution) first, then static obstacles (planned). Design detail:
[`ROADMAP.md`](ROADMAP.md).

**Build** (from `pipe_design/build`):

```bash
cmake .. && cmake --build . --target main_gen pipe_run_design pipe_benchmark -j
```

Depends on `libyaml-cpp-dev`.

---

## Corridor generators

| Name | Code | Notes |
|------|------|--------|
| `axis_ray` | `axis_ray/` | Preferred (LOS ray tree, no A\*) |
| `plane_slice` | `plane_slice/` | Planar A\* on axis slices |

Shared types: `GeneratorTypes.h` (`Instance::to_file` / `from_file`). Gaps / lengths derived from
`radius` where noted in each method’s `Types.h`.

---

## `main_gen`

Batch-generate YAML instances.

```bash
./main_gen --generator <axis_ray|plane_slice> --out <dir> [options]
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--generator` | `axis_ray` \| `plane_slice` | required |
| `--out` | output directory (created if needed) | required |
| `--num` | number of instances | `1` |
| `--seed` | base seed; instance `i` uses `seed+i` | `0` |
| `--size` | cubic `size_x=y=z` | `32` |
| `--size_x/y/z` | override one axis | — |
| `--radius` | uniform pipe radius | `0.5` |
| `--shell_margin` | extra shell beyond radius; `shell_cells=ceil((shell_margin+radius)/res)`; plan on core `size-2m`, then extend terminals to full face | `0` |
| `--terminal_gap` | slack beyond diameter (cells) | `2` |
| `--num_groups` | group count | `1` |
| `--terminals` | counts per group: `3` or `2,2,3` | `2` each |
| `--max_iterations` | outer round-robin cap | `256` |
| `--keep_failed` | write YAML even if generate fails | off |
| `--subdomain_n` | `plane_slice` only (`3\|4\|5`) | `3` |

**`axis_ray`:** `--init_waypoints_num`, `--attach_waypoints_num`,
`--min_path_size`, `--max_path_size`, `--path_gap`, `--branch_gap`,
`--max_attempts`.

**`plane_slice`:** `--plane_gap`.

Examples (from `pipe_design/build`, writes under `./bench_out`):

```bash
# axis_ray — preferred LOS ray tree
./main_gen --generator axis_ray --out ./bench_out --num 10 --seed 0 \
  --size 48 --radius 0.5 --num_groups 4 --terminals 3,3,3,3 \
  --init_waypoints_num 3 --attach_waypoints_num 2 \
  --min_path_size 2 --max_path_size 10 \
  --path_gap 1 --branch_gap 2 --max_attempts 64

# plane_slice — planar A* on axis slices
./main_gen --generator plane_slice --out ./bench_out --num 10 --seed 0 \
  --size 48 --radius 0.5 --num_groups 4 --terminals 3,3,3,3 \
  --subdomain_n 4 --plane_gap 2
```

Writes `instance_0.yaml`, `instance_1.yaml`, … Each file has `meta`, `grid`,
`obstacles` (empty until §obstacle paint), `groups`, `solution` (witness paths).

Exit: `0` all ok, `2` if any generate failed (unless only `--keep_failed` writes).

### Robust batch (easy / middle / hard × 100)

Script: [`gen_robust.sh`](gen_robust.sh) — `{easy,middle,hard}` × `NUM` instances
(default 100). Third argument selects the generator.

| Level | `axis_ray` groups × terms | init / attach | `plane_slice` groups × terms |
|-------|---------------------------|---------------|------------------------------|
| easy | 2 × 3 | 4 / 3 | 3 × 3 |
| middle | 4 × 4 | 5 / 4 | 6 × 4 |
| hard | 6 × 5 | 5 / 4 | 9 × 6 |

```bash
# from pipe_design/build (or set MAIN_GEN)
chmod +x ../include/benchmark/gen_robust.sh
../include/benchmark/gen_robust.sh ./bench_out 100              # both
../include/benchmark/gen_robust.sh ./bench_out 100 axis_ray    # one generator
../include/benchmark/gen_robust.sh ./bench_out 100 plane_slice

# optional 4th arg: shell_margin (else main_gen default 0)
../include/benchmark/gen_robust.sh ./bench_out 100 axis_ray 1.0
```

`GENERATOR`: `axis_ray` | `plane_slice` | `all` (default).
`SHELL_MARGIN`: optional 4th positional argument.

Layout: `bench_out/<axis_ray|plane_slice>/<easy|middle|hard>/instance_*.yaml`  
plus `bench_out/robust_summary.txt` (exit codes / yaml counts).

---

## Obstacle families

*TBD* — layout family + density outside witness corridors (see ROADMAP §2).

---

## YAML schema

*Stable fields today:* `meta`, `grid`, `obstacles`, `groups`, `solution`.  
Loaders for `problem::Problem` and schema freeze: TBD.

---

## `DesignReport`

High-level solve result JSON (success-rate under time / iteration budget).  
Code: [`design_report/DesignReport.h`](design_report/DesignReport.h).

Provenance: `instance_id` + `instance_path` only. Per-group `tasks[]` store final
paths (`cost` + `cells`). Batch: `write_design_reports_jsonl`.

---

## `pipe_run_design`

CLI runner: YAML instance(s) → CBS / PP / PBS → `DesignReport`.

```bash
# single instance
./pipe_run_design --instance ./bench_out/axis_ray/easy/instance_0.yaml \
  --export ./runs/cbs/instance_0.jsonl --algorithm cbs \
  --heuristic octile --action_model TwentySixConnected \
  --max_time_ms 60000 --max_iterations 10000

# directory batch (instance_*.yaml)
./pipe_run_design --instance_dir ./bench_out/axis_ray/easy \
  --export ./runs/cbs/axis_ray/easy.jsonl --algorithm cbs \
  --heuristic octile --action_model TwentySixConnected \
  --max_time_ms 60000 --max_iterations 10000 \
  --meta '{"generator":"axis_ray","level":"easy"}'
```

`--heuristic`: `manhattan` | `euclidean` | `octile` (default `octile`)  
`--action_model`: `SixConnected` | `EightConnected` | `TwentySixConnected` (default `TwentySixConnected`)

Writes `--export` JSONL file (one line per instance). Parent dirs are created as needed.

Batch over robust suite: [`run_bench.sh`](run_bench.sh) — plain `pipe_run_design` commands; edit the file to add runs.

```bash
# from pipe_design/build
../include/benchmark/run_bench.sh

# summarize
python ../include/benchmark/summarize_reports.py \
  /home/admin123456/Desktop/work/mapf_3d_benchmark/runs_results
```
