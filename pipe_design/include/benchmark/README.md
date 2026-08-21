# benchmark

Constructive **YAML** instances for multi-group pipe routing: corridors (+ witness
solution) first, then static obstacles (planned). Design detail:
[`ROADMAP.md`](ROADMAP.md).

**Build** (from `pipe_design/build`):

```bash
cmake .. && cmake --build . --target main_gen pipe_benchmark -j
```

Depends on `libyaml-cpp-dev`.

---

## Corridor generators

| Name | Code | Notes |
|------|------|--------|
| `axis_ray` | `axis_ray/` | Preferred (LOS ray tree, no A\*) |
| `plane_slice` | `plane_slice/` | Planar A\* on axis slices |

Shared types: `GeneratorTypes.h`, `YamlUtils.h`. Gaps / lengths derived from
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

Script: [`gen_robust.sh`](gen_robust.sh) — both generators, three levels, `NUM`
instances each (default 100).

| Level | groups | terminals / group | axis_ray init / attach |
|-------|--------|-------------------|-------------------------|
| easy | 2 | 3 | 5 / 4 (both > 3) |
| middle | 4 | 4 | 6 / 5 |
| hard | 6 | 5 | 8 / 6 |

Shared for all levels: `--size 48`, `path_gap=2`, `branch_gap=3` (`axis_ray`);
`radius=0.5`. Difficulty comes from groups / terminals / waypoints only.

```bash
# from pipe_design/build (or set MAIN_GEN)
chmod +x ../include/benchmark/gen_robust.sh
../include/benchmark/gen_robust.sh ./bench_out 100
```

Layout: `bench_out/<axis_ray|plane_slice>/<easy|middle|hard>/instance_*.yaml`  
plus `bench_out/robust_summary.txt` (exit codes / yaml counts).

---

## Obstacle families

*TBD* — layout family + density outside witness corridors (see ROADMAP §2).

---

## YAML schema

*Stable fields today:* `meta`, `grid`, `obstacles`, `groups`, `solution`.  
Loaders for `problem::Problem` and schema freeze: TBD.
