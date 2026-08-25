# MovingAI benchmark data

Local store for [MovingAI](https://www.movingai.com/benchmarks/) / ShortestPathLab downloads.
Scripts under `tools/gen/` and `tools/bench/` should read from here — do **not** commit large archives.

## Layout

```text
movingai/
  grids/     # P0 — 2D `.map` + `.scen` (rooms, mazes, dao, …)
  voxels/    # P1 — 3D voxel maps (optional)
  README.md  # this file
```

Per-suite layout (e.g. `grids/dao/`):

```text
grids/<suite>/
  map/          # *.map
  scen/         # *.map.scen
  png/          # optional previews (not used by instance gen)
```

Generated `SearchInstance` JSON and C++ dumps are **not** under `data/` by default — they go to a caller-chosen `--output-dir` (see `tools/gen/movingai_grids.py`).

## Generate instances

Code in `tools/gen/` (not under `data/`). **One map → one instance → one output directory:**

```bash
python3 pipe_design/tools/gen/movingai_grids.py \
  --suite dao --map arena --limit 10 \
  --output-dir /path/to/runs

# → /path/to/runs/arena/instance.json
#    SearchInstance.output_dir = /path/to/runs/arena

python3 pipe_design/tools/gen/movingai_grids.py --suite dao \
  --model astar --model-kwargs '{"heuristic_weight": 1.0}' \
  --heuristic octile --heap LazyBinaryHeap \
  --output-dir /path/to/runs
```

Search config (`model`, `heuristic`, `heap`, kwargs) comes from CLI. `--output-dir` is required; each map stem is a subdirectory of that root.

Format reference: https://movingai.com/benchmarks/formats.html

## Download (grids — P0)

- Hub: https://www.movingai.com/benchmarks/grids.html  
- Format: https://movingai.com/benchmarks/formats.html  
- Mirror: https://bitbucket.org/shortestpathlab/benchmarks (`grid-maps/`)

Unpack so each suite sits under `grids/<suite>/` with MovingAI’s `map/` + `scen/` (+ optional `png/`) split, e.g. current `dao`:

```text
grids/dao/map/arena.map
grids/dao/scen/arena.map.scen
grids/dao/png/…          # optional previews
```

Loaders pair `map/<name>.map` with `scen/<name>.map.scen`.

## Download (voxels — P1)

- https://benchmarks.pathfinding.ai/3d-benchmarks/voxel-maps/  
- Mirror: `voxel-maps/` on the Bitbucket repo above  

Place under `voxels/<suite>/`.

## Git

`.map`, `.scen`, and archives are gitignored. Keep only this README (and empty dirs via `.gitkeep`) in version control. Do not commit generated instance/report trees.
