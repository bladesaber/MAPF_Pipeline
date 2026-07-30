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

`.map`, `.scen`, and archives are gitignored. Keep only this README (and empty dirs via `.gitkeep`) in version control.
