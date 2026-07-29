# grid — Occupancy & Clearance

Domain occupancy for planners (`ROADMAP` §3–4.1). Shapes are spheres/capsules from `geom/`.
**Collision everywhere:** blocked iff `clearance < pipe_radius`.

This folder does **not** own paths, groups, or CBS branching policy — only spatial queries.

## Mental model

```text
StaticClearanceGrid     ObstacleChain
 (dense, bake once)      (sparse, COW per CBS node)
         \                 /
          \               /
           CompositeGrid
        is_free / clearance
               │
     A* / heuristics / CBS [R]
```

| Piece | Role | Lifetime |
|-------|------|----------|
| `StaticClearanceGrid` | Truncated distance field for **static** obstacles | Shared read-only across the CBS tree |
| `ObstacleChain` | Dynamic / constraint obstacles (`parent` + local delta) | One node per CBS branch; ancestors immutable |
| `CompositeGrid` | `min(static, chain)` occupancy facade | Cheap view; swap `chain` per node |

**Do not** deep-copy the dense static field per CBS child — only `extend()` the chain.

## Types

`CellCoord` — integer grid index `(x,y,z)`. World center:

`((x+0.5)*res, (y+0.5)*res, (z+0.5)*res)`.

`flat_index` (row-major): `x + y*size_x + z*size_x*size_y`. Prefer this over re-deriving indices in planners/heuristics.

## StaticClearanceGrid

- Each cell stores clearance to nearest **static** obstacle surface, clamped to `[0, clearance_cap]`.
- Far cells stay at `clearance_cap` → `bake` only stamps a local neighborhood (`r + clearance_cap`).
- `clearance_cap` should be `≥ max_obstacle_radius + max_pipe_radius`.
- Hot path: `is_free(cell, pipe_radius)` ≡ `clearance(cell) >= pipe_radius`.
- OOB clearance is `0` (treated as blocked).

```cpp
StaticClearanceGrid g(nx, ny, nz, resolution, clearance_cap);
g.bake(spheres, capsules);
```

## ObstacleChain

Immutable linked list for CBS constraints / routed pipes as occupancy:

```text
root = ObstacleChain::make_root()
child = root->extend(sphere_or_capsule)   // shares parent; parent never mutated
```

- Effective set = locals ∪ ancestors.
- `collides(point, r)` early-exits; prefer over full `clearance` when you only need a bool.
- Analytic queries in world space (`Vec3`), not cell indices.

## CompositeGrid

Non-owning view: holds `const StaticClearanceGrid*` + `shared_ptr<const ObstacleChain>`.

```cpp
CompositeGrid world(static_grid, chain);  // chain may be null / empty root
world.is_free(cell, pipe_radius);         // static first, then chain (short-circuit)
world.set_chain(cbs_node_chain);          // switch CBS node without rebaking static
```

`clearance(cell) = min(static.clearance(cell), chain.clearance(cell_center))`.

## Hot-path rules

1. Prefer `is_free` / `collides` over full `clearance` unless the value is needed (e.g. adaptive step).
2. Bake static once; CBS only extends `ObstacleChain`.
3. Heuristics that ignore the chain (e.g. `Dijkstra` true-distance) stay admissible under extra dynamic obstacles — search must still use `CompositeGrid::is_free` with the real `pipe_radius`.

## Files

```text
Types.h                 CellCoord
StaticClearanceGrid.h   dense static DF
ObstacleChain.h         COW constraint list
CompositeGrid.h         combined query
src/grid/*.cpp
tests/test_grid.cpp
```

## Extending

| Next | Notes |
|------|-------|
| KD-tree / spatial index on chain | When dynamic obstacle count is large (P2) |
| Multi-resolution / corridors | Optional acceleration (P3–P4) |
| Same-group routed pipes as capsules | Usually `chain.extend(capsule)` after a successful route |

Keep geom math in `geom/`; keep search in `planner/`. New occupancy backends should still expose `is_free` / `clearance` so A* and CBS stay swappable.
