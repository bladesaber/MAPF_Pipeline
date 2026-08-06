# heuristic — Remaining-cost estimates for search

Strategy objects used by low-level planners (`ROADMAP` §4.3).
Independent of open sets and CBS; A* / Theta* / Hybrid-A* take a `const Heuristic&`.

## Interface

```cpp
class Heuristic {
  virtual double estimate(from, to) const = 0;
};
```

Scale must match edge costs (usually grid `resolution`).

A* stores a **non-owning** pointer — keep the heuristic alive for the search.

## Implementations

| Class | Formula / method | Match connectivity |
|-------|------------------|-------------------|
| `Manhattan` | `step_cost * (|dx|+|dy|+|dz|)` | 6-connect |
| `Euclidean` | `step_cost * ‖Δ‖₂` | any (weaker for grids) |
| `Octile` | 3D octile (`√3`/`√2`/1 mix) | 26-connect; 2D-octile when `dz=0` |
| `Dijkstra` | multi-source true distance on static grid | 6-connect; `prepare(goals)` |

Planned: Chebyshev, Dubins, Landmark/ALT.

## Usage

```cpp
heuristic::Manhattan h(grid.static_grid().resolution());
astar.reset(grid, start, goal, radius, h);

// True-distance (static obstacles; relaxed bake occupancy):
heuristic::Dijkstra td(grid.static_grid());
td.prepare(goal);                    // single source
td.prepare({a, b, c});               // attach-set / multi-goal
// td.clear();                       // drop field when sources change
astar.reset(grid, start, goal, radius, td);  // search still uses real radius
```

`Dijkstra::estimate` returns distance to the **nearest prepared source** (`to` ignored).
Bake uses relaxed free checks (`clearance > 0`, or optional `bake_radius`), not the Task pipe radius — admissible for any thicker pipe; search must still call `is_free(..., pipe_radius)`.

Include `heuristic/Geometric.h` for the common set, or individual headers.

## Extending

1. Subclass `Heuristic`; keep `estimate` admissible w.r.t. the planner’s edge costs.
2. Stateful heuristics: `prepare(...)` before `AStar::reset`.
3. Do **not** put turn/clearance **edge** penalties here; those belong in the cost model.
4. `Dijkstra` ignores `ObstacleChain` on purpose — still admissible when CBS adds dynamic constraints.
