# planner/low — Low-Level Grid Search

Single-agent path search on `CompositeGrid` (`ROADMAP` low-level).  
Owns **node storage** and the search loop; does **not** own grid / heuristic / open set / action model.

## Pieces

| File | Role |
|------|------|
| `LowLevelPlanner.h` | Base: `SearchStatus`, `step` / `path` / `search`, `status_` / `expansions_` |
| `SearchNode.h` | `SearchNode` (cell), `DirectedSearchNode` (cell + heading), state keys + hash |
| `AStar.h` / `.cpp` | `AStar<Node>` — steppable A\* (`reset` / `step` / `search`) |
| `GreedyBestFirst.h` | A\* variant: `f = h` (ignore g for ranking) |
| `Dijkstra.h` | A\* variant: `f = g` (uniform-cost; ≠ `heuristic::Dijkstra`) |
| `ARAStar.h` / `.cpp` | Anytime Repairing A\* — inflate ε, then repair via INCONS |
| `ANAStar.h` / `.cpp` | Anytime Nonparametric A\* — key `(G−g)/h`, no ε schedule |
| `FocalAStar.h` / `.cpp` | Focal A\* (`A*_ε`); OPEN by `f` + owned FOCAL (`PairingHeap` / `MinHgCompare`) |
| `AdaptiveStepAStar.h` / `.cpp` | Clearance-adaptive shell A\*; covered ball anti jump-back |
| `LineOfSight.h` | Cell-center LOS + Euclidean edge cost (Theta\*) |
| `ThetaStar.h` / `.cpp` | Basic Theta\* — parent LOS on expand |
| `LazyThetaStar.h` / `.cpp` | Lazy Theta\* — deferred LOS in `before_expand` |
| `DirectedAStar.h` / `.cpp` | `AStar<DirectedSearchNode>` + heading-aware `expand` |
| `LPAStar.h` / `.cpp` | Lifelong Planning A\* — `update_vertex` / `step` / `update_obstacles` |
| `Path.h` | Discrete cell path + `cost` |
| `REPLAN_ROADMAP.md` | Path replan methods (re-search, stitching, …) |

Related (not in this folder): `planner/open`, `planner/motion`, `heuristic/`.

## Ownership

```text
Caller owns:  CompositeGrid, Heuristic, OpenSet, ActionModel
AStar owns:   unordered_map<Node::Key, unique_ptr<Node>>
OpenSet holds: non-owning SearchNode*  (DirectedSearchNode* is-a SearchNode*)
```

Pass references into `reset(...)`. Reuse the same `OpenSet` across runs; `reset` clears it.

## State identity (`Node::Key`)

| Node | Key | When to use |
|------|-----|-------------|
| `SearchNode` | `CellStateKey` (cell) | Shortest path, no turn model |
| `DirectedSearchNode` | `DirectedStateKey` (cell + heading bin) | Turn cost / heading continuity |

Contract for any `Node` used with `AStar<Node>`:

- derive from `SearchNode` (for `OpenSet`)
- `using Key = ...` with `std::hash<Key>`
- `static Key make_key(const CellCoord&)` — start states
- `static Node from_key(const Key&)`
- `key() const`

`DirectedSearchNode` also has `make_key(cell, heading, yaw_bins, pitch_bins)` for successors.

## Search loop

```text
reset(grid, starts, goals, r, h, open, actions)
    → seed open with start nodes (g=0), or Success if start ∈ goals
step()   one expansion: pop → closed → goal? → expand
search() step until Success / Failure / max_expansions
path()   parent chain from goal_node_ (only if Success)
```

`expand` (virtual):

1. `actions_->successors(from, in_heading, buf)` — connectivity + turn soft/hard cost live in motion
2. skip out-of-bounds / blocked (`is_free`)
3. `ensure_node(key)` → relax g / parent / f → `push` or `decrease_key` (+ re-`push` if false)

Default `AStar::expand`: `in_heading = none` → no turn filter/penalty.  
`DirectedAStar::expand`: `in_heading = current->in_dir`, successor key includes heading bin; updates `set_arrival` / `turns_streak`.

Priority: `f = combine_key(g, h)` default `g + w·h` (`set_heuristic_weight`).  
`h` = min heuristic over all goals.

| Planner | `combine_key` |
|---------|---------------|
| `AStar` | `g + w·h` |
| `GreedyBestFirst` | `h` |
| `Dijkstra` | `g` |
| `ARAStar` | `g + ε·h` (ε decreases over repair rounds) |
| `ANAStar` | `−(G−g)/h` after first incumbent; else `h` |
| `FocalAStar` | OPEN `g+h`; pick from FOCAL `f ≤ w·f_min` by `h` |
| `ThetaStar` / `LazyThetaStar` | same as A\* (`g + w·h`); differ in `expand` / LOS |

## Usage sketch

```cpp
LazyBinaryHeap open;
Manhattan h(1.0);
SixConnected actions(1.0);
// actions.set_turn_weight(1.0);      // soft
// actions.set_max_turn_deg(90.0);    // hard (needs in_heading → DirectedAStar)

AStar<SearchNode> plain;
plain.reset(grid, start, goal, radius, h, open, actions);
plain.search();
Path p = plain.path();

GreedyBestFirst greedy;
greedy.reset(grid, start, goal, radius, h, open, actions);
greedy.search();

Dijkstra ucs;  // planner::Dijkstra — not heuristic::Dijkstra
ucs.reset(grid, start, goal, radius, h, open, actions);
ucs.search();

ARAStar ara;
ara.set_eps_schedule(/*start=*/5.0, /*target=*/1.0, /*step=*/0.5);
ara.reset(grid, start, goal, radius, h, open, actions);
ara.search(100);           // may still be Running with an incumbent
Path early = ara.path();   // OK while Running
ara.search();              // run to ε = 1

ANAStar ana;
ana.reset(grid, start, goal, radius, h, open, actions);
ana.search(50);            // incumbent via path()
ana.search();              // optimal when Success

LazyBinaryHeap open;
FocalAStar focal;
focal.set_focal_weight(2.0);
focal.reset(grid, start, goal, radius, h, open, actions);
focal.search();

Euclidean eu(1.0);  // preferred for Theta*
ThetaStar theta;
theta.set_step_cost(1.0);
theta.reset(grid, start, goal, radius, eu, open, actions);
theta.search();  // path cells are any-angle waypoints

LazyThetaStar lazy_theta;
lazy_theta.set_step_cost(1.0);
lazy_theta.reset(grid, start, goal, radius, eu, open, actions);
lazy_theta.search();

DirectedAStar directed;
directed.set_heading_bins(8, 1);
directed.reset(grid, start, goal, radius, h, open, actions);
directed.search();
```

## Extending

| Goal | Where |
|------|--------|
| New connectivity / turn rules | `planner/motion` (`ActionModel`), not here |
| New open-set | `planner/open` (`OpenSet&` in `reset`) |
| New heuristic | `heuristic::Heuristic` |
| Heading-aware / Hybrid-style expand | subclass `AStar<YourNode>`, override `expand` (see `DirectedAStar`) |
| Different f | override `combine_key` (see `GreedyBestFirst`, `Dijkstra`, `ARAStar`, `ANAStar`) |
| Anytime / repair | `ARAStar` / `ANAStar` (own `step` / `expand`; `path()` while Running) |
| Focal / FOCAL list | `FocalAStar` (`set_focal_weight`; FOCAL = `PairingHeapBase` + `MinHgCompare`) |
| Theta* / any-angle | `ThetaStar` / `LazyThetaStar` (+ `LineOfSight`) |

Do **not** put CBS, group scheduling, or topology in this folder.

## Status

`Ready` → `Running` → `Success` | `Failure`.  
Invalid / blocked starts or goals → `Failure` at `reset`. Empty open after expansions → `Failure`.
