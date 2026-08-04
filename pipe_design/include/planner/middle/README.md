# planner/middle — Speed / Quality Wrappers

Optional wrappers around low-level search (`ROADMAP` §4.3).  
Same `step` / `path` / `search` contract as `LowLevelPlanner` via `MiddleLevelPlanner`.

## Pieces

| File | Role |
|------|------|
| `MiddleLevelPlanner.h` | Base: inherits `LowLevelPlanner` (no extra API) |
| `BidirectionalAStar.h` / `.cpp` | Bi-dir A\* over `AStar<SearchNode>`; weight via `set_heuristic_weight` |

Focal A\* lives in `planner/low/FocalAStar` (search-loop variant, not a wrapper).

## Ownership

Same as low-level: caller owns grid / heuristic / open sets / actions.  
`BidirectionalAStar` needs **two** `OpenSet`s (forward + backward).

## Usage sketch

```cpp
LazyBinaryHeap open_fwd, open_bwd;
Manhattan h(1.0);
SixConnected actions(1.0);

BidirectionalAStar bi;
bi.set_heuristic_weight(1.0);  // optional; forwarded to both A*
bi.reset(grid, start, goal, radius, h, open_fwd, open_bwd, actions);
bi.search();
Path p = bi.path();
```

Weighted A\* (`f = g + w·h`) is on `AStar::set_heuristic_weight`, not a middle class.
