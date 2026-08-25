# Known issues / future fixes

1. **Dijkstra heuristic + high-level planners**  
   `heuristic::Dijkstra` needs `prepare(goals)` before each low-level search (goals change per Task). Recomputing after every Task (or after every `ObstacleChain` change) is expensive. Today multi/high never call `prepare`, so Dijkstra h is effectively unused (`estimate` → 0). Decide later whether to wire per-Task `prepare` on static grid only, or leave Dijkstra out of the high-level experiment path.

2. **High-level planners are A\*-centric**  
   CBS / PP / PBS `reset` take `AStar<SearchNode>&` (and related open/heuristic/action APIs). Other low-level models (Focal, Theta*, ARA*/ANA*, AdaptiveStep, …) are not pluggable yet. Need a more universal low-level interface later so `pipe_run_design` `--low_model` can select beyond A*.

3. **Any-angle low-level paths vs grid occupancy consumers**  
   Any-angle / long-step planners in `planner/low/` do **not** return a dense 6/8/26 cell chain:

   | Planner | Path shape today |
   |---------|------------------|
   | `ThetaStar` / `LazyThetaStar` | Parent LOS shortcuts → sparse **waypoints** (may skip cells) |
   | `AdaptiveStepAStar` | Clearance-adaptive Chebyshev shell jumps → large steps; covered-ball anti jump-back |

   Downstream (ObstacleChain witness sampling, CBS constrain geometry, clearance checks, viz as tubes) often assumes a **grid-based** polyline with consecutive (or near-consecutive) cells. Sparse any-angle paths leave gaps that are freeness-valid in continuous LOS but invisible to cell-centric consumers.

   **Planned option (future):** optional **path reconstruction / densify** after any-angle search succeeds — fill each waypoint→waypoint (or jump) segment with a grid path, so callers can choose:

   - keep any-angle waypoints (cost / length reporting, continuous clearance), or  
   - export a grid-based path for high-level / sampling / collision bookkeeping.

   **Working idea:** use **greedy best-first** (`GreedyBestFirst`) as a **sub-area** densifier between consecutive waypoints (local start/goal = segment ends; restrict search to a corridor / AABB around the LOS segment). Cheap and goal-directed for short local fills.

   **Open questions:**

   - Greedy can fail locally even when a grid path exists → fallback to A\* (or Bresenham / 3D line rasterize when only cell coverage is needed, not a search).
   - Densify must respect the same `pipe_radius` + `CompositeGrid` freeness as the parent planner.
   - AdaptiveStep “covered” semantics vs Theta\* Euclidean parent edges may need different corridor shapes.
   - Whether densify runs inside `path()` / a post-hook, or as an explicit runner flag (e.g. `--densify_path`).
