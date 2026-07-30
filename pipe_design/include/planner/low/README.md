# planner/low — Low-Level Grid Search

Single-agent path search on `CompositeGrid` (`ROADMAP` low-level).  
Owns **node storage** and the search loop; does **not** own grid / heuristic / open set / action model.

## Pieces

| File | Role |
|------|------|
| `SearchNode.h` | `SearchNode` (cell), `DirectedSearchNode` (cell + heading), state keys + hash |
| `AStar.h` / `.cpp` | `AStar<Node>` — steppable A\* (`reset` / `step` / `search`) |
| `DirectedAStar.h` / `.cpp` | `AStar<DirectedSearchNode>` + heading-aware `expand` |
| `Path.h` | Discrete cell path + `cost` |

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
| Different f | override `combine_key` |
| Theta* / any-angle | new planner type; reuse `Path` / keys as needed |

Do **not** put CBS, group scheduling, or topology in this folder.

## Status

`Ready` → `Running` → `Success` | `Failure`.  
Invalid / blocked starts or goals → `Failure` at `reset`. Empty open after expansions → `Failure`.
