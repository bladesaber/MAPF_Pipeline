# planner/motion — Action / successor models

Edge generators for low-level search (`ROADMAP` connectivity).
Separate from heuristics: this defines **moves and edge costs**; heuristics estimate remaining cost.

## Interface

```cpp
struct Successor {
  CellCoord cell;
  Heading heading;   // travel direction of this edge
  double cost;       // length (+ optional turn penalty)
};

class ActionModel {
  void successors(from, in_heading, out) const;  // primary
  void successors(from, out) const;              // in_heading = none
  set_turn_weight                                // soft cost, default 0
  set_max_turn_deg                               // hard filter, default 180
};
```

`in_heading`: direction used to **enter** `from` (none at start).  
`Successor::heading`: direction of the step **leaving** `from`.

## Implementations

| Class | Moves | Base length |
|-------|-------|-------------|
| `SixConnected` | ±x, ±y, ±z | uniform `step_cost` |
| `EightConnected` | planar 8 (fixed z) | face `1`, diagonal `√2` × `step_cost` |
| `TwentySixConnected` | all 26 cube neighbors | face / edge / corner × `step_cost` |
| `AdaptiveShell` | Chebyshev shell `max(|d|)=s` | Euclidean · `step_cost`; planner sets `s` |

`AdaptiveShell::offsets_for(s)` caches shell offsets. Caller / planner sets `s` via
`set_step_size` (e.g. from clearance); motion does not query the grid.

**Soft turn cost**: `combine_edge_cost` adds cost when `in_heading` bends toward `heading`.  
**Hard turn filter**: `accept_turn` drops edges with turn angle > `max_turn_deg` (e.g. 90 keeps straight + right angles; reverse is already excluded when max < 180).

## Usage

```cpp
motion::SixConnected actions(res);
actions.set_turn_weight(1.0);     // optional soft cost
actions.set_max_turn_deg(90.0);   // optional hard filter
actions.successors(cell, in_heading, out);
```

Include `planner/motion/Motion.h` for all models.
