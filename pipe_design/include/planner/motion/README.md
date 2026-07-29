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
  set_turn_weight                                // default 0
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

Turn penalty (optional): `combine_edge_cost` adds cost when `in_heading` bends toward `heading`.

## Usage

```cpp
motion::SixConnected actions(res);
actions.set_turn_weight(1.0);       // optional
actions.successors(cell, in_heading, out);
```

Include `planner/motion/Motion.h` for all models.
