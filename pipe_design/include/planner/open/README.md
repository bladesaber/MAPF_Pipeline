# planner/open — Open Set for Best-First Search

Priority queue used by low-level A\* / best-first planners (`ROADMAP` §4.2).
Planners own `SearchNode` storage; this folder only orders **non-owning** `SearchNode*`.

## Role

```text
LowLevelPlanner::search
    │
    ├─ closed / g-map  (planner)
    └─ OpenSet         (this folder)  ← which node to expand next
```

Swappable via `OpenSet*` (`OpenSet` = `OpenSetBase<SearchNode, MinFgCompare>`).
Do not put CBS / topology / grid logic here.

## Contract (`OpenSet` / `OpenSetBase`)

`OpenSetBase<Node, Compare>` is the generic interface; compare is a strict
worse-than functor (default `MinFgCompare`: min `f`, tie → larger `g`).
LPA* uses `LPAOpenSet` (lex `[k1;k2]`, remove+insert) — not an `OpenSet` backend.

| Op | Meaning |
|----|---------|
| `push` | Insert node currently on open |
| `pop` | Remove best node (undefined if empty) |
| `top` | Best node without removing (undefined if empty) |
| `contains` | Whether this open set tracks the node |
| `remove` | Unlink arbitrary node if present; `true` if it was in the set |
| `decrease_key` | After `node->f` improved while on open; `true` = fixed in place, `false` = caller must `push` again |
| `clear` / `empty` / `size` | Lifecycle |

**Ordering (default min-heap):** smaller `f` first; if `f` equal, prefer **larger `g`** (more progress).

**Ownership:** never delete nodes. Heap handles live in the open-set side maps — **not** on `SearchNode` (see `SearchNode.h`).

## Implementations

| Class | `decrease_key` | `remove` | Notes | Typical use |
|-------|----------------|----------|-------|-------------|
| `LazyBinaryHeap` | Always `false` → re-`push`; stale skipped on pop | Logical only (caller must `Closed`) | Fast when reopenings are rare | **Default grid A\*** |
| `BinaryHeap` | `true` (index map + sift) | True unlink | Exact, simple | Baseline / correctness |
| `PairingHeap` | `true` (meld / detach) | True unlink | Strong amortized decrease | Many decrease_keys / Focal OPEN |
| `LPAOpenSet` | n/a — `remove` + re-`push` | Via `PairingHeapBase` | Lex `[k1;k2]`; not an `OpenSet` | **LPA\* / D\* Lite U** |
| `BucketHeap` | `true` (move between bins) | True unlink | Dial-style; `bin = floor(f/delta)` | Bounded, discrete-ish `f` |

### Lazy vs true decrease

Unified caller pattern after updating `f`/`g`:

```text
if (first time on open):  push(node)
else if (!decrease_key(node)):  push(node)   // Lazy returns false
```

- **True decrease** (`Binary` / `Pairing` / `Bucket`): `decrease_key` fixes order in place → `true`.
- **Lazy** (`LazyBinaryHeap`): `decrease_key` is a no-op → `false`; caller re-`push`es. Keep `status == Open` while logically open; set `Closed` on expand so stale duplicates are discarded.

### `BucketHeap` specifics

- Construct with `delta > 0` and optional `max_cost` (inclusive). `f` outside `[0, max_cost]` throws.
- Within a bin, `pop` picks the exact best by `f`/`g` (not only the quantized key).

## A\* usage sketch

```text
push(start)                    // status = Open
while !empty:
  n = pop()
  if n already Closed / stale: skip   // especially LazyBinaryHeap
  n.status = Closed
  for each neighbor m:
    if improved g/f:
      update m
      if m was New: push(m)
      else if !decrease_key(m): push(m)   // lazy
```

## Files

```text
OpenSet.h           OpenSetBase<Node,Compare> + MinFgCompare + using OpenSet
BinaryHeap.h/.cpp
LazyBinaryHeap.h/.cpp
PairingHeap.h/.cpp
LPAOpenSet.h/.cpp   LPA* queue U (lex LPAKey; remove+insert)
BucketHeap.h/.cpp
tests/test_open_set.cpp   correctness + speed microbench
```

## Extending

1. Subclass `OpenSet` (i.e. `OpenSetBase<SearchNode, MinFgCompare>`), or instantiate
   `OpenSetBase` with another `Compare` / `Node` for experiments.
2. Keep handles off `SearchNode`.
3. Add a deterministic unit test in `tests/test_open_set.cpp` (order, tie-break, decrease / stale).
4. Prefer composition: planner holds `std::unique_ptr<OpenSet>`, not a hard-coded heap type.
