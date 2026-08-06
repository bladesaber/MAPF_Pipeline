# conflict — Group–group geometric validation (`[V]`)

Detail companion to [`../ROADMAP.md`](../ROADMAP.md) §1 / §8.

Detect, cluster, and rank **cross-group** pipe clashes. Input is routed Tasks from
a **TopologyPlan** (via `multi/`). Output feeds high-level `[F]` and TopologyPlan
`replan` for the next round.

---

## Unlike classical MAPF

| Classical MAPF | This domain (pipe radii) |
|----------------|--------------------------|
| Conflict = same vertex / edge at same time | Conflict = two **sphere** volumes violate clearance |
| One shared cell | Two witnesses (`point_a`, `point_b`) — often different positions |
| Single path vs path | Group = **tree / set** of paths → collect **all** hits, then cluster |
| `find_first` often enough | `find_all` is the default; pick later via score |

Same-group paths are **not** conflicts (sharing / ConnectedSets allowed).

---

## Design (agreed, keep simple)

**Sphere-centric pipeline.** Normalize every path to an ordered sphere list by
**line-discretizing** capsules (sample along the segment in world space). This is
**not** snapping to 6/8-connect grid paths. Original `Task.path` is never mutated;
normalized geometry lives in `SampledPath` only.

```text
Tasks (paths)
  → normalize(path)     → SampledPath[]     # independent sphere lists
  → build PathGraph     → per-group graph   # structure / distance-to-terminal
  → find_conflicts      → Conflict[]       # all sphere–sphere hits (cross-group)
  → find_clusters       → ConflictCluster[]  # using graph adjacency
  → score_all              → write Conflict/Cluster `.score`
  → select_conflict        → read `.score` (required)
  → select_cluster         → read `.score` (optional)
  → high/ [F]
```

| Piece | Role |
|-------|------|
| `SampledPoint` / `SampledPath` | Normalized tube; independent of original Path |
| `Conflict` | Atomic sphere–sphere evidence |
| `ConflictCluster` | Connected conflicts; may touch >2 groups |
| `PathGraph` | Per-group undirected network (`NodeId`); cluster + near-end |
| `ConflictAnalyzer` | `[V]` driver; required `score(Conflict)` / `select_conflict`; optional cluster variants |

### Why cluster + graph

One sample on A may hit many samples on B. Max-`deficit` alone is wrong: the path
often already collides earlier. Graph adjacency groups hits into a **conflict band**
that may touch **several groups**. Picking which Conflict inside a cluster to branch
on belongs to **Scorer** (subclass), not a fixed heuristic on the cluster type.
Graph also enables legacy-style “near terminal” selection later.

### Score

`score(Conflict)` / `select_conflict` **must** be overridden.
`score(ConflictCluster)` / `select_cluster` are optional (base: max-of-members / argmax).

---

## Types (P0)

```text
SampledPoint  = { center, radius }
SampledPath   = { group_id, task_id?, samples: SampledPoint[] }

Conflict = {
  group_a, group_b,
  task_a?, task_b?,
  sample_a, sample_b: SampledPoint,   # witness spheres
  deficit
}

ConflictCluster = {
  groups: GroupId[],                  # may be >2
  conflicts: Conflict[],
  max_deficit()
}
```

---

## Catalog

| Piece | Priority | Notes |
|-------|----------|-------|
| `Types.h` (`Conflict`, `ConflictCluster`, `SampledPath`) | P0 | Sphere-centric evidence |
| `PathGraph` | P0 | `NodeId` adj list + Dijkstra; no Boost |
| `ConflictAnalyzer` | P0 | Required: `score(Conflict)`, `select_conflict`; optional cluster score/select |
| Line-discrete `normalize` | P1 | Independent `SampledPath` |
| `find_conflicts` / `find_clusters` | P1 | Sphere–sphere + graph adjacency |
| Constraint hint (cluster → chain sphere) | P2 | CBS `ObstacleChain` |

---

## Layout

```text
include/conflict/
  ROADMAP.md
  Types.h
  PathGraph.h
  ConflictAnalyzer.h

src/conflict/
  PathGraph.cpp
  ConflictAnalyzer.cpp
```
