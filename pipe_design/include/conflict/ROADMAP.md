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

## Critical rule — detection must not be stricter than occupancy

Three places decide “blocked / conflict” in the stack:

| Layer | Code | Role |
|-------|------|------|
| High-level `[V]` | `ConflictAnalyzer::find_conflicts` | Declare group–group conflicts → CBS/PBS `[F]` |
| Low-level search | `CompositeGrid::is_free` | Accept / reject cells while routing |
| Path → obstacle | `chain_with_path` (multi/) | Bake a routed path into `ObstacleChain` |

**Rule (mandatory):** `ConflictAnalyzer` must follow the **most relaxed** detection rule
among these three — never stricter than what `CompositeGrid` + `chain_with_path` can
enforce.

Formally:

```text
Conflicts reported by Analyzer  ⊆  configurations that chain / is_free treat as illegal
```

Equivalently: if Analyzer flags a clash, applying the corresponding constraint /
occupancy must make low-level search reject that geometry.

**Why:** If Analyzer is **stricter** than occupancy:

1. Analyzer reports a conflict → high-level adds a constraint / extends the chain  
2. Low-level replan still sees the region as free (`is_free` / chain geometry disagree)  
3. Search returns essentially the same path  
4. Analyzer reports the **same** conflict again → **infinite CBS/PBS loop**

If Analyzer is **relaxed** (or aligned) relative to occupancy, every reported conflict
is enforceable → replan can make progress.

**Prefer alignment over arbitrary looseness:** use the same clearance predicate
(`dist < r_a + r_b` / `clearance < pipe_radius`) and compatible sphere–capsule
geometry. Sparse line-sampling in Analyzer is naturally no stricter than continuous
capsules in `chain_with_path`; do **not** add tighter margins only on the Analyzer
side. Constraint witnesses written to `ObstacleChain` must be strong enough that
`is_free` fails at those witnesses.

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
| `SampleKdTree` | P0 | Exact KD-tree on `SampledPoint` (+ radius weight); radius/knn search |
| `ConflictAnalyzer` | P0 | Required: `score(Conflict)`, `select_conflict`; optional cluster score/select |
| Line-discrete `normalize` | P1 | Independent `SampledPath` |
| `find_conflicts` / `find_clusters` | P1 | Sphere–sphere via KD radius search + graph adjacency |
| Constraint hint (cluster → chain sphere) | P2 | CBS `ObstacleChain` |

---

## Layout

```text
include/conflict/
  ROADMAP.md
  Types.h
  PathGraph.h
  SampleKdTree.h
  ConflictAnalyzer.h

src/conflict/
  PathGraph.cpp
  SampleKdTree.cpp
  ConflictAnalyzer.cpp
```
