# pipe_nurbs

Stage-2 surface from `pipe_smooth` parametric tubes (Hermite centerlines + radius).

**Goal:** signed-distance solid → editable **NURBS / BRep** (STEP), not marching-cubes mesh as the deliverable. MC (or similar) may be an intermediate only; final exchange is CAD-grade surfaces.

Upstream: `pipe_smooth` graph / `(p, v, r)`. Downstream: STEP / IGES / OCC BRep for edit & FEM prep.

---

## Layout

| path | role |
|------|------|
| `io/` | load `pipe_smooth` exports (graphs, radii, topology) |
| `sdf/` | tube / capsule SDF and unions (robust under high \(\kappa\), junctions) |
| `fit/` | SDF or section samples → NURBS / B-spline surfaces (patches, sweeps, fits) |
| `export/` | write STEP / IGES / OCC shapes |

---

## Design notes

- Large curvature and complex junctions make OCC pipe-sweep + boolean brittle; SDF is the solid oracle.
- Triangle meshes from isosurfacing are large and hard to edit — keep them optional (QA / viz), not the product.
- Editable source of truth stays parametric (`pipe_smooth` DOF) plus fitted NURBS controls; regenerate rather than hand-edit dense meshes.

Algorithms and milestones: `ROADMAP.md`.

Status: scaffold; target is SDF → NURBS (mesh = QA only).
