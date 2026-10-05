# pipe_nurbs ROADMAP

**Target:** \(\phi \rightarrow\) NURBS / STEP.

Centerlines and radius belong only to **building** \(\phi\). The NURBS stage must not see \(\gamma\) or \(r\).

## 1. SDF → NURBS

---

## Split

| Stage | In | Out |
|-------|----|-----|
| SDF | \(\gamma\), \(r\) from `pipe_smooth` | \(\phi(x)=\min_i(\mathrm{dist}(x,\gamma_i)-r_i)\) |
| NURBS | \(\phi\), \(\nabla\phi\) on \(\{\phi=0\}\) | watertight NURBS patches / STEP |

Error of the surface is vs \(\{\phi=0\}\), not vs a mesh.

---

## NURBS (from \(\phi\) only)

1. Sample \(\phi=0\) with normals \(n=\nabla\phi/\|\nabla\phi\|\).
2. Chart the implicit (quad layout / local T-mesh). Topology follows \(\phi\), including merged lobes when \(\kappa r \gtrsim 1\).
3. Fit NURBS (patches, T-spline then NURBS, or HB-splines). Few control points, sewable STEP.

**Not in this stage:** polar frames along \(\gamma\), circular sections from \(r\), pipe-sweep, boolean of tubes, mesh→NURBS reverse engineering.

---

## 2. Mesh → NURBS (auxiliary)

For an existing triangle mesh (QA isosurface or an external tessellation), not for the main \(\phi\) path.

1. Repair / simplify the mesh.
2. Quad layout or T-mesh on the surface.
3. Fit NURBS (same sew / STEP target as §1).

Error is vs the mesh. Use to compare against §1 or to ingest meshes that have no \(\phi\). Do not feed this mesh back as the source for §1.
