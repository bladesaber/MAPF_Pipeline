# pipe_nurbs ROADMAP

**Target:** $\phi \rightarrow$ NURBS / STEP.

Centerlines and radius belong only to **building** $\phi$. The NURBS stage must not see $\gamma$ or $r$.

## 1. Fit path: quad wrapping → fit

Main path for $\phi \rightarrow$ NURBS / STEP. Two stages; geometry of the spline is **re-tied** to a chosen target by optimizing cage / control parameters (unlike §2 conversion).

| Stage | Role | Out |
|-------|------|-----|
| **Quad wrapping** | Build a coarse, boundary-near **quad control cage** (or sparse quad layout) whose topology charts the shape | cage $C$ (quad mesh + optional knots / creases) |
| **Fit** | Each cage face drives a surface piece (NURBS / T-NURCC / G-NURBS); optimize cage vertices (and optionally weights, local refine) so the surface matches the target | watertight patches |

### 1.1 Fit objects (what the surface is tied to)

Fit means: given a surface $S(\cdot\,;\,P)$ from the cage, move $P$ so that $S$ stays close to the original shape (field, mesh, or point cloud). Here
$$P=\{\text{cage / control vertices, optional weights}\}.$$
Wrapping fixes the parameter domain / patch layout; fit then solves for $P$ (and may refine the cage).

**Complete target.** One data term plus optional normal and fairness:
$$
\min_P\ E(P)
= E_{\mathrm{data}}(P)+\mu E_{n}(P)+\lambda E_{\mathrm{fair}}(P).
$$
- $E_{\mathrm{data}}$: pull $S$ onto the chosen object (§1.1.1 field, §1.1.2 mesh, or §1.1.3 point cloud).
- $E_{n}$: align $S$ normals with a reference $n^\ast$ — $\nabla f/\|\nabla f\|$ on a field, or sample normals on a mesh / point cloud (omit if $\mu=0$):
$$E_{n}(P)=\int_S\Big\|\frac{\partial_u S\times\partial_v S}{\|\partial_u S\times\partial_v S\|}-n^\ast\Big\|^2\,dA
=\int_\Omega\Big\|\frac{S_u\times S_v}{\|S_u\times S_v\|}-n^\ast\Big\|^2\,\|S_u\times S_v\|\,du\,dv.$$
A sum over samples is just quadrature of this integral (the only form available when $n^\ast$ lives on a point cloud).
- $E_{\mathrm{fair}}$: keep $S$ smooth (thin-plate; a membrane $\int\|S_u\|^2+\|S_v\|^2$ is the weaker first-order variant):
$$E_{\mathrm{fair}}(P)=\int_\Omega \big(\|S_{uu}\|^2+2\|S_{uv}\|^2+\|S_{vv}\|^2\big)\,du\,dv.$$

#### 1.1.1 Field

Target: scalar field $f$ with isovalue $\tau$ (also level set / density). No mesh is geometric truth; error is vs $\{f=\tau\}$.

**Soft zero-level (discrete).** Sample parameters $\{(u_k,v_k)\}$ on the cage patches:
$$E_{\mathrm{field}}(P)=\sum_k w_k\,\big|f\big(S(u_k,v_k;P)\big)-\tau\big|^2.$$

**Continuous form (shrinking).** Mean absolute field on the surface:
$$E_{\mathrm{field}}(P)=\frac{1}{\mathrm{Area}(S)}\int_S \big|f(x)-\tau\big|\,dA(x),$$

#### 1.1.2 Mesh

Target: a triangle / quad surface $M$ with vertices or samples $\{p_i\}$. Common in papers (AMRTO, G-NURBS, generative B-Rep). Often $M$ is an isosurface of a field (**indirect**: field $\rightarrow$ mesh $\rightarrow$ fit); the fit residual below is then vs $M$, not vs $f$.

**Point-to-surface least squares.** Pair each sample $p_i\in M$ with a parameter $(u_i,v_i)$ and minimize
$$E_{\mathrm{mesh}}(P)=\sum_i \|S(u_i,v_i;P)-p_i\|^2.$$
How $(u_i,v_i)$ is chosen is the correspondence step below. Once parameters are frozen, $E_{\mathrm{mesh}}$ is (multi-)linear in $P$ for a fixed NURBS / G-NURBS basis; re-solve $P$, then optionally update correspondences and iterate.

**Correspondence (how to get $(u_i,v_i)$).**

1. **Closest point (foot).** Geometric projection onto the *current* spline:
$$(u_i,v_i)=\arg\min_{(u,v)\in\Omega}\|S(u,v;P^{(t)})-p_i\|^2.$$
Interior feet satisfy the orthogonality conditions
$$(p_i-S)\cdot S_u=0,\qquad (p_i-S)\cdot S_v=0.$$

2. **Harmonic mapping.** Map a mesh chart $M_j\subset M$ (one sparse-layout face) to the unit square $D=[0,1]^2$, then sample in $D$ to get fitting pairs. Fix boundary vertices of $M_j$ on $\partial D$ (e.g. by edge arc-length). For interior vertices $V'$, minimize discrete harmonic energy
$$E_{\mathrm{harm}}(V')=\frac12\sum_{(i,j)\in E}w_{ij}\,\|v'_i-v'_j\|^2$$
(cotangent or mean-value weights $w_{ij}$), i.e. solve the Laplace system $\Delta_{\mathrm{disc}}v'=0$ with those Dirichlet data. Each $p_i\in M_j$ then has $(u_i,v_i)=\varphi(p_i)\in D$, and the NURBS is fit in that parameter domain (AMRTO). Parameters are **fixed by the mesh chart**, independent of the evolving $P$ (unless the layout is rebuilt).

3. **Chord-length parameters.** Assign $(u_i,v_i)$ from cumulative chord lengths on a structured sample grid over the chart (classical NURBS Book surface approximation). On a $r\times s$ array of samples $\{p_{a,b}\}$,
$$
\tilde u_a=\sum_{k=1}^{a}\|p_{k,b}-p_{k-1,b}\|
\Big/\sum_{k=1}^{r-1}\|p_{k,b}-p_{k-1,b}\|
\quad\text{(per row $b$; then average over $b$)},
$$
and likewise $\tilde v_b$ from columns; optionally use centripetal $\sqrt{\|\Delta p\|}$ instead of $\|\Delta p\|$. Gives a **fixed** $(u,v)$ grid for LS / lofting when samples are already organized; no projection onto $S$. Poor if the sampling is highly irregular — prefer harmonic on freeform charts.

Closest point **depends on $P$** (reproject each outer iter); harmonic and chord-length are **fixed** until the chart / sample grid is rebuilt. Prefer closest point (or SD) for iterative geometric fit; harmonic for sparse layout charts on $M$ (AMRTO); chord-length for regular sample arrays.

**Squared-distance style** (tangent-plane / foot update, as in adaptive G-NURBS fitting). Uses closest-point feet. Let $S_i=S(u_i,v_i;P^{(t)})$ be the foot of $p_i$ on the current surface, with unit normal $n_i$ and orthonormal tangents $t_{i,1},t_{i,2}$. With point-plane distance $d_i=(p_i-S_i)\cdot n_i$ and tangential offsets $\varepsilon_{i,\alpha}=(p_i-S_i)\cdot t_{i,\alpha}$,
$$E_{\mathrm{SD}}(P)=\sum_i\Big(
d_i^2
+\sum_{\alpha=1}^{2}\frac{d_i}{d_i-\rho_{i,\alpha}}\,\varepsilon_{i,\alpha}^2
\Big),$$
where $\rho_{i,\alpha}$ are curvature radii in the $t_{i,\alpha}$ directions (Pottmann–Hofer SD error; drop the tangential sum for a pure point-plane model). Freeze feet / frames from $P^{(t)}$, solve for $P^{(t+1)}$, iterate.

**Minimax residual** (optional $E_{\mathrm{data}}$ instead of $L^2$):
$$E_{\infty}(P)=\sup_{x\in S(P)}\mathrm{dist}(x,M).$$
This is still a fit: shrink the worst gap, often by local cage refinement until $E_\infty<\delta$. Two-sided coverage (Hausdorff) is a conversion QA metric, not this fitter — see §2.

#### 1.1.3 Point cloud

Target: samples $\{p_i\}$ (optionally with normals $n_i$), no mesh connectivity. Same geometric residuals as §1.1.2; parameterization comes from projection onto the current $S$ or from a prior chart on the cage.

$$E_{\mathrm{pc}}(P)=\sum_i \|S(u_i,v_i;P)-p_i\|^2.$$
If samples have normals, set $n^\ast=n_i$ in $E_n$. Use for scans / sparse samples; if $\phi$ exists, prefer §1.1.1.

---

## 2. Conversion path (non-fit): extract → SubD → NURBS

Auxiliary path for QA isosurfaces or external meshes. **Not** least-squares fitting of control points to a target mesh / point cloud / field.

Pipeline: field or oriented samples $\rightarrow$ surface mesh $\rightarrow$ remesh / quad layout $\rightarrow$ Catmull–Clark cage $\rightarrow$ PCCM bicubic NURBS / STEP.

Error vs $\{\phi=0\}$ is only as good as the extracted mesh (and later remesh); the SubD→NURBS step is a **representation conversion**, not a fit that re-ties geometry to $\phi$. After conversion, report the mesh gap
$$L^\infty(S,M)=\sup_{x\in S}\mathrm{dist}(x,M),\qquad
d_H(S,M)=\max\Big\{\sup_{x\in S}\mathrm{dist}(x,M),\;\sup_{y\in M}\mathrm{dist}(y,S)\Big\}$$
($d_H$ also flags parts of $M$ that $S$ never covers). These numbers check the converted patches; they are not the §1 objective.

### 2.1 Field / point cloud → mesh

Extract a surface mesh from a scalar / indicator field (or from oriented samples). Output feeds §2.2.

#### 2.1.1 Poisson reconstruction

Oriented points $\rightarrow$ indicator $\chi$ $\rightarrow$ isosurface (Kazhdan et al. 2006). For scans / point clouds; if $\phi$ already exists, skip.

##### Given

Oriented samples $S = \{(p_i, n_i)\}_{i=1}^N$, with $p_i \in \mathbb{R}^3$ near an unknown closed surface $\partial M$ and $n_i$ an inward (or consistently oriented) unit normal. Goal: recover a watertight surface approximating $\partial M$.

##### Indicator field

Define the solid indicator
$$\chi(x) =
\begin{cases}
1, & x \in M \quad \text{(inside)} \\
0, & x \notin M \quad \text{(outside)}.
\end{cases}$$
Then $\nabla\chi = 0$ almost everywhere, and near $\partial M$ it aligns with the inward normal. Oriented samples are therefore samples of $\nabla\chi$.

##### Vector field from samples

Splat normals with a kernel $\varphi$ (e.g. Gaussian; in code often a compact octree / B-spline basis):
$$\vec{V}(x) = \sum_{i=1}^N n_i\, \varphi(x - p_i).$$
Optional area weights $a_i$ if sampling is non-uniform: $\vec{V}(x)=\sum_i a_i n_i\,\varphi(x-p_i)$. Then
$$\vec{V}(x) \approx \nabla\chi \quad\text{near }\partial M, \qquad \vec{V}(x) \approx 0 \quad\text{elsewhere.}$$
Raw $\vec{V}$ is generally **not** curl-free, so no exact $\chi$ with $\nabla\chi = \vec{V}$; fit in least squares.

##### Variational problem

Unconstrained quadratic energy over the volume $\Omega$:
$$E(\chi) = \int_\Omega \|\nabla\chi - \vec{V}\|^2\, dx
= \int_\Omega \big(\|\nabla\chi\|^2 - 2\langle\nabla\chi,\vec{V}\rangle + \|\vec{V}\|^2\big)\, dx.$$
Drop $\int\|\vec{V}\|^2$ (independent of $\chi$). Equivalent objective:
$$E(\chi) \equiv \int_\Omega \big(\|\nabla\chi\|^2 - 2\langle\nabla\chi,\vec{V}\rangle\big)\, dx.$$
Minimize $E(\chi)$ (free up to an additive constant of $\chi$).

**Euler–Lagrange.** For variation $\chi + \varepsilon\eta$ with compactly supported $\eta$,
$$\frac{d}{d\varepsilon}\Big|_{\varepsilon=0} E(\chi+\varepsilon\eta) = 2\int_\Omega \langle\nabla\eta,\, \nabla\chi - \vec{V}\rangle\, dx = 0.$$
Integration by parts (boundary terms vanish or are handled by the solver domain) gives
$$\int_\Omega \eta\, \big(\Delta\chi - \nabla\cdot\vec{V}\big)\, dx = 0 \quad\forall\eta
\quad\Rightarrow\quad \Delta\chi = \nabla\cdot\vec{V}.$$

**Shortcut.** Formally impose $\nabla\chi = \vec{V}$ and take divergence on both sides:
$$\nabla\cdot(\nabla\chi) = \nabla\cdot\vec{V} \quad\Rightarrow\quad \Delta\chi = \nabla\cdot\vec{V}.$$
Same Poisson equation; the variational form justifies it when $\vec{V}$ is inconsistent.

##### Solve and extract mesh

1. Discretize $\Delta\chi = \nabla\cdot\vec{V}$ on an adaptive octree / FEM / multigrid basis $\rightarrow$ sparse SPD system.
2. Fix the additive gauge (e.g. set mean of $\chi$, or match $\chi$ to sample locations).
3. Choose isovalue $\tau$ (often near the average of $\chi(p_i)$) and extract $\{\chi = \tau\}$ by MC / dual contouring.

**Screened Poisson** (Kazhdan–Hoppe): add a soft point constraint $\int \alpha\,(\chi - \chi_0)^2$ so the zero level stays closer to the input samples.


#### 2.1.2 Marching Cubes

Primal grid isosurface: vertices on edges, triangle mesh. Simple; coarse grids show staircasing / aliasing.

##### Given

Scalar field $f:\mathbb{R}^3\rightarrow\mathbb{R}$ sampled on a regular (or octree) primal grid, and isovalue $\tau$. Goal: approximate the level set
$$S = \{x \mid f(x)=\tau\}$$
by a triangle mesh. In this repo $f$ is $\phi$ (SDF) or $\chi$ (Poisson indicator).

##### Cell classification

For each cube with 8 corner samples $f_0,\ldots,f_7$, assign a bit per corner:
$$b_i = \begin{cases} 1, & f_i \ge \tau \\ 0, & f_i < \tau. \end{cases}$$
The 8-bit pattern indexes one of $2^8=256$ configurations (15 unique topologies up to rotation/reflection). Corners with opposite signs imply that $S$ crosses the shared edge.

##### Edge vertices (linear interpolate)

If edge $(a,b)$ has $f_a < \tau \le f_b$ (or swapped), place a vertex by 1D linear interpolation:
$$x = a + t\,(b-a), \qquad t = \frac{\tau - f_a}{f_b - f_a} \in [0,1].$$
All mesh vertices lie on **primal edges**; no vertex inside the cell.

##### Triangulation lookup

A fixed table maps each bit pattern to a list of triangles whose vertices are those edge points. Adjacent cubes share edge vertices $\Rightarrow$ the mesh is watertight (no cracks) when the table is consistent across faces.

**Ambiguity.** Some face/internal configurations admit two topologically different triangulations; wrong choice can create holes or non-manifold edges. Standard fixes: asymptotic decider, preferred polarity, or dual methods (§2.1.3).

##### Properties / limits

- Output is always triangles (not quads); connectivity follows the grid, not surface curvature.
- Grid-aligned staircasing and normal noise on coarse voxels; refine the grid or remesh before §2.2.
- Exact only for trilinear $f$ inside each cube; true $\phi$ / $\chi$ are approximated by the samples.

#### 2.1.3 Dual Marching Cubes

Dual of the MC surface (Nielson): vertices inside cells, **quad** patches, same separating topology as MC. Preferable when the mesh feeds quad layout (§2.2).

##### Duality vs MC (§2.1.2)

MC places vertices on **primal edges** and faces inside cubes. Dual MC inverts that association:

| | Marching Cubes | Dual Marching Cubes |
|---|---|---|
| Vertex | on a grid edge (sign change) | inside a cube (one per surface patch in that cube) |
| Face | triangle(s) inside a cube | quad around a grid edge (four cubes that share the edge) |

Same bit-pattern / separating property as MC; connectivity is the dual graph of the MC patch complex.

##### Construction

1. Classify cubes as in MC; for each cube that the level set crosses, identify each connected **patch** of the MC triangulation inside that cube (usually one; two sheets $\Rightarrow$ two patches).
2. Place one dual vertex $v$ per patch, in the interior of the cube. Default: centroid of that patch’s edge-crossing points. With gradients / Hermite data: minimize a quadratic error function (QEF) so $v$ snaps to sharp edges / corners (same placement idea as Dual Contouring).
3. For every primal grid edge that $S$ crosses, the four adjacent cubes each contribute a dual vertex; connect them into one **quad**. Shared edge $\Rightarrow$ shared quad edge $\Rightarrow$ watertight mesh.

Result: quad-dominant (typically pure quad) mesh; no MC-style skinny triangles from forced diagonals.

##### Properties / limits

- Quads are usually better shaped than MC triangles; still grid-tied — coarse voxels retain blockiness unless vertices are QEF-fitted or the grid is refined.
- Multiple sheets in one cell get multiple vertices (avoids pinching separate components into one point).
- Not yet a Catmull–Clark cage: valence / sizing still need remesh or field-guided quad layout before §2.2.
- **Name collision:** Schaefer–Warren “Dual Marching Cubes” builds a feature-aligned **dual grid** then runs (generalized) MC on it — different algorithm, same goal (feature-aware, crack-free isosurface). Dual Contouring is the close cousin that emits one quad per intersected primal edge with QEF vertices.

### 2.2 Mesh → NURBS

For an existing triangle mesh (from §2.1 or external tessellation).

1. Repair / simplify; remesh for sizing (more uniform vertex spacing / feature-aware density); triangle → quad mesh (quad layout).
2. Treat the quad mesh as a Catmull–Clark control cage (optional 1-level refine to isolate extraordinary vertices).
3. SubD → NURBS via PCCM (or equivalent): one bicubic patch per coarse quad → sew / STEP.

**Singularity problem:** NURBS patches need a regular grid network (tensor-product control net). A singularity is a vertex whose degree (valence) is not equal to 4; it breaks that grid regularity and causes unbalanced patch layout / continuity near the vertex (hard for exact bicubic NURBS; PCCM approximates there).

#### 2.2.1 Laplacian smoothing

##### Update

For vertex $v_i$ with neighbor set $\mathcal{N}(i)$,
$$\bar{v}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} v_j, \qquad
v_i \leftarrow (1-\lambda)\, v_i + \lambda\, \bar{v}_i, \quad \lambda \in (0,1].$$
Uniform weights above; cotangent / area weights better on irregular triangles. Iterate a few times.

**Local smooth.** Restrict updates to a vertex mask / ROI, or use spatially varying $\lambda$ (e.g. $\lambda\approx 0$ on features). Topology and resolution unchanged — not SubD-style local refinement.

##### Properties / limits

- Smooths high-frequency grid noise; over-iteration **shrinks volume** and rounds sharp features.
- Mitigations: small $\lambda$ / few iters; Taubin ($\lambda$–$\mu$) to reduce shrinkage; project back onto $\{\phi=\tau\}$ when $\phi$ is available.
- Does **not** fix bad topology or place singularities — that remains remesh / cross-field (§2.2 step 1) and PCCM (§2.2 step 3).

#### 2.2.2 Subdivision

**Subdivision:** full iterative algorithmic process to smooth a curve / surface (limit under repeated rules). **Refinement:** topological or geometric operation of splitting a coarse mesh (or simplex) into a larger set of smaller elements.

##### Refinement vs subdivision

| | Refinement | Subdivision (Catmull–Clark) |
|---|---|---|
| Goal | Add resolution / improve sizing or isolate singularities | Define a **smooth limit surface** from the control cage |
| Geometry | Often remesh / insert vertices; may stay on or near the old surface | Each step inserts face/edge/vertex points **and** repositions old vertices by fixed rules |
| Topology | Can be local (ROI remesh, T-mesh) | Classic CC is **global** (whole mesh); no true local refinement |
| Limit | Still a mesh | Infinite refine $\rightarrow$ $C^2$ (regular) / $C^1$ (extraordinary) surface |

In this pipeline: **remesh / Laplacian** = refinement-style cleanup; **CC** = smooth surface model for the cage.

##### Catmull–Clark (one step, quad mesh)

For each step on a pure quad mesh:

1. **Face point:** centroid of each face with vertices $v_1,\ldots,v_k$: $F = \frac{1}{k}\sum_{i=1}^{k} v_i$.
2. **Edge point:** for edge $(v_a,v_b)$ with adjacent face points $F_0,F_1$: $E = \frac{v_a + v_b + F_0 + F_1}{4}$.
3. **Vertex point:** for old vertex $V$ of valence $n$, let $\bar{F}$ be the average of the $n$ adjacent face points and $\bar{R}$ the average of the midpoints of the $n$ incident edges: $V' = \frac{\bar{F} + 2\bar{R} + (n-3)V}{n}$.
4. Connect: each old quad $\rightarrow$ four quads. Extraordinary vertices (valence $\neq 4$) persist.

**Output:** a refined quad control mesh (and, in the limit of infinite steps, a smooth surface: $C^2$ on regular valence-4 regions $=$ uniform bicubic B-spline; $C^1$ at extraordinary vertices). Finite levels feed PCCM (§2.2 step 3).

#### 2.2.3 PCCM (Patching Catmull–Clark Meshes)

Peters (SIGGRAPH 2000): convert a Catmull–Clark control mesh into standard **bicubic NURBS** patches for CAD / STEP exchange.

**Idea.** Regular CC regions are already uniform bicubic B-splines. PCCM builds **one bicubic NURBS patch per coarse quad** (degree $p=3$, order $p+1=4$; typically a $4\times 4$ control net in clamped / Bézier form), optionally after 0–1 CC steps to isolate extraordinary vertices, via knot insertion; near extraordinary vertices, adjust control points so patches meet at least $G^1$ and interpolate the CC limit point.

| | Regular (valence 4) | Extraordinary neighborhood |
|---|---|---|
| vs CC limit | exact | approximate |
| Join | parametric $C^2$ | at least tangent ($G^1$) |
