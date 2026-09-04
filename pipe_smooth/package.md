# Package survey (`pipe_smooth`)

Candidate libraries for the rewrite. Not commitments. Aligns with `ROAMMAP.md` §1.5.

**Intended stack (summary):** Python + JAX (geometry AD) · SymPy (symbolic → codegen) · C++ backends via nanobind (OCC / PETSc / deal.II|MFEM as needed).

---

## 1. Automatic differentiation (Python)

Geometry loop: flexible $L,K,S$, sampling, multi-objective scalarization.

| Package | Strengths | Weaknesses | Fit for `pipe_smooth` |
|---------|-----------|------------|------------------------|
| **JAX** | `grad` / `jacrev` / `vmap`; functional style; good for scientific opt & batch sampling | Steeper API; debugging less “Pythonic” than PyTorch | **Preferred default** for geometry AD |
| **PyTorch** | Mature, dynamic graph, GPU, team familiarity (legacy code) | Heavier for pure scientific kernels; easy to grow messy tensor scripts | Solid alternative if JAX onboarding cost is high |

| Criterion | JAX | PyTorch |
|-----------|-----|---------|
| Swap objectives / compose costs | Excellent | Excellent |
| Jacobian / Hessian workflows | First-class | Possible, more boilerplate |
| GPU | Yes (XLA) | Yes |
| Ecosystem for NLP / IPOPT | Indirect | Indirect |

**Note:** CasADi is optional later for mid-size NLP + IPOPT, not listed as primary geometry AD here.

---

## 2. C++ stack

Use C++ at boundaries (CAD, large sparse PDE, hot kernels)—not for the whole package.

### 2.1 Python ↔ C++

| Package | Strengths | Weaknesses | Fit |
|---------|-----------|------------|-----|
| **nanobind** | Fast to build/load; modern C++ API; good for new projects | Younger than pybind11; fewer StackOverflow hits | **Preferred** for new bindings |
| **pybind11** | Ubiquitous docs/examples; battle-tested | Heavier compile/binary vs nanobind | Fine if team already standardized on it |

### 2.2 OpenCASCADE (CAD / BRep)

| Package | Strengths | Weaknesses | Fit |
|---------|-----------|------------|-----|
| **OpenCASCADE (OCC)** | BRep, curves/surfaces, already in repo; tube solids, port axes | Large API; binding/packaging friction | **Default** for CAD geometry / manufacturing solids |

### 2.3 PETSc (± MPI)

| Package | Strengths | Weaknesses | Fit |
|---------|-----------|------------|-----|
| **PETSc** (± **MPI**) | Scalable sparse KSP/SNES; industry backbone for large/high-dim discrete systems | Steep; MPI ops/dev overhead | **Default numerical backend** when PDE leaves “small Python FEM” |

### 2.4 High-order / large FEM: deal.II vs MFEM

| Package | Strengths | Weaknesses | Fit |
|---------|-----------|------------|-----|
| **deal.II** | Rich FEM tutorials; adaptivity; strong academic FEM base | Heavier C++ templates; build complexity | Prefer if team wants classic FEM + teaching/debug path |
| **MFEM** | Lightweight, MPI-friendly, GPU-aware; good for high-order / large DOF | Smaller “batteries included” vs deal.II | Prefer if scale / GPU / high-order is primary |

**Policy:** expose a PDE adapter (`forward` + $\partial J/\partial X$); pick deal.II *or* MFEM per backend—do not hard-wire FEniCS/dolfinx as the only option.

---

## 3. Symbolic mathematics

Up-front derivation and codegen (bases, curvature, simple adjoints)—not the runtime optim loop.

| Package | Strengths | Weaknesses | Fit for `pipe_smooth` |
|---------|-----------|------------|------------------------|
| **SymPy** | `pip`-friendly; `lambdify` / codegen to NumPy·JAX·C++; same env as AD stack | Slower on huge expressions | **Default** in-repo symbolic + codegen |
| **SymEngine** | Fast C++ symbolic core; can back SymPy | Narrower high-level API alone | Optional accelerator when expressions get large |
| **SageMath** | Broad CAS (algebra, geometry, many systems in one) | Heavy distro; awkward as a library dependency / CI | Optional **notebook** for hard hand derivations—not a runtime dep |

| Criterion | SymPy | SymEngine | SageMath |
|-----------|-------|-----------|----------|
| Codegen → JAX / C++ | Strong | Via tools / SymPy | Indirect export |
| Heavy pure-math exploration | Adequate | Speed for big trees | Best |

**Division:** SymPy (default) → generate JAX or C++/Eigen kernels; SymEngine if SymPy is too slow; Sage only for offline heavy algebra, then paste/export expressions back.

---

## 4. One-page recommendation

| Layer | Pick | Fallback |
|-------|------|----------|
| Geometry AD | **JAX** | PyTorch |
| Bindings | **nanobind** | pybind11 |
| CAD | **OpenCASCADE** | — |
| Large linear/nonlinear algebra | **PETSc** (± MPI) | — |
| Heavy FEM | **deal.II** *or* **MFEM** (adapter) | external CFD later |
| Symbolic | **SymPy** | SymEngine speedup; Sage offline only |
