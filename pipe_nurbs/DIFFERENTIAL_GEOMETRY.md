# Background: Differentiable Geometry of Surfaces

Notation follows a regular parametric patch
$$
\mathbf{r}\colon\Omega\subset\mathbb{R}^2\to\mathbb{R}^3,\qquad
\mathbf{r}(u,v)=\bigl(x(u,v),\,y(u,v),\,z(u,v)\bigr).
$$
Partial derivatives: $\mathbf{r}_u=\partial_u\mathbf{r}$, $\mathbf{r}_v=\partial_v\mathbf{r}$, $\mathbf{r}_{uu}=\partial_{uu}\mathbf{r}$, etc. Assume $\mathbf{r}_u\times\mathbf{r}_v\neq\mathbf{0}$ (regular / immersion).

Unit normal
$$
\mathbf{n}=\frac{\mathbf{r}_u\times\mathbf{r}_v}{\|\mathbf{r}_u\times\mathbf{r}_v\|}.
$$
In the fit path this is the geometric normal of $S(\cdot\,;\,P)$; for a scalar field $f$, a reference normal is $n^\ast=\nabla f/\|\nabla f\|$ on $\{f=\tau\}$.

---

## 1. First fundamental form

The **first fundamental form** is the inner product on the tangent plane induced by the Euclidean dot product of $\mathbb{R}^3$:
$$
\mathrm{I}(\mathbf{x},\mathbf{y})=\langle\mathbf{x},\mathbf{y}\rangle.
$$
It acts on **two** tangent vectors, not on a single “$\mathrm{d}\mathbf{r}$” by default.

### 1.1 Definition (two tangent vectors)

Write any tangent vectors in the basis $\{\mathbf{r}_u,\mathbf{r}_v\}$ with scalar coefficients $a,b,c,d$:
$$
\mathbf{x}=a\,\mathbf{r}_u+b\,\mathbf{r}_v,\qquad
\mathbf{y}=c\,\mathbf{r}_u+d\,\mathbf{r}_v.
$$
Then
$$
\begin{aligned}
\mathrm{I}(\mathbf{x},\mathbf{y})
&=\mathrm{I}\bigl(a\mathbf{r}_u+b\mathbf{r}_v,\,c\mathbf{r}_u+d\mathbf{r}_v\bigr)\\
&=ac\,\langle\mathbf{r}_u,\mathbf{r}_u\rangle+(ad+bc)\,\langle\mathbf{r}_u,\mathbf{r}_v\rangle+bd\,\langle\mathbf{r}_v,\mathbf{r}_v\rangle\\
&=E\,ac+F\,(ad+bc)+G\,bd,
\end{aligned}
$$
where the **coefficients of the first fundamental form** are
$$
\begin{aligned}
E&=g_{11}=\langle\mathbf{r}_u,\mathbf{r}_u\rangle,\\
F&=g_{12}=\langle\mathbf{r}_u,\mathbf{r}_v\rangle,\\
G&=g_{22}=\langle\mathbf{r}_v,\mathbf{r}_v\rangle.
\end{aligned}
$$
In matrix form, if $\mathbf{x}$ and $\mathbf{y}$ have coordinate columns $(a,b)^\top$ and $(c,d)^\top$,
$$
\mathrm{I}(\mathbf{x},\mathbf{y})
=
\begin{pmatrix}a&b\end{pmatrix}
\begin{bmatrix}E&F\\F&G\end{bmatrix}
\begin{pmatrix}c\\d\end{pmatrix}.
$$

**One-argument shorthand.** $\mathrm{I}(\mathbf{v})$ means the vector with itself:
$$
\mathrm{I}(\mathbf{v})=\langle\mathbf{v},\mathbf{v}\rangle=\|\mathbf{v}\|^2.
$$

### 1.2 Line element: $du$, $dv$ replace the coefficients

An infinitesimal displacement on the surface is the **single** tangent vector
$$
\mathrm{d}\mathbf{r}=\mathrm{d}u\,\mathbf{r}_u+\mathrm{d}v\,\mathbf{r}_v
$$
— so its coefficients in $\{\mathbf{r}_u,\mathbf{r}_v\}$ are $(a,b)=(\mathrm{d}u,\mathrm{d}v)$. The squared length is the quadratic form $\mathrm{I}(\mathrm{d}\mathbf{r})=\mathrm{I}(\mathrm{d}\mathbf{r},\mathrm{d}\mathbf{r})$:
$$
\mathrm{d}s^2
=E\,\mathrm{d}u^2+2F\,\mathrm{d}u\,\mathrm{d}v+G\,\mathrm{d}v^2.
$$
Here $\mathrm{d}u,\mathrm{d}v$ are **not** a different object from $a,b$: they are the calculus notation for those coordinate components when the tangent vector is an infinitesimal. For two **different** tangents one keeps four scalars $a,b,c,d$; for the line element one sets $(a,b)=(c,d)=(\mathrm{d}u,\mathrm{d}v)$.

Along a curve $\mathbf{r}(u(t),v(t))$, replace $\mathrm{d}u\to u'\,\mathrm{d}t$, $\mathrm{d}v\to v'\,\mathrm{d}t$ (or write $du/dt$, $dv/dt$).

### 1.3 Lengths and areas

- **Arc length** of $\mathbf{r}(u(t),v(t))$:
  $$
  L=\int\sqrt{E\Bigl(\frac{du}{dt}\Bigr)^2+2F\frac{du}{dt}\frac{dv}{dt}+G\Bigl(\frac{dv}{dt}\Bigr)^2}\,\mathrm{d}t.
  $$
- **Area element** (Lagrange identity):
  $$
  A=\iint_\Omega\|\mathbf{r}_u\times\mathbf{r}_v\|\,\mathrm{d}u\,\mathrm{d}v
  =\iint_\Omega\sqrt{EG-F^2}\,\mathrm{d}u\,\mathrm{d}v.
  $$

$\mathrm{I}$ is **intrinsic**: lengths and angles on the surface; isometries preserve $\mathrm{I}$.

### 1.4 Orthonormal tangent frame

When $F=0$ and $E=G=1$, $\{\mathbf{r}_u,\mathbf{r}_v\}$ is orthonormal. In general Gram–Schmidt $\mathbf{r}_u,\mathbf{r}_v$ to unit tangents $\mathbf{t}_1,\mathbf{t}_2$ for SD fitting: $\varepsilon_\alpha=(p-S)\cdot\mathbf{t}_\alpha$.

---

## 2. Second fundamental form

The **second fundamental form** $\mathrm{I\!I}$ (shape tensor) is a quadratic form on the tangent plane. With $\mathrm{I}$, it determines extrinsic curvatures (principal curvatures). Geometrically it measures how the surface bends **away from its tangent plane**.

<img src="figures/second_fundamental_form.png" alt="Second fundamental form: height of a nearby point above the tangent plane" width="600" />

*Figure (Wikipedia): at a point of tangency, a nearby surface point sits a vertical height above the tangent plane. That leading height is quadratic in the tangential displacement $(dx,dy)$ — this quadratic form is $\mathrm{I\!I}$.*

### 2.1 Motivation (graph + second-order Taylor)

**Step 1 — second-order Taylor (general).** Locally write the surface as a graph $z=f(x,y)$ with $f$ twice continuously differentiable. Expand at $(0,0)$:
$$
\begin{aligned}
f(x,y)
&=f(0,0)
+f_x(0,0)\,x+f_y(0,0)\,y\\
&\quad+\tfrac12 f_{xx}(0,0)\,x^2+f_{xy}(0,0)\,xy+\tfrac12 f_{yy}(0,0)\,y^2
+o(x^2+y^2).
\end{aligned}
$$

**Step 2 — local tangent coordinates.** Translate/rotate so the contact point is the origin and the **tangent plane is $z=0$**. The reason the first-order terms vanish is geometric: we take the tangent plane itself as the coordinate plane, so the two tangent vectors span the $(x,y)$-plane and the surface has **zero slope** relative to that plane at the origin. Equivalently $f_x=f_y=0$ there (and we place the origin on the surface so $f(0,0)=0$):
$$
f(0,0)=0,\qquad
f_x(0,0)=0,\qquad
f_y(0,0)=0.
$$
(The function is not identically zero — only value and first derivatives vanish at the contact point; bending appears at order two.)

**Step 3 — simplify equation.** Substituting Step 2 into Step 1 leaves only the quadratic (and higher) part:
$$
z=f(x,y)
=\tfrac12 f_{xx}(0,0)\,x^2+f_{xy}(0,0)\,xy+\tfrac12 f_{yy}(0,0)\,y^2
+o(x^2+y^2).
$$

Set
$$
L:=f_{xx}(0,0),\qquad
M:=f_{xy}(0,0),\qquad
N:=f_{yy}(0,0).
$$
Then
$$
z=L\frac{x^2}{2}+M\,xy+N\frac{y^2}{2}+\text{higher-order terms}.
$$
In the figure, the vertical dashed segment from the nearby point down to the tangent plane is this height $z$ (to leading order).

**Step 4 — read off the quadratic form $\mathrm{I\!I}$.** For an infinitesimal tangential displacement $(\mathrm{d}x,\mathrm{d}y)$, the same quadratic part is written as the second fundamental form
$$
\mathrm{I\!I}
=L\,\mathrm{d}x^2+2M\,\mathrm{d}x\,\mathrm{d}y+N\,\mathrm{d}y^2,
$$
so that the leading height is
$$
dz=\tfrac12\,\mathrm{I\!I}(\mathrm{d}x,\mathrm{d}y)+o(\mathrm{d}x^2+\mathrm{d}y^2)
$$

### 2.2 Classical notation (parametric surface)

For a regular patch $\mathbf{r}(u,v)$, take the unit normal
$$
\mathbf{n}=\frac{\mathbf{r}_u\times\mathbf{r}_v}{\|\mathbf{r}_u\times\mathbf{r}_v\|}.
$$
The coefficients of $\mathrm{I\!I}$ are
$$
L=\mathbf{r}_{uu}\cdot\mathbf{n},\qquad
M=\mathbf{r}_{uv}\cdot\mathbf{n},\qquad
N=\mathbf{r}_{vv}\cdot\mathbf{n}.
$$

**Why $\cdot\mathbf{n}$ (not $\mathbf{r}_{uu}$ alone).** The second derivative $\mathbf{r}_{uu}$ is a vector in $\mathbb{R}^3$. Split it into a part in the tangent plane and a part along $\mathbf{n}$:
$$
\mathbf{r}_{uu}
=\underbrace{\mathbf{r}_{uu}^{\parallel}}_{\in\mathrm{span}\{\mathbf{r}_u,\mathbf{r}_v\}}
+(\mathbf{r}_{uu}\cdot\mathbf{n})\,\mathbf{n}.
$$
In the graph picture of §2.1, the height $z$ above the tangent plane is measured **only along the normal**. The tangential piece $\mathbf{r}_{uu}^{\parallel}$ stays in the plane and does **not** contribute to $z$; only the normal component does. The same holds for $\mathbf{r}_{uv}$ and $\mathbf{r}_{vv}$. (These match $L=f_{xx}$, $M=f_{xy}$, $N=f_{yy}$ when the surface is $z=f(x,y)$ and $\mathbf{n}$ points along $z$.)

**Quadratic form / matrix.** With those coefficients, and with $\mathrm{d}u,\mathrm{d}v$ the coordinate components of a tangent vector in $\{\mathbf{r}_u,\mathbf{r}_v\}$ (as in §1.2),
$$
\mathrm{I\!I}=L\,\mathrm{d}u^2+2M\,\mathrm{d}u\,\mathrm{d}v+N\,\mathrm{d}v^2
=
\begin{pmatrix}\mathrm{d}u&\mathrm{d}v\end{pmatrix}
\begin{bmatrix}L&M\\M&N\end{bmatrix}
\begin{pmatrix}\mathrm{d}u\\\mathrm{d}v\end{pmatrix}.
$$

Equivalently $\mathrm{I\!I}=-\mathrm{d}\mathbf{r}\cdot\mathrm{d}\mathbf{n}$ after differentiating $\mathbf{n}\cdot\mathbf{r}_u=\mathbf{n}\cdot\mathbf{r}_v=0$.

**For an SDF** $s$ with Hessian $\mathbf{H}$ (useful when the target is $\{\phi=\tau\}$):
$$
L=-\mathbf{r}_u\cdot\mathbf{H}\cdot\mathbf{r}_u,\quad
M=-\mathbf{r}_u\cdot\mathbf{H}\cdot\mathbf{r}_v,\quad
N=-\mathbf{r}_v\cdot\mathbf{H}\cdot\mathbf{r}_v.
$$

### 2.3 Shape operator and curvatures

#### 2.3.1 Weingarten map

Here $\mathbf{n}$ is the **unit normal** of the surface. The **shape operator** / **Weingarten map** is a **linear operator on the tangent space** given by the directional derivative
$$
S(\mathbf{v})=-D_{\mathbf{v}}\mathbf{n},
$$
i.e. it measures how $\mathbf{n}$ changes as you move along the surface in the tangent direction $\mathbf{v}$. It is **directional**: the input is a tangent vector $\mathbf{v}$, and the output $S(\mathbf{v})$ is again a **tangent vector** (the tangential rate of change of $\mathbf{n}$ along $\mathbf{v}$). It is linked to the two fundamental forms by
$$
\mathrm{I\!I}(\mathbf{X},\mathbf{Y})=\mathrm{I}\bigl(S(\mathbf{X}),\mathbf{Y}\bigr)
$$
for tangent vectors $\mathbf{X},\mathbf{Y}$. In the $(\mathbf{r}_u,\mathbf{r}_v)$-basis its matrix is
$$
\bigl(S^i{}_j\bigr)
=
\begin{pmatrix}E&F\\F&G\end{pmatrix}^{-1}
\begin{pmatrix}L&M\\M&N\end{pmatrix}
=(\mathrm{I})^{-1}(\mathrm{I\!I}).
$$
Normal curvature in a unit tangent direction $\mathbf{t}$ is the associated Rayleigh quotient:
$$
\kappa_n(\mathbf{t})=\mathrm{I\!I}(\mathbf{t},\mathbf{t})=\mathrm{I}\bigl(S(\mathbf{t}),\mathbf{t}\bigr).
$$

#### 2.3.2 Eigenvalues and eigenvectors (principal curvatures / directions)

Diagonalize $S$ on the tangent plane. The **eigenvalues** are the **principal curvatures** $\kappa_1,\kappa_2$ — the maximum and minimum of the normal curvature $\kappa_n(\mathbf{t})$ over unit tangent directions $\mathbf{t}$.

The corresponding **eigenvectors** are the **principal directions**: the tangent directions you follow to attain that maximum or minimum curvature (orthogonal w.r.t. $\mathrm{I}$ when $\kappa_1\neq\kappa_2$).

| Quantity | Formula |
|----------|---------|
| Mean curvature | $H=\tfrac12(\kappa_1+\kappa_2)=\dfrac{LG-2MF+NE}{2(EG-F^2)}$ |
| Gaussian curvature | $K=\kappa_1\kappa_2=\dfrac{LN-M^2}{EG-F^2}=\dfrac{\det\mathrm{I\!I}}{\det\mathrm{I}}$ |
| Principal radii | $\rho_i=1/\kappa_i$ (Pottmann–Hofer SD weights) |

### 2.4 Link to fairness energies

Thin-plate fairness (`ROADMAP.md` §1)
$$
E_{\mathrm{fair}}=\int_\Omega\bigl(\|\mathbf{r}_{uu}\|^2+2\|\mathbf{r}_{uv}\|^2+\|\mathbf{r}_{vv}\|^2\bigr)\,\mathrm{d}u\,\mathrm{d}v
$$
penalizes all second derivatives of $\mathbf{r}$. $\mathrm{I\!I}$ keeps only the **normal** parts $(\mathbf{r}_{uu}\cdot\mathbf{n}$, etc.). Membrane energy $\int(\|\mathbf{r}_u\|^2+\|\mathbf{r}_v\|^2)$ is first-order and ties to $\mathrm{I}$.

---

## 3. Geodesics

A **geodesic** is a curve on a surface that is **as straight as possible**: among nearby curves on the surface connecting the same endpoints, it critically minimizes length (locally the shortest path).

### 3.1 Length from the first fundamental form

Let $\gamma(t)=\mathbf{r}\bigl(u(t),v(t)\bigr)$ be a curve on the patch. Its speed comes from $\mathrm{I}$ (§1.2):
$$
\Bigl\|\frac{\mathrm{d}\gamma}{\mathrm{d}t}\Bigr\|
=\sqrt{\mathrm{I}\Bigl(\frac{\mathrm{d}\gamma}{\mathrm{d}t},\frac{\mathrm{d}\gamma}{\mathrm{d}t}\Bigr)}
=\sqrt{E\Bigl(\frac{\mathrm{d}u}{\mathrm{d}t}\Bigr)^2+2F\frac{\mathrm{d}u}{\mathrm{d}t}\frac{\mathrm{d}v}{\mathrm{d}t}+G\Bigl(\frac{\mathrm{d}v}{\mathrm{d}t}\Bigr)^2}.
$$
Arc length is the integral of that speed:
$$
L[\gamma]
=\int_{t_0}^{t_1}\sqrt{E\Bigl(\frac{\mathrm{d}u}{\mathrm{d}t}\Bigr)^2+2F\frac{\mathrm{d}u}{\mathrm{d}t}\frac{\mathrm{d}v}{\mathrm{d}t}+G\Bigl(\frac{\mathrm{d}v}{\mathrm{d}t}\Bigr)^2}\,\mathrm{d}t.
$$
(Equivalently one often minimizes the **energy** $\mathcal{E}[\gamma]=\int\mathrm{I}(\mathrm{d}\gamma/\mathrm{d}t,\mathrm{d}\gamma/\mathrm{d}t)\,\mathrm{d}t$; constant-speed critical points of $\mathcal{E}$ are the same as those of $L$.)

### 3.2 Euler–Lagrange $\rightarrow$ geodesic equations

Treat $L$ (or $\mathcal{E}$) as a calculus-of-variations functional in the unknowns $u(t),v(t)$. The **Euler–Lagrange equations** for that Lagrangian yield the **geodesic equations**: a system of **second-order ODEs** for $\bigl(u(t),v(t)\bigr)$ (not a PDE on the surface field). In Christoffel form,
$$
\begin{aligned}
\frac{\mathrm{d}^2u}{\mathrm{d}t^2}
+\Gamma^u_{uu}\Bigl(\frac{\mathrm{d}u}{\mathrm{d}t}\Bigr)^2
+2\Gamma^u_{uv}\frac{\mathrm{d}u}{\mathrm{d}t}\frac{\mathrm{d}v}{\mathrm{d}t}
+\Gamma^u_{vv}\Bigl(\frac{\mathrm{d}v}{\mathrm{d}t}\Bigr)^2
&=0,\\[0.5em]
\frac{\mathrm{d}^2v}{\mathrm{d}t^2}
+\Gamma^v_{uu}\Bigl(\frac{\mathrm{d}u}{\mathrm{d}t}\Bigr)^2
+2\Gamma^v_{uv}\frac{\mathrm{d}u}{\mathrm{d}t}\frac{\mathrm{d}v}{\mathrm{d}t}
+\Gamma^v_{vv}\Bigl(\frac{\mathrm{d}v}{\mathrm{d}t}\Bigr)^2
&=0.
\end{aligned}
$$
The Christoffel symbols are determined by $E,F,G$ and their first derivatives (intrinsic — from $\mathrm{I}$ alone). Write $W=EG-F^2$ and subscripts for partials ($E_u=\partial_u E$, etc.):
$$
\begin{aligned}
\Gamma^u_{uu}&=\frac{GE_u-2FF_u+FE_v}{2W},&
\Gamma^u_{uv}&=\frac{GE_v-FG_u}{2W},&
\Gamma^u_{vv}&=\frac{2GF_v-GG_u-FG_v}{2W},\\[0.5em]
\Gamma^v_{uu}&=\frac{2EF_u-EE_v-FE_u}{2W},&
\Gamma^v_{uv}&=\frac{EG_u-FE_v}{2W},&
\Gamma^v_{vv}&=\frac{EG_v-2FF_v+FG_u}{2W}.
\end{aligned}
$$

**Geometric meaning.** The ambient acceleration $\mathrm{d}^2\gamma/\mathrm{d}t^2$ has no tangential component: it is parallel to the surface normal $\mathbf{n}$ (geodesic curvature vanishes). Equivalently, the covariant derivative of $\mathrm{d}\gamma/\mathrm{d}t$ along $\gamma$ is zero: $\nabla_{\mathrm{d}\gamma/\mathrm{d}t}(\mathrm{d}\gamma/\mathrm{d}t)=0$.

### 3.3 Remarks

- Geodesics are **intrinsic**: they depend only on $\mathrm{I}$, not on $\mathrm{I\!I}$.
- Great circles on a sphere are geodesics; straight lines in the plane are geodesics.
- Existence/uniqueness: for each point and each initial tangent direction there is a unique geodesic through that point in that direction (locally).

---

## 4. Riemannian manifold (symbol background)

Abstract language behind §§1–3. A surface patch in $\mathbb{R}^3$ is the main example used in this note.

- **$T_pM$ (tangent space).** For a smooth manifold $M$ and a point $p\in M$, $T_pM$ is the **tangent space** at $p$: the vector space of all tangent vectors to $M$ at $p$. It is a **space**, not a single vector; an individual tangent vector is $\mathbf{v}\in T_pM$. On a parametric patch, $T_pM=\mathrm{span}\{\mathbf{r}_u,\mathbf{r}_v\}$ at $p=\mathbf{r}(u,v)$.

- **$(M,g)$ (Riemannian manifold).** A pair where $M$ is a **smooth manifold** (the underlying shape / point set) and $g$ is a **Riemannian metric**: at each $p\in M$,
  $$
  g_p\colon T_pM\times T_pM\to\mathbb{R}.
  $$

  $g_p$ satisfies, for all $p\in M$:
  1. **Bilinearity:** $g_p(aX+bY,Z)=a\,g_p(X,Z)+b\,g_p(Y,Z)$ (and likewise in the second slot).
  2. **Symmetry:** $g_p(X,Y)=g_p(Y,X)$.
  3. **Positive definiteness:** $g_p(X,X)>0$ for $X\neq 0$.
  4. **Smoothness:** $g_p$ varies smoothly with $p$.

  In local coordinates $(x^1,\ldots,x^n)$, write the coordinate basis vectors $\partial_i=\partial/\partial x^i$. Then $g_{ij}$ is the metric pairing of basis vector $i$ with basis vector $j$:
  $$
  g_{ij}(x)=g(\partial_i,\partial_j)=g\Bigl(\frac{\partial}{\partial x^i},\frac{\partial}{\partial x^j}\Bigr).
  $$

  **Input / output.** $g_p$ takes **two tangent vectors** $X,Y\in T_pM$ and returns a **scalar** $g_p(X,Y)\in\mathbb{R}$ (their inner product). It does **not** return a vector, and it is **not** “the partial derivative of a function $f$” — even though the coordinate basis vectors $\partial_i$ can *act on* functions by differentiation, feeding $\partial_i,\partial_j$ into $g$ only produces the number $g_{ij}$.

  Collecting all components gives **one** $n\times n$ symmetric positive-definite matrix. Each entry $g_{ij}$ is a **scalar** at $x$ (not a small matrix):
  $$
  (g_{ij})
  =
  \begin{pmatrix}
  g_{11}&g_{12}&\cdots&g_{1n}\\
  g_{21}&g_{22}&\cdots&g_{2n}\\
  \vdots&\vdots&\ddots&\vdots\\
  g_{n1}&g_{n2}&\cdots&g_{nn}
  \end{pmatrix},
  \qquad
  g_{ij}(x)\in\mathbb{R}.
  $$
  One also writes $g=g_{ij}(x)\,dx^i\otimes dx^j$. On a surface patch, $n=2$ and
  $$
  (g_{ij})=\begin{pmatrix}E&F\\F&G\end{pmatrix}=\mathrm{I}.
  $$

  **What $g$ gives you.** Once you have $g$, you can measure geometric quantities:

  | Concept | Formula using $g$ |
  |---|---|
  | Length of a vector $X$ | $\|X\|=\sqrt{g(X,X)}$ |
  | Length of a curve $\gamma$ | $L(\gamma)=\int\sqrt{g(\dot\gamma,\dot\gamma)}\,dt$ |
  | Angle between $X,Y$ | $\cos\theta=\dfrac{g(X,Y)}{\|X\|\,\|Y\|}$ |
  | Volume element | $dV_g=\sqrt{\det(g_{ij})}\,dx^1\wedge\cdots\wedge dx^n$ (wedge product builds the oriented coordinate volume form; $\sqrt{\det g_{ij}}$ corrects when the coordinate basis is not orthonormal — including non-orthogonal bases — so the parallelepiped volume matches $g$) |
  | Gradient of a function | $g^{ij}\partial_j f$ |
  | Laplacian | $\Delta f=\dfrac{1}{\sqrt{\det g}}\,\partial_i\!\left(\sqrt{\det g}\,g^{ij}\partial_j f\right)$ |
  | Christoffel symbols | $\Gamma^k_{ij}=\tfrac12 g^{kl}(\partial_i g_{jl}+\partial_j g_{il}-\partial_l g_{ij})$ |
  | Curvature | Computed from the $\Gamma$'s (and their derivatives) |
