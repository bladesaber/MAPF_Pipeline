# pipe_smooth TODOs

## Curvature cost via endpoint features (estimate later)

Idea: replace sampled bend energy with a frozen map from dimensionless Hermite ends → κr risk.

- Features (normalize by radius): θ0, θ1, ‖v0‖/r, ‖v1‖/r, ‖p1−p0‖/r
- Label: whether max(κ)·r exceeds threshold (or better: regress κ_max·r / soft cost)
- Train offline, freeze; use as cheap K surrogate in optimize
- Open: planar vs 3D DOF; joints between segs; classifer vs soft regressor for AD

## Obstacle fusion (priority)

Agree: pairwise clear checks scale with **active segments × obstacles** (and group–group similarly). Dense sphere clouds or many CAD primitives make `n_obs` / `p_max_obs` blow up; packing + shuffle/truncate only caps the symptoms.

Need an **obstacle fusion / coarsening** stage before clear / union:

- [ ] Cluster nearby spheres (and maybe short capsules) into fewer proxies (merged sphere, AABB, or capsule hull)
- [ ] Optional hierarchical / BVH broadphase so only local candidates enter the packed active set
- [ ] Keep fine primitives for final verify; use fused set for optimize loops
- [ ] Expose fusion knobs in vis_bench (merge radius / max leaf count) and measure `n_obs` before vs after

Related band-aids already in place: `p_max_obs` / `p_max_grp` / `p_max_union` truncate — replace with fusion so budgets stay stable without silently dropping hot hits.


### union cost radius scale — done
`J_union` uses `(d/(r_a+r_b))² Δs` with frozen `r_sum` in `UnionHit` / packed buf.

### 关于当前相邻segment reverse存在不一致问题
解决方法1: 
	在不成环的图里面总是可以通过改变uage的方向从而实现无方向冲突的图

#### 参考：usage / segment 方向消歧（图论）

**设定.** 格子 \(b_{p,s}\in\{0,1\}\)：path \(p\) 走 segment \(s\) 时是否 `reverse`。  
允许：**翻行** \(\tau_p\)（usage 掉头）、**翻列** \(\sigma_s\)（改存贮 \(u\to v\)）；不可单改一格。  
目标：\(\tau_p\oplus\sigma_s\oplus b_{p,s}=0\)，即 GF(2) 方程 \(x_p+y_s=b_{p,s}\)。

**局部 mutex.** 两 path 若有 ≥2 个不相交共享连通块，且一块同向、一块反向 → 2×2 奇偶冲突，无解。  
两 path 交至多一块连续子路时，相对同向/反向在整块上为常数 → 该 2×2 恒相容。

**注意（命题过强）.** 「不成环 + 两两至多一个共享连通域 ⇒ 一定能全 `reverse=false`」**不成立**。  
反例：星形 + 三条叶–叶 path，两两只共一边，仍无法同时定向。  
树上任意两 path 的边交至多一条子路，故无「双块」mutex，但仍可能有 ≥3 条 path 的全局冲突。

**文献名.** Tree / Unordered Tree Orientation；Steiner Forest Orientation（有序端点）。  
判定：相交 path 建 conflict graph（2-染色 / 边方向是否矛盾）；\(k=0\) 全保留时多项式可判，非恒可行。

**何时总可以.** 充分条件例如：所有 path 共一端点（根向外 arborescence；如 `A→B*`）；或 path 族 laminar。  
实现上：树上跑一遍 conflict / 解 \(x_p+y_s=b\) 即可，不必假设恒成立。

**与切向无关.** 全 `reverse=false` 只统一 travel 与存贮边向；节点 `v` 反弦 / Hermite 过冲是另一问题。



