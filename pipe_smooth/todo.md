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


### union cost has big problem, it is scale by radius

### 关于当前相邻segment reverse存在不一致问题
解决方法1: 
	在不成环的图里面总是可以通过改变uage的方向从而实现无方向冲突的图

解决方法2:
	可能不需要使用bend energy这种繁琐的cost,可能使用两端点P1, P2的向量差就能解决这个问题,而且是完备的:
	1. 但由于两端点距离会影响这个度量,所以应该以端点距离作为基,从而无量纲化
	2. P1,P2的向量角度差与幅度差决定最大曲率
	所以为每个HermiteSegment添加端点sign也是一个解决方法


