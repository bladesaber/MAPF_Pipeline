# pipe_smooth TODOs

## Obstacle fusion (priority)

Agree: pairwise clear checks scale with **active segments × obstacles** (and group–group similarly). Dense sphere clouds or many CAD primitives make `n_obs` / `p_max_obs` blow up; packing + shuffle/truncate only caps the symptoms.

Need an **obstacle fusion / coarsening** stage before clear / union:

- [ ] Cluster nearby spheres (and maybe short capsules) into fewer proxies (merged sphere, AABB, or capsule hull)
- [ ] Optional hierarchical / BVH broadphase so only local candidates enter the packed active set
- [ ] Keep fine primitives for final verify; use fused set for optimize loops
- [ ] Expose fusion knobs in vis_bench (merge radius / max leaf count) and measure `n_obs` before vs after

Related band-aids already in place: `p_max_obs` / `p_max_grp` / `p_max_union` truncate — replace with fusion so budgets stay stable without silently dropping hot hits.
