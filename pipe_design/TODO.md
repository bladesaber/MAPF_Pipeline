1. in multi-terminals path search condition, in order to generate topology like tree / branch rather than connect single long chain (this is usually caused by sequence path planning), after we setting past path as goals. we need to add different additional cost to different points of this past path goals. Such as adding higher additional cost to terminal point while encourage path search algorithm connect middle points of past path, then structure be more like tree.

2. `is_move_free(Vec3)` analytic path is O(N) over all baked spheres/capsules (+ chain). A* / long-step / Theta* call it often → too slow when N is large.
   Fix: after `bake`, build a spatial index (uniform grid or BVH) on obstacle AABBs. Query: (1) broadphase — obstacles whose AABB intersects the segment AABB; (2) narrowphase — `signed_clearance(segment, obs)` only on those k candidates (O(k), k ≪ N).

3. `ObstacleChain` collision is O(full ancestor list × all local obstacles) per query. CBS with many constraints makes A* edge checks expensive.
   Fix: spatial hash (or per-node COW index: reuse parent index, insert only local obstacles). Query only nearby candidates, then precise `signed_clearance` — same broadphase/narrowphase idea as (2).

4. Optional early-out for `is_move_free(Vec3)`: if `min(clearance(from), clearance(to))` is large enough vs segment length `L` (and `pipe_radius`), skip analytic checks.
   Caveat: endpoint clearance alone does not always imply the whole segment is free (obstacle can sit in the middle). Only use as a proven-safe prune with a sufficient condition, or as a heuristic — never as an unproven correctness skip.
