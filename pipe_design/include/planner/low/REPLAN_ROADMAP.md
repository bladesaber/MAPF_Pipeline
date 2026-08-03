# Path Replan Methods

Incremental low-level replan after `ObstacleChain` / occupancy deltas.

Detail companion to [`../../../ROADMAP.md`](../../../ROADMAP.md) §4.2.

---

## Re-search

Baseline: discard the previous search state (and optionally ignore $P$) and run a fresh low-level search on the updated grid.

Given original query $(s, g)$ (or the same multi-start / multi-goal sets) and an updated grid:

1. Clear open / node map (cold `reset`).
2. **A\*** (or the active low-level planner) from the **real** starts to the **real** goals under new `is_free(·, r)`.
3. Solution is the reconstructed path; no stitching with $P$.

Optional pre-check: if a previous path $P$ is still entirely free, may reuse $P$ without search (same as Stitching step 2).

---

## Stitching

Given previous path

$$
P = (c_0 = s, \ldots, c_n = g)
$$

and an updated grid:

1. **Scan** $P$ under new `is_free(·, r)`.
2. If **entire** $P$ **free** → reuse $P$ (no search).
3. Else find the blocked segment and set anchors:
   - $a$ = last free cell of the leading free prefix (enter conflict before the next cell);
   - $b$ = first free cell of the trailing free suffix (leave conflict at / after this cell).
   If $s$ is blocked or no free suffix reaches $g$, fail or fall back to **Re-search**.
4. **Local A\*** on the new grid: single start $a$ ($g(a) = 0$), single goal $b$.
5. Reconstruct: search path is $a \to \ldots \to b$; full path is

$$
P[s \ldots a] \;+\; (a \to \ldots \to b) \;+\; P[b \ldots g]
$$

(omit duplicate $a$ / $b$ at joints). Re-check the concatenated path with `is_free`.

---

## Local Repair replan

Broader than **Stitching**: repair only inside an influence region $R$ induced by the occupancy delta $\Delta$ (new/removed spheres/capsules), optionally restricting expansion to a corridor around $R$. Path-anchored stitch ($a \to b$) is the usual instance when $P$ is available.

Given $P$, updated grid, pipe radius $r$, and $\Delta$:

1. **Build** $R$: cells whose `is_free(·, r)` may change under $\Delta$ (obstacle neighborhood / AABB, optionally dilated by $k$ cells).
2. **Scan** $P$ on the new grid. If $P$ is entirely free and $P \cap R = \emptyset$ → reuse $P$. If $s$ or $g$ is blocked → fail or **Re-search**.
3. **Anchors** from $P$ vs $R$ (or vs the blocked segment of $P$):
   - $a$ = last cell of $P$ before entering $R$ or entering the break;
   - $b$ = first cell of $P$ after leaving $R$ or entering the break.
   Multiple breaks → one segment per break, or one $a..b$ spanning all breaks.
4. **Local A\*** $a \to b$ ($g(a) = 0$) on the new grid, with search domain = $R$ dilated by $k$ (cells outside treated as blocked). If $\Delta$ is unknown, $R$ may be taken as a dilation of the blocked subpath of $P$ (reduces to corridor-limited Stitching).
5. **Concatenate** $P[s \ldots a] + (a \to \ldots \to b) + P[b \ldots g]$; re-check full path with `is_free`.
6. On local failure or check fail: increase $k$ (grow $R$) and retry; if still failing → **Re-search** $(s, g)$.


