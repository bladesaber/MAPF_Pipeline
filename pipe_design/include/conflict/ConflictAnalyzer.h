#pragma once

#include "conflict/PathGraph.h"
#include "conflict/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "topology/Types.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace pipe_design::conflict {

/// Base driver for group–group `[V]` analysis (normalize → graph → conflicts → clusters → score → select).
///
/// Named Analyzer (not Validator / Resolver): it **detects and ranks** clashes; it does
/// not mutate `ObstacleChain` or replan — that stays in `planner/high/`.
///
/// Before `normalize` / `analyze`, set `grid` (cell centers) and `plans` (edge radii).
///
/// Ranking is stored on `Conflict::score` / `ConflictCluster::score`.
/// Required overrides: `score(Conflict&)`, `score(ConflictCluster&)`, `select_conflict`.
class ConflictAnalyzer {
public:
    ConflictAnalyzer() = default;
    virtual ~ConflictAnalyzer() = default;

    void set_grid(const grid::StaticClearanceGrid* grid) { grid_ = grid; }
    void set_plans(const topology::GlobalTopologyPlan* plans) { plans_ = plans; }

    /// Sample spacing along a segment ≈ `factor * pipe_radius` (clamped above PathGraph eps).
    void set_sample_step_factor(float factor) { sample_step_factor_ = factor; }
    float sample_step_factor() const { return sample_step_factor_; }

    const grid::StaticClearanceGrid* grid() const { return grid_; }
    const topology::GlobalTopologyPlan* plans() const { return plans_; }

    /// Clear normalized paths, graphs, conflicts, clusters.
    virtual void clear();

    /// Default: clear → normalize → build_graphs → find_conflicts → find_clusters → score_all.
    /// Pointers must remain valid for the call; null entries are skipped.
    virtual void analyze(const std::vector<const topology::Task*>& tasks);

    /// Line-discretize Task paths into `sampled_paths_` (does not mutate `tasks`).
    virtual void normalize(const std::vector<const topology::Task*>& tasks);

    /// Build per-group `PathGraph` from `sampled_paths_`.
    /// Shared terminals merge via `PathGraph` quantized keys (`eps`).
    /// Each group must form exactly one connected component (throws otherwise).
    virtual void build_graphs();

    /// Cross-group sphere–sphere hits → `conflicts_` (all hits, not first-only).
    virtual void find_conflicts();

    /// Cluster `conflicts_` via PathGraph adjacency (same/neighbor witness nodes) → `clusters_`.
    virtual void find_clusters();

    /// Compute and store `conflict.score`. Subclass must implement.
    virtual void score(Conflict& conflict) const = 0;

    /// Compute and store `cluster.score`. Subclass must implement (base throws).
    virtual void score(ConflictCluster& cluster) const;

    /// Score every conflict, then every cluster (writes `.score` fields).
    virtual void score_all();

    /// Pick one conflict using stored `.score`. Subclass must implement.
    virtual const Conflict* select_conflict() const = 0;

    /// Optional: pick one cluster using stored `.score`. Default: argmax; nullptr if empty.
    virtual const ConflictCluster* select_cluster() const;

    const std::unordered_map<topology::GroupId, std::vector<SampledPath>>& sampled_paths() const {
        return sampled_paths_;
    }
    const std::unordered_map<topology::GroupId, PathGraph>& graphs() const { return graphs_; }
    const std::vector<Conflict>& conflicts() const { return conflicts_; }
    const std::vector<ConflictCluster>& clusters() const { return clusters_; }

protected:
    const grid::StaticClearanceGrid* grid_ = nullptr;
    const topology::GlobalTopologyPlan* plans_ = nullptr;
    float sample_step_factor_ = 1.f;

    /// Per-group normalized paths (one SampledPath per succeeded Task).
    std::unordered_map<topology::GroupId, std::vector<SampledPath>> sampled_paths_;
    std::unordered_map<topology::GroupId, PathGraph> graphs_;
    std::vector<Conflict> conflicts_;
    std::vector<ConflictCluster> clusters_;
};

}  // namespace pipe_design::conflict
