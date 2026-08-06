#pragma once

#include "conflict/PathGraph.h"
#include "conflict/Types.h"
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
/// Ranking is stored on `Conflict::score` / `ConflictCluster::score`.
/// `score(...)` **writes** those fields; `select_*` only **reads** them (no recompute).
///
/// Required overrides: `score(Conflict&)`, `select_conflict`.
/// Optional overrides: `score(ConflictCluster&)`, `select_cluster`.
class ConflictAnalyzer {
public:
    ConflictAnalyzer() = default;
    virtual ~ConflictAnalyzer() = default;

    /// Clear normalized paths, graphs, conflicts, clusters.
    virtual void clear();

    /// Default: clear → normalize → build_graphs → find_conflicts → find_clusters → score_all.
    virtual void analyze(const std::vector<topology::Task>& tasks);

    /// Line-discretize Task paths into `sampled_paths_` (does not mutate `tasks`).
    virtual void normalize(const std::vector<topology::Task>& tasks);

    /// Build per-group `PathGraph` from `sampled_paths_` (junction merge policy in subclass / later).
    virtual void build_graphs();

    /// Cross-group sphere–sphere hits → `conflicts_` (all hits, not first-only).
    virtual void find_conflicts();

    /// Cluster `conflicts_` using path-graph adjacency → `clusters_`.
    virtual void find_clusters();

    /// Compute and store `conflict.score`. Subclass must implement.
    virtual void score(Conflict& conflict) const = 0;

    /// Compute and store `cluster.score`. Default: max of members' `.score` (score members first).
    virtual void score(ConflictCluster& cluster) const;

    /// Score every conflict, then every cluster (writes `.score` fields).
    virtual void score_all();

    /// Pick one conflict using stored `.score`. Subclass must implement.
    virtual const Conflict* select_conflict() const = 0;

    /// Optional: pick one cluster using stored `.score`. Default: argmax; nullptr if empty.
    virtual const ConflictCluster* select_cluster() const;

    const std::vector<SampledPath>& sampled_paths() const { return sampled_paths_; }
    const std::unordered_map<topology::GroupId, PathGraph>& graphs() const { return graphs_; }
    const std::vector<Conflict>& conflicts() const { return conflicts_; }
    const std::vector<ConflictCluster>& clusters() const { return clusters_; }

protected:
    std::vector<SampledPath> sampled_paths_;
    std::unordered_map<topology::GroupId, PathGraph> graphs_;
    std::vector<Conflict> conflicts_;
    std::vector<ConflictCluster> clusters_;
};

}  // namespace pipe_design::conflict
