#pragma once

#include "conflict/ConflictAnalyzer.h"
#include "geom/Types.h"

#include <unordered_map>
#include <vector>

namespace pipe_design::conflict {

/// ConflictAnalyzer that prefers conflicts near **pipe terminals** (not graph degree-1).
///
/// Logic aligned with Python `HeadConflictSelector`:
///   1. Per group, rank conflicts by PathGraph distance to that group's pipe terminals
///   2. `candidates_` = conflicts with at least one side rank == 0
///   3. `select_conflict` returns the candidate with max `score` (`-rank_sum`)
///
/// Ranking is **relative within each group's conflict set** — absolute distance is
/// meaningless across differently sized trees. Use `score_all`, not per-conflict `score`.
///
/// Higher `Conflict::score` is more urgent (`score = -rank_sum`).
class NearTerminalAnalyzer : public ConflictAnalyzer {
public:
    /// `terminals[group_id]` = pipe mouth / design terminal positions for that group.
    explicit NearTerminalAnalyzer(
        std::unordered_map<topology::GroupId, std::vector<geom::Vec3>> terminals);

    void clear() override;

    /// Not supported: near-terminal urgency is a relative rank over the full conflict set.
    [[noreturn]] void score(Conflict& conflict) const override;
    void score(ConflictCluster& cluster) const override;
    void score_all() override;

    const Conflict* select_conflict() const override;

    /// Conflicts with at least one per-group rank == 0 (filled by `score_all`).
    const std::vector<const Conflict*>& candidates() const { return candidates_; }

    const std::unordered_map<topology::GroupId, std::vector<geom::Vec3>>& terminals() const {
        return terminals_;
    }

private:
    std::unordered_map<topology::GroupId, std::vector<geom::Vec3>> terminals_;
    std::vector<const Conflict*> candidates_;
};

}  // namespace pipe_design::conflict
