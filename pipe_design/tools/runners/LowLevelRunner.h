#pragma once

#include "formats/SearchInstance.h"
#include "formats/SearchReport.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Heuristic.h"

#include <memory>
#include <vector>

namespace pipe_design::tools {

/// Per-query factory: geometric heuristics ignore `q`; Dijkstra prepares on `q.goals`.
std::unique_ptr<heuristic::Heuristic> make_heuristic(
    const SearchInstance& instance,
    const grid::StaticClearanceGrid& static_grid,
    const SearchQuery& q);

/// Bake instance grid once, run plain A* per query (limits from SearchInstance).
class LowLevelRunner {
public:
    std::vector<SearchReport> run(const SearchInstance& instance) const;
};

}  // namespace pipe_design::tools
