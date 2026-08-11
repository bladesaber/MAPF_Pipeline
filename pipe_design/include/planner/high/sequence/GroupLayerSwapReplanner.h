#pragma once

#include "planner/high/sequence/SequenceReplanner.h"

namespace pipe_design::planner {

/// Failure-driven repair via **group layers** (maximal consecutive same-group runs).
///
/// On the first failed Task in layer L (prev layer P):
///   block = L prefix through the failed Task (inclusive);
///   swap that block with a suffix of P (length 1, then 2, … whole P).
/// Prefers `stop_on_failure` so later Tasks stay Ready / unseen.
/// Example order:  [A1, A2, B1, B2, A3, A4, A5, B3, …]
/// Layers:         [gA={A1,A2}, gB={B1,B2}, gA'={A3,A4,A5}, …]
/// If A4 fails:    block = [A3, A4]  (gA' prefix through failure; A5 stays)
/// Swap with prev-layer suffix (try |suffix|=1 first → B2):
///   → [A1, A2, B1, A3, A4, B2, A5, B3, …]
/// Then try longer suffixes of gB if that order was already tried.
///
/// Defaults to `stop_on_failure` so Tasks after the first failure stay Ready.
class GroupLayerSwapReplanner : public SequenceReplanner {
public:
    GroupLayerSwapReplanner() { stop_on_failure_ = true; }
    ~GroupLayerSwapReplanner() override = default;

    std::vector<std::string> replan(
        const std::vector<std::string>& order,
        const std::vector<topology::Task>& tasks) override;
};

}  // namespace pipe_design::planner
