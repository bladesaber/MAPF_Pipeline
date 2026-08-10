#include "planner/high/SequenceReplanner.h"

#include <algorithm>
#include <limits>

namespace pipe_design::planner {

std::vector<std::string> SequenceReplanner::replan(
    const std::vector<std::string>& order,
    const std::vector<conflict::Conflict>& conflicts) const {
    const conflict::Conflict* best = nullptr;
    float best_deficit = -std::numeric_limits<float>::infinity();

    for (const conflict::Conflict& c : conflicts) {
        if (c.task_a.empty() || c.task_b.empty()) {
            continue;
        }
        if (std::find(order.begin(), order.end(), c.task_a) == order.end()) {
            continue;
        }
        if (std::find(order.begin(), order.end(), c.task_b) == order.end()) {
            continue;
        }
        if (best == nullptr || c.deficit > best_deficit) {
            best = &c;
            best_deficit = c.deficit;
        }
    }
    if (best == nullptr) {
        return order;
    }

    std::vector<std::string> out = order;
    const auto ia = std::find(out.begin(), out.end(), best->task_a);
    const auto ib = std::find(out.begin(), out.end(), best->task_b);
    std::iter_swap(ia, ib);
    return out;
}

}  // namespace pipe_design::planner
