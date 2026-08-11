#include "planner/high/sequence/SequenceReplanner.h"

#include <unordered_map>

namespace pipe_design::planner {

void SequenceReplanner::clear() {
    order_.clear();
    history_.clear();
    status_ = ReplanStatus::Ready;
    // keep stop_on_failure_ (policy preference)
}

bool SequenceReplanner::order_used(const std::vector<std::string>& order) const {
    for (const PlanRecord& record : history_) {
        if (record.order == order) {
            return true;
        }
    }
    return false;
}

PlanRecord SequenceReplanner::make_plan_record(
    const std::vector<std::string>& order,
    const std::vector<topology::Task>& tasks) {
    std::unordered_map<std::string, SearchStatus> status_of;
    status_of.reserve(tasks.size());
    for (const topology::Task& t : tasks) {
        status_of[t.task_id] = t.status;
    }

    PlanRecord record;
    record.order = order;
    record.statuses.reserve(order.size());
    for (const std::string& id : order) {
        const auto it = status_of.find(id);
        record.statuses.push_back(
            it != status_of.end() ? it->second : SearchStatus::Ready);
    }
    return record;
}

}  // namespace pipe_design::planner
