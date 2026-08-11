#include "planner/high/sequence/FailureBlockReplanner.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace pipe_design::planner {

namespace {

std::vector<std::string> move_group_before(
    const std::vector<std::string>& order,
    const topology::GroupId& late_group,
    const std::string& early_task,
    const std::unordered_map<std::string, topology::GroupId>& group_of) {
    std::vector<std::string> late_tasks;
    std::vector<std::string> rest;
    late_tasks.reserve(order.size());
    rest.reserve(order.size());
    for (const std::string& id : order) {
        const auto it = group_of.find(id);
        if (it != group_of.end() && it->second == late_group) {
            late_tasks.push_back(id);
        } else {
            rest.push_back(id);
        }
    }
    if (late_tasks.empty()) {
        return order;
    }

    const auto insert_at = std::find(rest.begin(), rest.end(), early_task);
    if (insert_at == rest.end()) {
        return order;
    }

    std::vector<std::string> out;
    out.reserve(order.size());
    out.insert(out.end(), rest.begin(), insert_at);
    out.insert(out.end(), late_tasks.begin(), late_tasks.end());
    out.insert(out.end(), insert_at, rest.end());
    return out;
}

}  // namespace

std::vector<std::string> FailureBlockReplanner::replan(
    const std::vector<std::string>& order,
    const std::vector<topology::Task>& tasks) {
    order_ = order;
    status_ = ReplanStatus::Running;
    history_.push_back(make_plan_record(order, tasks));

    std::unordered_map<std::string, const topology::Task*> by_id;
    std::unordered_map<std::string, topology::GroupId> group_of;
    by_id.reserve(tasks.size());
    group_of.reserve(tasks.size());
    for (const topology::Task& t : tasks) {
        by_id[t.task_id] = &t;
        group_of[t.task_id] = t.group_id;
    }

    std::string early_task;
    std::string late_task;

    for (const std::string& id : order) {
        const auto it = by_id.find(id);
        if (it == by_id.end() || it->second->status != SearchStatus::Failure) {
            continue;
        }
        late_task = id;
        for (const std::string& prev : order) {
            if (prev == id) {
                break;
            }
            const auto pit = group_of.find(prev);
            if (pit != group_of.end() && pit->second != group_of[id]) {
                early_task = prev;
            }
        }
        break;
    }

    if (early_task.empty() || late_task.empty()) {
        status_ = ReplanStatus::Failure;
        return {};
    }
    if (group_of[early_task] == group_of[late_task]) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    std::vector<std::string> next =
        move_group_before(order, group_of[late_task], early_task, group_of);
    if (next == order || order_used(next)) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    order_ = next;
    status_ = ReplanStatus::Success;
    return next;
}

}  // namespace pipe_design::planner
