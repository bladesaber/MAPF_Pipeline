#include "planner/high/sequence/GroupLayerSwapReplanner.h"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pipe_design::planner {

namespace {

/// One maximal run of consecutive Tasks from the same group.
struct GroupLayer {
    topology::GroupId group_id;
    std::size_t begin = 0;  // inclusive index in order
    std::size_t end = 0;    // exclusive
};

std::vector<GroupLayer> build_layers(
    const std::vector<std::string>& order,
    const std::unordered_map<std::string, topology::GroupId>& group_of) {
    std::vector<GroupLayer> layers;
    for (std::size_t i = 0; i < order.size();) {
        const auto git = group_of.find(order[i]);
        if (git == group_of.end()) {
            return {};
        }
        GroupLayer layer;
        layer.group_id = git->second;
        layer.begin = i;
        ++i;
        while (i < order.size()) {
            const auto jt = group_of.find(order[i]);
            if (jt == group_of.end() || jt->second != layer.group_id) {
                break;
            }
            ++i;
        }
        layer.end = i;
        layers.push_back(std::move(layer));
    }
    return layers;
}

/// Swap adjacent segments [suf_begin, suf_end) and [fail_begin, fail_end).
std::vector<std::string> swap_adjacent(
    const std::vector<std::string>& order,
    std::size_t suf_begin,
    std::size_t suf_end,
    std::size_t fail_begin,
    std::size_t fail_end) {
    std::vector<std::string> out;
    out.reserve(order.size());
    out.insert(out.end(), order.begin(), order.begin() + static_cast<std::ptrdiff_t>(suf_begin));
    out.insert(out.end(),
               order.begin() + static_cast<std::ptrdiff_t>(fail_begin),
               order.begin() + static_cast<std::ptrdiff_t>(fail_end));
    out.insert(out.end(),
               order.begin() + static_cast<std::ptrdiff_t>(suf_begin),
               order.begin() + static_cast<std::ptrdiff_t>(suf_end));
    out.insert(out.end(),
               order.begin() + static_cast<std::ptrdiff_t>(fail_end),
               order.end());
    return out;
}

}  // namespace

std::vector<std::string> GroupLayerSwapReplanner::replan(
    const std::vector<std::string>& order,
    const std::vector<topology::Task>& tasks) {
    order_ = order;
    status_ = ReplanStatus::Running;
    history_.push_back(make_plan_record(order, tasks));

    // Map task_id → group / status for layering and first-failure lookup.
    std::unordered_map<std::string, topology::GroupId> group_of;
    std::unordered_map<std::string, SearchStatus> status_of;
    group_of.reserve(tasks.size());
    status_of.reserve(tasks.size());
    for (const topology::Task& t : tasks) {
        group_of[t.task_id] = t.group_id;
        status_of[t.task_id] = t.status;
    }

    const std::vector<GroupLayer> layers = build_layers(order, group_of);
    if (layers.empty()) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    // First failed Task in schedule order (expects stop_on_failure for a clean cut).
    std::size_t fail_index = order.size();
    for (std::size_t i = 0; i < order.size(); ++i) {
        const auto it = status_of.find(order[i]);
        if (it != status_of.end() && it->second == SearchStatus::Failure) {
            fail_index = i;
            break;
        }
    }
    if (fail_index >= order.size()) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    std::size_t layer_i = layers.size();
    for (std::size_t li = 0; li < layers.size(); ++li) {
        if (fail_index >= layers[li].begin && fail_index < layers[li].end) {
            layer_i = li;
            break;
        }
    }
    // Need a previous foreign layer to swap with.
    if (layer_i == 0 || layer_i >= layers.size()) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    const GroupLayer& failed = layers[layer_i];
    const GroupLayer& prev = layers[layer_i - 1];
    if (failed.group_id == prev.group_id) {
        status_ = ReplanStatus::Failure;
        return {};
    }

    // Failed block = [layer start .. failed Task] (inclusive). Leave later Tasks in the
    // layer (e.g. A5) where they are.
    const std::size_t fail_begin = failed.begin;
    const std::size_t fail_end = fail_index + 1;

    // Escalate: swap with prev-layer suffix of length 1, then 2, … whole prev layer.
    const std::size_t prev_len = prev.end - prev.begin;
    for (std::size_t k = 1; k <= prev_len; ++k) {
        const std::size_t suf_begin = prev.end - k;
        const std::size_t suf_end = prev.end;
        std::vector<std::string> next =
            swap_adjacent(order, suf_begin, suf_end, fail_begin, fail_end);
        if (next == order || order_used(next)) {
            continue;
        }
        order_ = next;
        status_ = ReplanStatus::Success;
        return next;
    }

    status_ = ReplanStatus::Failure;
    return {};
}

}  // namespace pipe_design::planner
