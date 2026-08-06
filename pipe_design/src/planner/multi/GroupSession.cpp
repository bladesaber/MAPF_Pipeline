#include "planner/multi/GroupSession.h"

#include <utility>

namespace pipe_design::planner {

const std::vector<grid::CellCoord>& GroupSession::set_of(const topology::TerminalId& id) const {
    return comps_[term_comp_.at(id)];
}

void GroupSession::merge_path(
    const topology::TerminalId& u, const topology::TerminalId& v, const Path& path) {
    const std::size_t iu = term_comp_.at(u);
    const std::size_t iv = term_comp_.at(v);
    std::vector<grid::CellCoord> merged;
    const auto& pref = comps_[iu];
    const auto& suff = comps_[iv];
    merged.reserve(pref.size() + path.cells.size() + suff.size());
    merged.insert(merged.end(), pref.begin(), pref.end());
    if (path.cells.size() > 2) {
        merged.insert(merged.end(), path.cells.begin() + 1, path.cells.end() - 1);
    }
    if (iu != iv) {
        merged.insert(merged.end(), suff.begin(), suff.end());
    }
    const std::size_t nid = comps_.size();
    comps_.push_back(std::move(merged));
    for (auto& kv : term_comp_) {
        if (kv.second == iu || kv.second == iv) {
            kv.second = nid;
        }
    }
}

void GroupSession::reset(
    const topology::GroupTopologyPlan& plan,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
    plan_ = &plan;
    grid_ = &grid;
    low_ = &low;
    heuristic_ = &heuristic;
    open_ = &open;
    actions_ = &actions;
    expansions_ = 0;
    cursor_ = 0;
    last_finished_ = static_cast<std::size_t>(-1);
    tasks_ = plan.ordered_tasks();
    term_comp_.clear();
    comps_.clear();

    if (plan_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    comps_.reserve(plan.terminals.size());
    for (const topology::Terminal& t : plan.terminals) {
        const grid::CellCoord c = grid_->world_to_cell(t.pos);
        if (!grid_->in_bounds(c)) {
            status_ = SearchStatus::Failure;
            return;
        }
        term_comp_[t.id] = comps_.size();
        comps_.push_back({c});
    }

    if (tasks_.empty()) {
        status_ = SearchStatus::Success;
        return;
    }
    status_ = SearchStatus::Running;
}

SearchStatus GroupSession::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (cursor_ >= tasks_.size()) {
        status_ = SearchStatus::Success;
        return status_;
    }

    topology::Task& task = tasks_[cursor_];
    const topology::TopologyEdge* edge = plan_->find_edge(task.edge_id);
    if (edge == nullptr || term_comp_.count(edge->u) == 0 || term_comp_.count(edge->v) == 0) {
        task.set_result(SearchStatus::Failure, {});
        status_ = SearchStatus::Failure;
        return status_;
    }

    low_->reset(
        *grid_,
        set_of(edge->u),
        set_of(edge->v),
        edge->radius,
        *heuristic_,
        *open_,
        *actions_);
    low_->search();
    expansions_ += low_->expansions();

    if (low_->status() != SearchStatus::Success) {
        task.set_result(SearchStatus::Failure, {});
        status_ = SearchStatus::Failure;
        return status_;
    }

    task.set_result(SearchStatus::Success, low_->path());
    merge_path(edge->u, edge->v, task.path);
    last_finished_ = cursor_;
    ++cursor_;

    status_ = (cursor_ >= tasks_.size()) ? SearchStatus::Success : SearchStatus::Running;
    return status_;
}

Path GroupSession::last_path() const {
    if (last_finished_ >= tasks_.size()) {
        return {};
    }
    return tasks_[last_finished_].path;
}

}  // namespace pipe_design::planner
