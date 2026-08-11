#include "planner/multi/PrioritySequencer.h"

#include "grid/CompositeGrid.h"

#include <utility>

namespace pipe_design::planner {

const topology::GroupTopologyPlan* PrioritySequencer::group_of(const topology::GroupId& id) const {
    return plans_->find_group(id);
}

const std::vector<grid::CellCoord>& PrioritySequencer::set_of(const topology::TerminalId& id) const {
    return comps_[term_comp_.at(id)];
}

void PrioritySequencer::merge_path(
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

void PrioritySequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    bool stop_on_failure) {
    std::vector<std::string> order;
    for (const topology::Task& t : plans.ordered_tasks()) {
        order.push_back(t.task_id);
    }
    reset(plans, order, grid, low, heuristic, open, actions, stop_on_failure);
}

void PrioritySequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    const std::vector<topology::Task>& past,
    bool stop_on_failure) {
    std::vector<std::string> order;
    for (const topology::Task& t : plans.ordered_tasks()) {
        order.push_back(t.task_id);
    }
    reset(plans, order, grid, low, heuristic, open, actions, past, stop_on_failure);
}

void PrioritySequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const std::vector<std::string>& order,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    bool stop_on_failure) {
    plans_ = &plans;
    static_grid_ = &grid.static_grid();
    base_chain_ = grid.chain();
    low_ = &low;
    heuristic_ = &heuristic;
    open_ = &open;
    actions_ = &actions;
    expansions_ = 0;
    cursor_ = 0;
    last_finished_ = static_cast<std::size_t>(-1);
    stop_on_failure_ = stop_on_failure;
    tasks_.clear();
    term_comp_.clear();
    comps_.clear();
    committed_.clear();

    if (plans_ == nullptr || static_grid_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    std::unordered_map<std::string, topology::Task> by_id;
    for (topology::Task t : plans.enumerate_tasks()) {
        by_id.emplace(t.task_id, std::move(t));
    }

    tasks_.reserve(order.size());
    for (const std::string& id : order) {
        auto it = by_id.find(id);
        if (it == by_id.end()) {
            status_ = SearchStatus::Failure;
            return;
        }
        tasks_.push_back(it->second);
    }

    for (const topology::GroupTopologyPlan& g : plans.groups) {
        for (const topology::Terminal& t : g.terminals) {
            if (term_comp_.count(t.id) != 0) {
                throw std::runtime_error("Terminal ID " + t.id + " already exists");
            }
            const grid::CellCoord c = static_grid_->world_to_cell(t.pos);
            if (!static_grid_->in_bounds(c)) {
                status_ = SearchStatus::Failure;
                return;
            }
            term_comp_[t.id] = comps_.size();
            comps_.push_back({c});
        }
    }

    if (tasks_.empty()) {
        status_ = SearchStatus::Success;
        return;
    }
    status_ = SearchStatus::Running;
}

void PrioritySequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const std::vector<std::string>& order,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    const std::vector<topology::Task>& past,
    bool stop_on_failure) {
    reset(plans, order, grid, low, heuristic, open, actions, stop_on_failure);
    if (status_ != SearchStatus::Running) {
        return;
    }

    // Replay common successful prefix; cursor_ skips re-search.
    const std::size_t n = std::min(tasks_.size(), past.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (tasks_[i].task_id != past[i].task_id || !past[i].succeeded()) {
            break;
        }
        const topology::GroupTopologyPlan* gplan = group_of(tasks_[i].group_id);
        if (gplan == nullptr) {
            break;
        }
        const topology::TopologyEdge* edge = gplan->find_edge(tasks_[i].edge_id);
        if (edge == nullptr || term_comp_.count(edge->u) == 0 || term_comp_.count(edge->v) == 0) {
            break;
        }
        tasks_[i].set_result(SearchStatus::Success, past[i].path);
        merge_path(edge->u, edge->v, tasks_[i].path);
        committed_.push_back(Committed{tasks_[i].group_id, i, edge->radius});
        last_finished_ = i;
        cursor_ = i + 1;
    }

    if (cursor_ >= tasks_.size()) {
        status_ = SearchStatus::Success;
    }
}

SearchStatus PrioritySequencer::step_task() {
    if (cursor_ >= tasks_.size()) {
        status_ = SearchStatus::Success;
        return status_;
    }

    topology::Task& task = tasks_[cursor_];
    const topology::GroupTopologyPlan* gplan = group_of(task.group_id);
    if (gplan == nullptr) {
        task.set_result(SearchStatus::Failure, {});
        status_ = SearchStatus::Failure;
        return status_;
    }
    const topology::TopologyEdge* edge = gplan->find_edge(task.edge_id);
    if (edge == nullptr || term_comp_.count(edge->u) == 0 || term_comp_.count(edge->v) == 0) {
        task.set_result(SearchStatus::Failure, {});
        status_ = SearchStatus::Failure;
        return status_;
    }

    // Foreign groups' committed paths as obstacles on a temporary view — entry grid untouched.
    std::shared_ptr<const grid::ObstacleChain> chain = base_chain_;
    for (const Committed& c : committed_) {
        if (c.group_id == task.group_id) {
            continue;
        }
        chain = chain_with_path(std::move(chain), *static_grid_, tasks_[c.task_index].path, c.radius);
    }
    const grid::CompositeGrid view(*static_grid_, std::move(chain));

    low_->reset(
        view,
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
        ++cursor_;
        if (stop_on_failure_) {
            status_ = SearchStatus::Failure;
            return status_;
        }
        status_ = (cursor_ >= tasks_.size()) ? SearchStatus::Success : SearchStatus::Running;
        return status_;
    }

    task.set_result(SearchStatus::Success, low_->path());
    merge_path(edge->u, edge->v, task.path);
    committed_.push_back(Committed{task.group_id, cursor_, edge->radius});
    last_finished_ = cursor_;
    ++cursor_;

    status_ = (cursor_ >= tasks_.size()) ? SearchStatus::Success : SearchStatus::Running;
    return status_;
}

Path PrioritySequencer::last_path() const {
    if (last_finished_ >= tasks_.size()) {
        return {};
    }
    return tasks_[last_finished_].path;
}

}  // namespace pipe_design::planner
