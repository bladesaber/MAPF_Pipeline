#include "planner/multi/PartialOrderSequencer.h"
#include "planner/multi/PrioritySequencer.h"

#include "grid/CompositeGrid.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <utility>

namespace pipe_design::planner {

std::vector<std::string> PartialOrderSequencer::topological_sort(
    const std::vector<std::string>& seed,
    const std::vector<PriorityConstraint>& constraints) {
    std::unordered_set<std::string> known(seed.begin(), seed.end());
    if (known.size() != seed.size()) {
        throw std::runtime_error("topological_sort: seed contains duplicates");
    }

    std::unordered_map<std::string, std::vector<std::string>> succ;
    std::unordered_map<std::string, int> indeg;
    for (const std::string& id : seed) {
        indeg[id] = 0;
    }
    for (const PriorityConstraint& c : constraints) {
        if (known.count(c.higher_id) == 0 || known.count(c.lower_id) == 0 ||
            c.higher_id == c.lower_id) {
            throw std::runtime_error("topological_sort: invalid constraint");
        }
        succ[c.higher_id].push_back(c.lower_id);
        indeg[c.lower_id] += 1;
    }

    // Kahn (same as classic queue form): seed order among initial indegree-0 nodes.
    std::deque<std::string> queue;
    for (const std::string& id : seed) {
        if (indeg[id] == 0) {
            queue.push_back(id);
        }
    }

    std::vector<std::string> out;
    out.reserve(seed.size());
    while (!queue.empty()) {
        const std::string u = queue.front();
        queue.pop_front();
        out.push_back(u);
        for (const std::string& v : succ[u]) {
            if (--indeg[v] == 0) {
                queue.push_back(v);
            }
        }
    }
    return out;
}

void PartialOrderSequencer::rebuild_higher_graph() {
    higher_adj_.clear();
    for (const PriorityConstraint& c : constraints_) {
        higher_adj_[c.lower_id].push_back(c.higher_id);
    }
}

std::unordered_set<std::string> PartialOrderSequencer::higher_of(
    const std::string& task_id) const {
    std::unordered_set<std::string> higher;
    std::vector<std::string> stack;
    stack.push_back(task_id);
    std::unordered_set<std::string> seen{task_id};
    while (!stack.empty()) {
        const std::string u = stack.back();
        stack.pop_back();
        const auto it = higher_adj_.find(u);
        if (it == higher_adj_.end()) {
            continue;
        }
        for (const std::string& h : it->second) {
            if (seen.insert(h).second) {
                higher.insert(h);
                stack.push_back(h);
            }
        }
    }
    return higher;
}

const topology::GroupTopologyPlan* PartialOrderSequencer::group_of(
    const topology::GroupId& id) const {
    return plans_->find_group(id);
}

const std::vector<grid::CellCoord>& PartialOrderSequencer::set_of(
    const topology::TerminalId& id) const {
    return comps_[term_comp_.at(id)];
}

void PartialOrderSequencer::merge_path(
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

void PartialOrderSequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const std::vector<PriorityConstraint>& constraints,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    const PartialOrderPast& past) {
    plans_ = &plans;
    constraints_ = constraints;
    static_grid_ = &grid.static_grid();
    base_chain_ = grid.chain();
    low_ = &low;
    heuristic_ = &heuristic;
    open_ = &open;
    actions_ = &actions;
    expansions_ = 0;
    cursor_ = 0;
    last_finished_ = static_cast<std::size_t>(-1);
    tasks_.clear();
    order_.clear();
    term_comp_.clear();
    comps_.clear();
    committed_.clear();
    higher_adj_.clear();
    replanned_ids_.clear();
    past_ = past;

    if (plans_ == nullptr || static_grid_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    std::vector<std::string> seed;
    std::unordered_map<std::string, topology::Task> by_id;
    for (topology::Task t : plans.enumerate_tasks()) {
        seed.push_back(t.task_id);
        by_id.emplace(t.task_id, std::move(t));
    }

    order_ = topological_sort(seed, constraints_);
    if (order_.size() != seed.size()) {
        status_ = SearchStatus::Failure;
        return;
    }
    rebuild_higher_graph();

    tasks_.reserve(order_.size());
    for (const std::string& id : order_) {
        tasks_.push_back(by_id.at(id));
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

void PartialOrderSequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const std::vector<PriorityConstraint>& constraints,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
    reset(plans, constraints, grid, low, heuristic, open, actions, PartialOrderPast{});
}

SearchStatus PartialOrderSequencer::step_task() {
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

    if (can_reuse(task.task_id, past_)) {
        task.set_result(SearchStatus::Success, past_.by_id.at(task.task_id).task.path);
        merge_path(edge->u, edge->v, task.path);
        committed_.push_back(Committed{task.group_id, cursor_, edge->radius});
        last_finished_ = cursor_;
        ++cursor_;
        status_ = (cursor_ >= tasks_.size()) ? SearchStatus::Success : SearchStatus::Running;
        return status_;
    }

    // Obstacles: succeeded Tasks in higher_of(current) that are foreign groups.
    const std::unordered_set<std::string> higher = higher_of(task.task_id);
    std::shared_ptr<const grid::ObstacleChain> chain = base_chain_;
    for (const Committed& c : committed_) {
        if (c.group_id == task.group_id) {
            continue;
        }
        if (higher.count(tasks_[c.task_index].task_id) == 0) {
            continue;
        }
        chain = chain_with_path(std::move(chain), *static_grid_, tasks_[c.task_index].path, c.radius);
    }
    const grid::CompositeGrid view(*static_grid_, std::move(chain));

    replanned_ids_.push_back(task.task_id);
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
        status_ = SearchStatus::Failure;
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

Path PartialOrderSequencer::last_path() const {
    if (last_finished_ >= tasks_.size()) {
        return {};
    }
    return tasks_[last_finished_].path;
}

PartialOrderPast PartialOrderSequencer::snapshot() const {
    std::unordered_map<std::string, topology::GroupId> group_of_task;
    group_of_task.reserve(tasks_.size());
    for (const topology::Task& t : tasks_) {
        group_of_task.emplace(t.task_id, t.group_id);
    }

    PartialOrderPast past;
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
        const topology::Task& task = tasks_[i];
        PartialOrderPastEntry entry;
        entry.task = task;
        for (const std::string& h : higher_of(task.task_id)) {
            const auto it = group_of_task.find(h);
            if (it != group_of_task.end() && it->second != task.group_id) {
                entry.foreign_higher.insert(h);
            }
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (tasks_[j].group_id == task.group_id) {
                entry.same_group_prev.insert(tasks_[j].task_id);
            }
        }
        past.by_id.emplace(task.task_id, std::move(entry));
    }
    return past;
}

bool PartialOrderSequencer::can_reuse(
    const std::string& task_id, const PartialOrderPast& past) const {
    const auto pit = past.by_id.find(task_id);
    if (pit == past.by_id.end() || !pit->second.task.succeeded()) {
        return false;
    }

    std::unordered_map<std::string, topology::GroupId> group_of_task;
    group_of_task.reserve(tasks_.size());
    for (const topology::Task& t : tasks_) {
        group_of_task.emplace(t.task_id, t.group_id);
    }
    const auto git = group_of_task.find(task_id);
    if (git == group_of_task.end()) {
        return false;
    }
    const topology::GroupId& group_id = git->second;

    std::unordered_set<std::string> foreign_higher;
    for (const std::string& h : higher_of(task_id)) {
        const auto it = group_of_task.find(h);
        if (it != group_of_task.end() && it->second != group_id) {
            foreign_higher.insert(h);
        }
    }

    std::unordered_set<std::string> same_group_prev;
    for (const topology::Task& t : tasks_) {
        if (t.task_id == task_id) {
            break;
        }
        if (t.group_id == group_id) {
            same_group_prev.insert(t.task_id);
        }
    }

    // 1–2: context sets unchanged
    if (foreign_higher != pit->second.foreign_higher ||
        same_group_prev != pit->second.same_group_prev) {
        return false;
    }

    // 3–4: no donor replanned this session (path treated as changed)
    auto any_replanned = [&](const std::unordered_set<std::string>& ids) {
        for (const std::string& id : ids) {
            if (std::find(replanned_ids_.begin(), replanned_ids_.end(), id) !=
                replanned_ids_.end()) {
                return true;
            }
        }
        return false;
    };
    if (any_replanned(foreign_higher) || any_replanned(same_group_prev)) {
        return false;
    }
    return true;
}

}  // namespace pipe_design::planner
