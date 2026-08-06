#include "planner/multi/PrioritySequencer.h"

#include "geom/Types.h"
#include "grid/CompositeGrid.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <utility>

namespace pipe_design::planner {

namespace {

std::shared_ptr<const grid::ObstacleChain> chain_with_path(
    std::shared_ptr<const grid::ObstacleChain> chain,
    const grid::StaticClearanceGrid& sg,
    const Path& path,
    float radius) {
    if (path.cells.empty()) {
        return chain;
    }
    if (!chain) {
        chain = grid::ObstacleChain::make_root();
    }

    std::vector<geom::SphereObstacle> spheres;
    std::vector<geom::CapsuleObstacle> capsules;

    if (path.cells.size() == 1) {
        spheres.push_back({sg.cell_center(path.cells[0]), radius});
        return chain->extend(spheres, capsules);
    }

    // Merge maximal constant-direction runs, then:
    //   Chebyshev-adjacent endpoints → sphere (grid neighborhood; any-angle safe)
    //   otherwise → one capsule (covers long / any-angle hops)
    std::size_t i = 0;
    while (i + 1 < path.cells.size()) {
        const grid::CellCoord dir = path.cells[i + 1] - path.cells[i];
        std::size_t j = i + 1;
        while (j + 1 < path.cells.size() &&
               (path.cells[j + 1] - path.cells[j]) == dir) {
            ++j;
        }
        const geom::Vec3 a = sg.cell_center(path.cells[i]);
        const geom::Vec3 b = sg.cell_center(path.cells[j]);
        const grid::CellCoord d = path.cells[j] - path.cells[i];
        const int cheb = std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
        if (cheb == 1) {
            spheres.push_back({b, radius});
        } else {
            capsules.push_back({a, b, radius});
        }
        i = j;
    }

    return chain->extend(spheres, capsules);
}

}  // namespace

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
    const motion::ActionModel& actions) {
    std::vector<std::string> order;
    for (const topology::Task& t : plans.ordered_tasks()) {
        order.push_back(t.task_id);
    }
    reset(plans, order, grid, low, heuristic, open, actions);
}

void PrioritySequencer::reset(
    const topology::GlobalTopologyPlan& plans,
    const std::vector<std::string>& order,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
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

SearchStatus PrioritySequencer::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
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

Path PrioritySequencer::last_path() const {
    if (last_finished_ >= tasks_.size()) {
        return {};
    }
    return tasks_[last_finished_].path;
}

}  // namespace pipe_design::planner
