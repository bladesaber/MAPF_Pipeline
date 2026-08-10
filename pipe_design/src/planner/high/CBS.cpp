#include "planner/high/CBS.h"

#include "geom/Types.h"

#include <stdexcept>
#include <utility>

namespace pipe_design::planner {

void CBS::reset(
    const topology::GlobalTopologyPlan& plans,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    GroupSession& session,
    conflict::ConflictAnalyzer& analyzer) {
    plans_ = &plans;
    grid_ = &grid;
    low_ = &low;
    heuristic_ = &heuristic;
    open_set_ = &open;
    actions_ = &actions;
    session_ = &session;
    analyzer_ = &analyzer;

    solution_.clear();
    while (!open_.empty()) {
        open_.pop();
    }
    expansions_ = 0;

    if (plans_ == nullptr || grid_ == nullptr || low_ == nullptr || heuristic_ == nullptr ||
        open_set_ == nullptr || actions_ == nullptr || session_ == nullptr ||
        analyzer_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    CTNode root;
    for (const topology::GroupTopologyPlan& gp : plans_->groups) {
        if (!route_group(root, gp.group_id)) {
            status_ = SearchStatus::Failure;
            return;
        }
    }
    root.cost = path_cost(root);
    open_.push(std::move(root));
    status_ = SearchStatus::Running;
}

SearchStatus CBS::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_.empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    CTNode node = open_.top();
    open_.pop();
    ++expansions_;

    std::vector<const topology::Task*> refs;
    refs.reserve(node.tasks.size());
    for (const std::shared_ptr<const topology::Task>& t : node.tasks) {
        refs.push_back(t.get());
    }

    analyzer_->set_grid(&grid_->static_grid());
    analyzer_->set_plans(plans_);
    analyzer_->analyze(refs);

    if (analyzer_->conflicts().empty()) {
        solution_.clear();
        solution_.reserve(node.tasks.size());
        for (const std::shared_ptr<const topology::Task>& t : node.tasks) {
            solution_.push_back(*t);
        }
        status_ = SearchStatus::Success;
        return status_;
    }

    const conflict::Conflict* conflict = analyzer_->select_conflict();
    if (conflict == nullptr) {
        throw std::runtime_error(
            "CBS: ConflictAnalyzer::select_conflict returned null with non-empty conflicts");
    }

    // Child: constrain group_a (avoid sample_b). Shares parent's Task ptrs until route_group.
    {
        CTNode child = node;
        child.chains[conflict->group_a] =
            constrain(chain_for(child, conflict->group_a), *conflict, conflict->group_a);
        if (route_group(child, conflict->group_a)) {
            child.cost = path_cost(child);
            open_.push(std::move(child));
        }
    }

    // Child: constrain group_b (avoid sample_a).
    {
        CTNode child = node;
        child.chains[conflict->group_b] =
            constrain(chain_for(child, conflict->group_b), *conflict, conflict->group_b);
        if (route_group(child, conflict->group_b)) {
            child.cost = path_cost(child);
            open_.push(std::move(child));
        }
    }

    return status_;
}

std::shared_ptr<const grid::ObstacleChain> CBS::constrain(
    std::shared_ptr<const grid::ObstacleChain> chain,
    const conflict::Conflict& c,
    const topology::GroupId& group) const {
    if (group == c.group_a) {
        return chain->extend(geom::SphereObstacle{c.sample_b.center, c.sample_b.radius});
    }
    if (group == c.group_b) {
        return chain->extend(geom::SphereObstacle{c.sample_a.center, c.sample_a.radius});
    }
    throw std::runtime_error(
        "CBS::constrain: group '" + group + "' is not involved in the conflict");
}

std::shared_ptr<const grid::ObstacleChain> CBS::chain_for(
    const CTNode& node,
    const topology::GroupId& group) const {
    const auto it = node.chains.find(group);
    if (it != node.chains.end() && it->second) {
        return it->second;
    }
    if (grid_->chain()) {
        return grid_->chain();
    }
    return grid::ObstacleChain::make_root();
}

bool CBS::route_group(CTNode& node, const topology::GroupId& group) {
    const topology::GroupTopologyPlan* plan = plans_->find_group(group);
    if (plan == nullptr) {
        throw std::runtime_error("CBS: unknown group_id '" + group + "'");
    }

    grid::CompositeGrid view(grid_->static_grid(), chain_for(node, group));
    session_->reset(*plan, view, *low_, *heuristic_, *open_set_, *actions_);
    if (session_->search() != SearchStatus::Success) {
        return false;
    }

    std::vector<std::shared_ptr<const topology::Task>> kept;
    kept.reserve(node.tasks.size());
    for (const std::shared_ptr<const topology::Task>& t : node.tasks) {
        if (t->group_id != group) {
            kept.push_back(t);
        }
    }
    std::vector<topology::Task> neu = session_->tasks();
    for (topology::Task& t : neu) {
        kept.push_back(std::make_shared<const topology::Task>(std::move(t)));
    }
    node.tasks = std::move(kept);
    return true;
}

double CBS::path_cost(const CTNode& node) {
    double sum = 0.0;
    for (const std::shared_ptr<const topology::Task>& t : node.tasks) {
        sum += t->path.cost;
    }
    return sum;
}

}  // namespace pipe_design::planner
