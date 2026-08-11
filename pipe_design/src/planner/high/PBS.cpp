#include "planner/high/PBS.h"

#include <utility>

namespace pipe_design::planner {

bool PBS::has_constraint(
    const std::vector<PriorityConstraint>& cons,
    const std::string& higher_id,
    const std::string& lower_id) {
    for (const PriorityConstraint& c : cons) {
        if (c.higher_id == higher_id && c.lower_id == lower_id) {
            return true;
        }
    }
    return false;
}

bool PBS::route(PTNode& node) {
    sequencer_->reset(
        *plans_,
        node.constraints,
        *grid_,
        *low_,
        *heuristic_,
        *open_set_,
        *actions_,
        node.past);
    if (sequencer_->search() != SearchStatus::Success || sequencer_->any_failure()) {
        return false;
    }
    node.tasks = sequencer_->tasks();
    node.past = sequencer_->snapshot();
    node.cost = 0.0;
    for (const topology::Task& t : node.tasks) {
        node.cost += t.path.cost;
    }
    return true;
}

void PBS::reset(
    const topology::GlobalTopologyPlan& plans,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    PartialOrderSequencer& sequencer,
    conflict::ConflictAnalyzer& analyzer) {
    plans_ = &plans;
    grid_ = &grid;
    low_ = &low;
    heuristic_ = &heuristic;
    open_set_ = &open;
    actions_ = &actions;
    sequencer_ = &sequencer;
    analyzer_ = &analyzer;

    solution_.clear();
    open_ = std::priority_queue<PTNode, std::vector<PTNode>, PTNodeGreater>(
        PTNodeGreater{search_mode_});
    expansions_ = 0;

    if (plans_ == nullptr || grid_ == nullptr || low_ == nullptr || heuristic_ == nullptr ||
        open_set_ == nullptr || actions_ == nullptr || sequencer_ == nullptr ||
        analyzer_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    PTNode root;
    // Intra-group grow_order as priority edges (higher = earlier).
    for (const topology::GroupTopologyPlan& g : plans_->groups) {
        const std::vector<topology::Task> ordered = g.ordered_tasks();
        for (std::size_t i = 1; i < ordered.size(); ++i) {
            root.constraints.emplace_back(ordered[i - 1].task_id, ordered[i].task_id);
        }
    }

    if (!route(root)) {
        status_ = SearchStatus::Failure;
        return;
    }
    open_.push(std::move(root));
    status_ = SearchStatus::Running;
}

SearchStatus PBS::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_.empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    PTNode node = open_.top();
    open_.pop();
    ++expansions_;

    std::vector<const topology::Task*> refs;
    refs.reserve(node.tasks.size());
    for (const topology::Task& t : node.tasks) {
        refs.push_back(&t);
    }

    analyzer_->set_grid(&grid_->static_grid());
    analyzer_->set_plans(plans_);
    analyzer_->analyze(refs);

    if (analyzer_->conflicts().empty()) {
        solution_ = node.tasks;
        status_ = SearchStatus::Success;
        return status_;
    }

    const conflict::Conflict* conflict = analyzer_->select_conflict();
    if (conflict == nullptr || conflict->task_a.empty() || conflict->task_b.empty()) {
        throw std::runtime_error("PBS: ConflictAnalyzer::select_conflict returned null or empty task ids");
    }

    // Child: task_a ≺ task_b (a plans first).
    if (!has_constraint(node.constraints, conflict->task_a, conflict->task_b)) {
        PTNode child = node;
        child.depth = node.depth + 1;
        child.constraints.emplace_back(conflict->task_a, conflict->task_b);
        if (route(child)) {
            open_.push(std::move(child));
        }
    }

    // Child: task_b ≺ task_a.
    if (!has_constraint(node.constraints, conflict->task_b, conflict->task_a)) {
        PTNode child = node;
        child.depth = node.depth + 1;
        child.constraints.emplace_back(conflict->task_b, conflict->task_a);
        if (route(child)) {
            open_.push(std::move(child));
        }
    }

    return status_;
}

}  // namespace pipe_design::planner
