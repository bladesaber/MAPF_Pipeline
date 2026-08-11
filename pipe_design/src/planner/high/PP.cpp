#include "planner/high/PP.h"

#include <limits>
#include <utility>

namespace pipe_design::planner {

void PP::reset(
    const topology::GlobalTopologyPlan& plans,
    const grid::CompositeGrid& grid,
    AStar<SearchNode>& low,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions,
    PrioritySequencer& sequencer,
    SequenceReplanner& replanner) {
    plans_ = &plans;
    grid_ = &grid;
    low_ = &low;
    heuristic_ = &heuristic;
    open_set_ = &open;
    actions_ = &actions;
    sequencer_ = &sequencer;
    replanner_ = &replanner;

    order_.clear();
    tasks_.clear();
    solution_.clear();
    expansions_ = 0;
    depth_ = 0;
    cost_ = 0.0;

    if (plans_ == nullptr || grid_ == nullptr || low_ == nullptr || heuristic_ == nullptr ||
        open_set_ == nullptr || actions_ == nullptr || sequencer_ == nullptr ||
        replanner_ == nullptr) {
        status_ = SearchStatus::Failure;
        return;
    }

    for (const topology::Task& t : plans_->ordered_tasks()) {
        order_.push_back(t.task_id);
    }
    route();
    status_ = SearchStatus::Running;
}

SearchStatus PP::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }

    ++expansions_;

    if (!sequencer_->any_failure()) {
        solution_ = tasks_;
        status_ = SearchStatus::Success;
        return status_;
    }

    std::vector<std::string> next = replanner_->replan(order_, tasks_);
    if (next.empty() || replanner_->status() == ReplanStatus::Failure) {
        status_ = SearchStatus::Failure;
        return status_;
    }
    order_ = std::move(next);

    route();
    ++depth_;
    return status_;
}

void PP::route() {
    sequencer_->reset(
        *plans_,
        order_,
        *grid_,
        *low_,
        *heuristic_,
        *open_set_,
        *actions_,
        tasks_,
        replanner_->stop_on_failure());
    sequencer_->search();
    tasks_ = sequencer_->tasks();
    cost_ = 0.0;
    for (const topology::Task& t : tasks_) {
        if (!t.succeeded()) {
            cost_ = std::numeric_limits<double>::infinity();
            return;
        }
        cost_ += t.path.cost;
    }
}

}  // namespace pipe_design::planner
