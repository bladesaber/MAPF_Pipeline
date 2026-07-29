#include "planner/low/DirectedAStar.h"

namespace pipe_design::planner {

void DirectedAStar::set_heading_bins(int num_yaw_bins, int num_pitch_bins) {
    num_yaw_bins_ = num_yaw_bins;
    num_pitch_bins_ = num_pitch_bins;
}

void DirectedAStar::expand(DirectedSearchNode* current) {
    successor_buf_.clear();
    actions_->successors(current->coord, current->in_dir, successor_buf_);

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) || !grid_->is_free(succ.cell, pipe_radius_)) {
            continue;
        }

        const Key key = DirectedSearchNode::make_key(succ.cell, succ.heading, num_yaw_bins_, num_pitch_bins_);
        DirectedSearchNode* neighbor = ensure_node(key);
        if (neighbor->status == NodeStatus::Closed) {
            continue;
        }

        const double tentative_g = current->g + succ.cost;
        const bool discovered = neighbor->status == NodeStatus::New;
        if (!discovered && tentative_g >= neighbor->g) {
            continue;
        }

        neighbor->parent = current;
        neighbor->set_arrival(succ.heading, num_yaw_bins_, num_pitch_bins_);
        neighbor->turns_streak = current->next_turns_streak(succ.heading);
        set_node_costs(neighbor, tentative_g, heuristic_at(succ.cell));
        neighbor->status = NodeStatus::Open;
        if (discovered) {
            open_->push(neighbor);
        } else if (!open_->decrease_key(neighbor)) {
            open_->push(neighbor);
        }
    }
}

}  // namespace pipe_design::planner
