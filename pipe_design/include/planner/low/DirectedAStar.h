#pragma once

#include "planner/low/AStar.h"

namespace pipe_design::planner {

/// Heading-aware A*: state = (cell, incoming heading bin).
///
/// Shares node map / step / reset with AStar<DirectedSearchNode>; only expand
/// (and optional start-key) differ. Pass ActionModel::set_turn_weight(...) for
/// soft turn costs.
class DirectedAStar : public AStar<DirectedSearchNode> {
public:
    DirectedAStar() = default;

    void set_heading_bins(int num_yaw_bins, int num_pitch_bins = 1);
    int num_yaw_bins() const { return num_yaw_bins_; }
    int num_pitch_bins() const { return num_pitch_bins_; }

protected:
    void expand(DirectedSearchNode* current) override;

private:
    int num_yaw_bins_ = 8;
    int num_pitch_bins_ = 1;
};

}  // namespace pipe_design::planner
