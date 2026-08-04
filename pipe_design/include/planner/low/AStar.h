#pragma once

#include "geom/Heading.h"
#include "grid/CompositeGrid.h"
#include "grid/Types.h"
#include "heuristic/Heuristic.h"
#include "planner/low/LowLevelPlanner.h"
#include "planner/low/Path.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/open/OpenSet.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pipe_design::planner {

/// Templated A*: `Node` is SearchNode or DirectedSearchNode (must derive SearchNode).
/// Map keyed by `Node::Key`. One expansion per step.
///
/// Does not own grid, heuristic, open set, or action model.
/// For heading-aware search, use DirectedAStar (overrides expand).
template <typename Node = SearchNode>
class AStar : public LowLevelPlanner {
public:
    using Key = typename Node::Key;

    AStar() = default;
    ~AStar() override = default;

    void reset(
        const grid::CompositeGrid& grid,
        const std::vector<grid::CellCoord>& starts,
        const std::vector<grid::CellCoord>& goals,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    void reset(
        const grid::CompositeGrid& grid,
        grid::CellCoord start,
        grid::CellCoord goal,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    SearchStatus step() override;
    Path path() const override;

    const Node* last_expanded() const { return last_expanded_; }
    const Node* goal_node() const { return goal_node_; }

    bool is_goal(const grid::CellCoord& cell) const;

    const std::unordered_map<Key, std::unique_ptr<Node>>& nodes() const { return nodes_; }

    Node* find_node(const Key& key);
    const Node* find_node(const Key& key) const;

    void set_heuristic_weight(double weight);
    double heuristic_weight() const { return heuristic_weight_; }

protected:
    /// Open-set priority key. Override for custom mixes; default is g + w*h.
    virtual double combine_key(double g, double h) const;
    virtual void before_expand(Node* /*current*/) {}
    virtual void expand(Node* current);

    Node* ensure_node(const Key& key);
    double heuristic_at(const grid::CellCoord& from) const;
    void set_node_costs(Node* node, double g, double h) const;
    Path reconstruct(const Node* goal) const;

    const grid::CompositeGrid* grid_ = nullptr;
    float pipe_radius_ = 0.f;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;
    double heuristic_weight_ = 1.0;

    std::unordered_set<grid::CellCoord> goals_;
    std::unordered_map<Key, std::unique_ptr<Node>> nodes_;
    std::vector<motion::Successor> successor_buf_;

    Node* last_expanded_ = nullptr;
    Node* goal_node_ = nullptr;
};

}  // namespace pipe_design::planner
