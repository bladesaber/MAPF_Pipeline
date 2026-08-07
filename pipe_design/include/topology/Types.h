#pragma once

#include "geom/Heading.h"
#include "geom/Types.h"
#include "planner/low/LowLevelPlanner.h"
#include "planner/low/Path.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pipe_design::topology {

using TerminalId = std::string;
using EdgeId = std::string;
using GroupId = std::string;

/// Optional semantic tag for a terminal (does not affect connectivity by itself).
enum class TerminalRole : std::uint8_t {
    Unspecified = 0,
    Inlet,
    Outlet,
    Junction,
};

/// How an edge participates in a radius-spine / MST plan.
enum class EdgeKind : std::uint8_t {
    Unspecified = 0,
    Spine,
    Branch,
    Bridge,
};

/// Endpoint / port in a group's connection graph.
/// `heading` default is none (`Heading{}`); used when Hybrid / directed search cares.
struct Terminal {
    TerminalId id;
    geom::Vec3 pos;
    float radius = 0.f;
    geom::Heading heading;  // none = unspecified approach / leave direction
    TerminalRole role = TerminalRole::Unspecified;

    Terminal() = default;
    Terminal(TerminalId id_, geom::Vec3 pos_, float radius_,
             TerminalRole role_ = TerminalRole::Unspecified,
             geom::Heading heading_ = {})
        : id(std::move(id_))
        , pos(pos_)
        , radius(radius_)
        , heading(heading_)
        , role(role_) {}
};

/// Proposed segment between two terminals (one future low-level Task).
struct TopologyEdge {
    EdgeId id;
    TerminalId u;
    TerminalId v;
    float radius = 0.f;
    int level = 0;  // radius bin / spine level; larger = thicker / earlier
    EdgeKind kind = EdgeKind::Unspecified;

    TopologyEdge() = default;
    TopologyEdge(EdgeId id_, TerminalId u_, TerminalId v_, float radius_,
                 int level_ = 0, EdgeKind kind_ = EdgeKind::Unspecified)
        : id(std::move(id_))
        , u(std::move(u_))
        , v(std::move(v_))
        , radius(radius_)
        , level(level_)
        , kind(kind_) {}
};

/// One TopologyEdge as a schedulable job (+ optional multi-query result).
///
/// Plan export fills identity only (`status == Ready`, empty `path`).
/// Endpoints / radius stay on `TopologyEdge` — look up via `plan.find_edge(edge_id)`.
/// `GroupSession` / `PrioritySequencer` write `status` + `path` after each low-level search.
struct Task {
    std::string task_id;
    GroupId group_id;
    EdgeId edge_id;

    planner::SearchStatus status = planner::SearchStatus::Ready;
    planner::Path path;

    Task() = default;

    Task(std::string task_id_, GroupId group_id_, EdgeId edge_id_)
        : task_id(std::move(task_id_))
        , group_id(std::move(group_id_))
        , edge_id(std::move(edge_id_)) {}

    bool succeeded() const {
        return status == planner::SearchStatus::Success && !path.empty();
    }

    void clear_result() {
        status = planner::SearchStatus::Ready;
        path = {};
    }

    void set_result(planner::SearchStatus st, planner::Path p) {
        status = st;
        path = std::move(p);
    }
};

inline std::string make_task_id(const GroupId& group_id, const EdgeId& edge_id) {
    return group_id + ":" + edge_id;
}

/// One group's `[T]` result (data only; generation lives on `TopologyBuilder`).
///
/// `edges` is the unordered connection set; `grow_order` is an intra-group lay hint.
class GroupTopologyPlan {
public:
    GroupId group_id;
    std::vector<Terminal> terminals;
    std::vector<TopologyEdge> edges;
    std::vector<EdgeId> grow_order;

    GroupTopologyPlan() = default;
    explicit GroupTopologyPlan(GroupId group_id_) : group_id(std::move(group_id_)) {}
    virtual ~GroupTopologyPlan() = default;

    const TopologyEdge* find_edge(const EdgeId& edge_id) const {
        for (const TopologyEdge& e : edges) {
            if (e.id == edge_id) {
                return &e;
            }
        }
        return nullptr;
    }

    const Terminal* find_terminal(const TerminalId& terminal_id) const {
        for (const Terminal& t : terminals) {
            if (t.id == terminal_id) {
                return &t;
            }
        }
        return nullptr;
    }

    /// All jobs in `edges` order (identity only; ignores `grow_order`).
    virtual std::vector<Task> enumerate_tasks() const {
        std::vector<Task> out;
        out.reserve(edges.size());
        std::unordered_set<EdgeId> seen;
        for (const TopologyEdge& e : edges) {
            if (seen.count(e.id) != 0) {
                continue;
            }
            seen.insert(e.id);
            out.emplace_back(make_task_id(group_id, e.id), group_id, e.id);
        }
        return out;
    }

    /// Lay order: `grow_order` first, then any remaining edges (default for GroupSession).
    virtual std::vector<Task> ordered_tasks() const {
        std::vector<Task> out;
        out.reserve(edges.size());
        std::unordered_set<EdgeId> seen;

        auto push_edge = [&](const EdgeId& edge_id) {
            if (seen.count(edge_id) != 0 || find_edge(edge_id) == nullptr) {
                return;
            }
            seen.insert(edge_id);
            out.emplace_back(make_task_id(group_id, edge_id), group_id, edge_id);
        };

        for (const EdgeId& edge_id : grow_order) {
            push_edge(edge_id);
        }
        for (const TopologyEdge& e : edges) {
            push_edge(e.id);
        }
        return out;
    }
};

/// Aggregate of per-group plans (owned by `TopologyBuilder` for CBS / PBS).
class GlobalTopologyPlan {
public:
    std::vector<GroupTopologyPlan> groups;

    GlobalTopologyPlan() = default;
    explicit GlobalTopologyPlan(std::vector<GroupTopologyPlan> groups_)
        : groups(std::move(groups_)) {}
    virtual ~GlobalTopologyPlan() = default;

    const GroupTopologyPlan* find_group(const GroupId& group_id) const {
        for (const GroupTopologyPlan& p : groups) {
            if (p.group_id == group_id) {
                return &p;
            }
        }
        return nullptr;
    }

    /// Union of all groups' `enumerate_tasks()` (group order = `groups` order).
    virtual std::vector<Task> enumerate_tasks() const {
        std::vector<Task> out;
        for (const GroupTopologyPlan& plan : groups) {
            std::vector<Task> part = plan.enumerate_tasks();
            out.insert(out.end(), part.begin(), part.end());
        }
        return out;
    }

    /// Default schedule: concatenate each group's `ordered_tasks()`.
    /// Subclasses may interleave / re-prioritize across groups.
    virtual std::vector<Task> ordered_tasks() const {
        std::vector<Task> out;
        for (const GroupTopologyPlan& plan : groups) {
            std::vector<Task> part = plan.ordered_tasks();
            out.insert(out.end(), part.begin(), part.end());
        }
        return out;
    }
};

}  // namespace pipe_design::topology
