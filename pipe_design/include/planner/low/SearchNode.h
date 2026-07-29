#pragma once

#include "geom/Heading.h"
#include "grid/Types.h"

#include <cstdint>
#include <functional>

namespace pipe_design::planner {

enum class NodeStatus {
    New,      // not yet discovered
    Open,     // in the open set
    Closed,   // already expanded
};

/// Heading along the grid step from `from` → `to` (normalized). Same cell → none.
inline geom::Heading heading_between(const grid::CellCoord& from, const grid::CellCoord& to) {
    return geom::Heading::from_grid_step(to.x - from.x, to.y - from.y, to.z - from.z);
}

/// Closed-set / map identity for plain grid A* (cell only).
struct CellStateKey {
    grid::CellCoord coord{};

    CellStateKey() = default;
    explicit CellStateKey(grid::CellCoord c) : coord(c) {}

    bool operator==(const CellStateKey& o) const { return coord == o.coord; }
    bool operator!=(const CellStateKey& o) const { return !(*this == o); }
};

/// Closed-set identity for heading-aware search (cell + quantized incoming heading).
struct DirectedStateKey {
    static constexpr int kNoneHeadingBin = -1;

    grid::CellCoord coord{};
    int heading_bin = kNoneHeadingBin;

    DirectedStateKey() = default;
    DirectedStateKey(grid::CellCoord c, int bin) : coord(c), heading_bin(bin) {}

    bool operator==(const DirectedStateKey& o) const {
        return coord == o.coord && heading_bin == o.heading_bin;
    }
    bool operator!=(const DirectedStateKey& o) const { return !(*this == o); }

    static int bin_of(
        const geom::Heading& heading,
        int num_yaw_bins,
        int num_pitch_bins = 1) {
        if (heading.is_none()) {
            return kNoneHeadingBin;
        }
        return heading.bin(num_yaw_bins, num_pitch_bins);
    }
};

/// Cell-only search node (used by AStar).
struct SearchNode {
    using Key = CellStateKey;

    grid::CellCoord coord;
    double g = 0.0;
    double h = 0.0;
    double f = 0.0;
    SearchNode* parent = nullptr;
    NodeStatus status = NodeStatus::New;

    SearchNode() = default;
    explicit SearchNode(grid::CellCoord coord_) : coord(coord_) {}

    Key key() const { return Key{coord}; }

    static Key make_key(const grid::CellCoord& cell) { return Key{cell}; }

    static SearchNode from_key(const Key& key) { return SearchNode{key.coord}; }
};

/// Heading-aware node for DirectedAStar: state ≈ (cell, incoming heading).
struct DirectedSearchNode : SearchNode {
    using Key = DirectedStateKey;

    geom::Heading in_dir{};
    int heading_bin = DirectedStateKey::kNoneHeadingBin;

    /// Consecutive turns ending at this node (0 = straight arrival or start).
    int turns_streak = 0;

    DirectedSearchNode() = default;
    explicit DirectedSearchNode(grid::CellCoord coord_) : SearchNode(coord_) {}
    DirectedSearchNode(
        grid::CellCoord coord_,
        geom::Heading in_dir_,
        int heading_bin_ = DirectedStateKey::kNoneHeadingBin)
        : SearchNode(coord_), in_dir(in_dir_), heading_bin(heading_bin_) {}

    Key key() const { return Key{coord, heading_bin}; }

    /// Start / cell-only key (no incoming heading).
    static Key make_key(const grid::CellCoord& cell) {
        return Key{cell, DirectedStateKey::kNoneHeadingBin};
    }

    static Key make_key(
        const grid::CellCoord& cell,
        const geom::Heading& heading,
        int num_yaw_bins,
        int num_pitch_bins = 1) {
        return Key{cell, DirectedStateKey::bin_of(heading, num_yaw_bins, num_pitch_bins)};
    }

    static DirectedSearchNode from_key(const Key& key) {
        DirectedSearchNode node{key.coord};
        node.heading_bin = key.heading_bin;
        return node;
    }

    void set_arrival(
        const geom::Heading& heading,
        int num_yaw_bins,
        int num_pitch_bins = 1) {
        in_dir = heading;
        heading_bin = DirectedStateKey::bin_of(heading, num_yaw_bins, num_pitch_bins);
    }

    const DirectedSearchNode* directed_parent() const {
        return static_cast<const DirectedSearchNode*>(parent);
    }
    DirectedSearchNode* directed_parent() {
        return static_cast<DirectedSearchNode*>(parent);
    }

    bool is_straight(const geom::Heading& next_dir) const {
        return in_dir.is_none() || in_dir.nearly_equal(next_dir);
    }

    float heading_alignment(const geom::Heading& next_dir) const {
        return in_dir.alignment(next_dir);
    }

    int next_turns_streak(const geom::Heading& next_dir) const {
        if (is_straight(next_dir)) {
            return 0;
        }
        return turns_streak + 1;
    }
};

}  // namespace pipe_design::planner

namespace std {

template <>
struct hash<pipe_design::planner::CellStateKey> {
    std::size_t operator()(const pipe_design::planner::CellStateKey& k) const noexcept {
        return hash<pipe_design::grid::CellCoord>{}(k.coord);
    }
};

template <>
struct hash<pipe_design::planner::DirectedStateKey> {
    std::size_t operator()(const pipe_design::planner::DirectedStateKey& k) const noexcept {
        const std::size_t h = hash<pipe_design::grid::CellCoord>{}(k.coord);
        return h ^ (static_cast<std::size_t>(static_cast<uint32_t>(k.heading_bin)) * 2654435761u);
    }
};

}  // namespace std
