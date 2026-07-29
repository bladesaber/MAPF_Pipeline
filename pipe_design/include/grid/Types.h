#pragma once

#include <cstdint>
#include <functional>
#include <ostream>

namespace pipe_design::grid {

/// Integer cell coordinate on the search grid.
struct CellCoord {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    CellCoord() = default;
    CellCoord(int32_t x_, int32_t y_, int32_t z_) : x(x_), y(y_), z(z_) {}

    bool operator==(const CellCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    bool operator!=(const CellCoord& other) const { return !(*this == other); }

    CellCoord operator+(const CellCoord& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    CellCoord operator-(const CellCoord& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    friend std::ostream& operator<<(std::ostream& os, const CellCoord& c) {
        return os << '(' << c.x << ", " << c.y << ", " << c.z << ')';
    }
};

}  // namespace pipe_design::grid

namespace std {

template <>
struct hash<pipe_design::grid::CellCoord> {
    std::size_t operator()(const pipe_design::grid::CellCoord& c) const noexcept {
        const std::size_t hx = static_cast<std::size_t>(static_cast<uint32_t>(c.x));
        const std::size_t hy = static_cast<std::size_t>(static_cast<uint32_t>(c.y));
        const std::size_t hz = static_cast<std::size_t>(static_cast<uint32_t>(c.z));
        return hx * 73856093u ^ hy * 19349663u ^ hz * 83492791u;
    }
};

}  // namespace std
