#pragma once

#include <cstdint>
#include <ostream>

namespace pipe::grid {

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

}  // namespace pipe::grid
