//
// Created by admin123456 on 2024/6/7.
//

#include "grid_utils.h"

bool Grid::DiscreteGridEnv::is_valid_flag(size_t loc_flag) const {
    int x, y, z;
    tie(x, y, z) = flag2grid(loc_flag);
    return is_valid_grid(x, y, z);
}

bool Grid::DiscreteGridEnv::is_valid_grid(int x, int y, int z) const {
    // 不使用<=或>=的原因在于搜索空间外轮廓不一定为长方体
    if (x < 0 || x > size_of_x - 1) { return false; }
    if (y < 0 || y > size_of_y - 1) { return false; }
    if (z < 0 || z > size_of_z - 1) { return false; }
    return true;
}

vector<size_t> Grid::DiscreteGridEnv::get_valid_neighbors(
        size_t loc_flag, int step_scale, vector<tuple<int, int, int>> &candidates
) const {
    int x, y, z;
    tie(x, y, z) = flag2grid(loc_flag);

    vector<size_t> neighbors;
    int x_num, y_num, z_num;
    int x_, y_, z_;
    for (auto next: candidates) {
        tie(x_num, y_num, z_num) = next;
        x_ = x + x_num * step_scale;
        y_ = y + y_num * step_scale;
        z_ = z + z_num * step_scale;
        if (is_valid_grid(x_, y_, z_)) {
            neighbors.emplace_back(grid2flag(x_, y_, z_));
        }
    }

    return neighbors;
}