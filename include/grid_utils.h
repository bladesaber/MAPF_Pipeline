//
// Created by admin123456 on 2024/5/29.
//

#ifndef MAPF_PIPELINE_GRID_ENV_H
#define MAPF_PIPELINE_GRID_ENV_H

#include "common.h"
#include "math_utils.h"

using namespace std;

namespace Grid {
    class DiscreteGridEnv {
    public:
        double x_grid_length, y_grid_length, z_grid_length;
        int size_of_x, size_of_y, size_of_z;
        double x_init, y_init, z_init;
        double unit_length;

        DiscreteGridEnv(
                int size_of_x, int size_of_y, int size_of_z,
                double x_init, double y_init, double z_init,
                double x_grid_length, double y_grid_length, double z_grid_length
        ) : size_of_x(size_of_x), size_of_y(size_of_y), size_of_z(size_of_z),
            x_init(x_init), y_init(y_init), z_init(z_init),
            x_grid_length(x_grid_length), y_grid_length(y_grid_length), z_grid_length(z_grid_length) {
            assert(size_of_x <= 1000 & size_of_y <= 1000 & size_of_z <= 1000);
            unit_length = sqrt(pow(x_grid_length, 2.0) + pow(y_grid_length, 2.0) + pow(z_grid_length, 2.0));
        }

        ~DiscreteGridEnv() {};

        // inline函数直接在.h定义
        inline tuple<int, int, int> xyz2grid(double x, double y, double z) const {
            int x_grid = (x - x_init) / x_grid_length;
            int y_grid = (y - y_init) / y_grid_length;
            int z_grid = (z - z_init) / z_grid_length;
            assert(is_valid_grid(x_grid, y_grid, z_grid));
            return make_tuple(x_grid, y_grid, z_grid);
        }

        inline size_t xyz2flag(double x, double y, double z) const {
            int x_grid, y_grid, z_grid;
            tie(x_grid, y_grid, z_grid) = xyz2grid(x, y, z);
            return grid2flag(x_grid, y_grid, z_grid);
        }

        inline size_t grid2flag(int x_grid, int y_grid, int z_grid) const {
            return z_grid * (size_of_x * size_of_y) + y_grid * size_of_x + x_grid;
        }

        inline tuple<int, int, int> flag2grid(size_t loc_flag) const {
            int z_grid = loc_flag / (size_of_x * size_of_y);
            int y_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) / size_of_x;
            int x_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) % size_of_x;
            return make_tuple(x_grid, y_grid, z_grid);
        }

        inline tuple<double, double, double> grid2xyz(int x_grid, int y_grid, int z_grid) const {
            return make_tuple(
                    x_grid * x_grid_length + x_init,
                    y_grid * y_grid_length + y_init,
                    z_grid * z_grid_length + z_init
            );
        }

        inline tuple<double, double, double> flag2xyz(size_t loc_flag) const {
            int x_grid, y_grid, z_grid;
            tie(x_grid, y_grid, z_grid) = flag2grid(loc_flag);
            return grid2xyz(x_grid, y_grid, z_grid);
        }

        bool is_valid_grid(int x, int y, int z) const;

        bool is_valid_flag(size_t loc_flag) const;

        vector<size_t> get_valid_neighbors(
                size_t loc_flag, int step_scale, vector<tuple<int, int, int>> &candidates
        ) const;

        inline double get_manhattan_cost(size_t loc_flag0, size_t loc_flag1) const {
            // 由于grid不一定是正规方体，xyz轴的网格长度可能不相等
            double x0, y0, z0;
            tie(x0, y0, z0) = flag2xyz(loc_flag0);
            double x1, y1, z1;
            tie(x1, y1, z1) = flag2xyz(loc_flag1);
            return compute_manhattan_cost(x0, y0, z0, x1, y1, z1);
        }

        inline double get_euler_cost(size_t loc_flag0, size_t loc_flag1) const {
            double x0, y0, z0;
            tie(x0, y0, z0) = flag2xyz(loc_flag0);
            double x1, y1, z1;
            tie(x1, y1, z1) = flag2xyz(loc_flag1);
            return compute_euler_cost(x0, y0, z0, x1, y1, z1);
        }

        inline double get_curvature_cost(
                int grid_x0, int grid_y0, int grid_z0,
                int grid_x1, int grid_y1, int grid_z1,
                int grid_x2, int grid_y2, int grid_z2,
                double ref_radius, double weight, bool is_left
        ) const {
            double x0, y0, z0, x1, y1, z1, x2, y2, z2;
            tie(x0, y0, z0) = grid2xyz(grid_x0, grid_y0, grid_z0);
            tie(x1, y1, z1) = grid2xyz(grid_x1, grid_y1, grid_z1);
            tie(x2, y2, z2) = grid2xyz(grid_x2, grid_y2, grid_z2);

            double radius = curvature_radius_3point(x0, y0, z0, x1, y1, z1, x2, y2, z2, is_left);
            if (radius >= ref_radius) {
                return 0.0;
            }
            return (1.0 - radius / ref_radius) * weight;
        }
    };
}

#endif //MAPF_PIPELINE_GRID_ENV_H
