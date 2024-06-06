//
// Created by admin123456 on 2024/5/29.
//

#ifndef MAPF_PIPELINE_GRID_ENV_H
#define MAPF_PIPELINE_GRID_ENV_H

#include "common.h"
#include "math_utils.h"

using namespace std;

namespace Grid {
    list<tuple<int, int, int>> candidate_1D{
            tuple<int, int, int>(1, 0, 0),
            tuple<int, int, int>(-1, 0, 0),
            tuple<int, int, int>(0, 1, 0),
            tuple<int, int, int>(0, -1, 0),
            tuple<int, int, int>(0, 0, 1),
            tuple<int, int, int>(0, 0, -1),
    };

    list<tuple<int, int, int>> candidate_2D{
            tuple<int, int, int>(1, 1, 0),
            tuple<int, int, int>(1, -1, 0),
            tuple<int, int, int>(-1, 1, 0),
            tuple<int, int, int>(-1, -1, 0),
            tuple<int, int, int>(0, 1, 1),
            tuple<int, int, int>(0, 1, -1),
            tuple<int, int, int>(0, -1, 1),
            tuple<int, int, int>(0, -1, -1),
            tuple<int, int, int>(1, 0, 1),
            tuple<int, int, int>(1, 0, -1),
            tuple<int, int, int>(-1, 0, 1),
            tuple<int, int, int>(-1, 0, -1),
    };

    list<tuple<int, int, int>> candidate_3D{
            tuple<int, int, int>(1, 1, 1),
            tuple<int, int, int>(1, -1, 1),
            tuple<int, int, int>(1, 1, -1),
            tuple<int, int, int>(1, -1, -1),
            tuple<int, int, int>(-1, 1, 1),
            tuple<int, int, int>(-1, -1, 1),
            tuple<int, int, int>(-1, 1, -1),
            tuple<int, int, int>(-1, -1, -1),
    };


    class ContinueGridEnv {
        /*
        每个搜索搭配1个Env，各个Env具有不同step_length
         */
    public:
        double xmin, xmax, ymin, ymax, zmin, zmax;
        double x_step, y_step, z_step;
        double x_init, y_init, z_init;
        int size_of_x, size_of_y, size_of_z;
        int x_shift, y_shift, z_shift;

        ContinueGridEnv(
                double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
                double x_init, double y_init, double z_init,
                double x_step, double y_step, double z_step
        ) : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
            x_init(x_init), y_init(y_init), z_init(z_init), x_step(x_step), y_step(y_step), z_step(z_step) {

            size_of_x = floor((xmax - xmin) / x_step);
            size_of_y = floor((ymax - ymin) / y_step);
            size_of_z = floor((zmax - zmin) / z_step);
            assert(size_of_x <= 1000 & size_of_y <= 1000 & size_of_z <= 1000);

            x_shift = floor((x_init - xmin) / x_step);
            y_shift = floor((y_init - ymin) / y_step);
            z_shift = floor((z_init - zmin) / z_step);
        }

        ~ContinueGridEnv() {}

        inline size_t xyz2flag(double x, double y, double z) const {
            // 这里误差距离为 step * 0.5
            int x_grid = (x - x_init) / x_step + x_shift;
            int y_grid = (y - y_init) / y_step + y_shift;
            int z_grid = (z - z_init) / z_step + z_shift;
            return z_grid * (size_of_x * size_of_y) + y_grid * size_of_x + x_grid;
        }

        inline tuple<int, int, int> flag2grid(size_t loc_flag) const {
            int z_grid = loc_flag / (size_of_x * size_of_y) - z_shift;
            int y_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) / size_of_x - y_shift;
            int x_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) % size_of_x - x_shift;
            return make_tuple(x_grid, y_grid, z_grid);
        }

        inline tuple<double, double, double> flag2xyz(size_t loc_flag) const {
            int x_grid, y_grid, z_grid;
            tie(x_grid, y_grid, z_grid) = flag2grid(loc_flag);
            return make_tuple(x_grid * x_step + x_init, y_grid * y_step + y_init, z_grid * z_step + z_shift);
        }

        bool is_valid_coord(double x, double y, double z) const {
            /*
             * 不使用<=或>=的原因在于搜索空间外轮廓不一定为长方体
             * */
            if (x < xmin || x > xmax) { return false; }
            if (y < ymin || y > ymax) { return false; }
            if (z < zmin || z > zmax) { return false; }
            return true;
        }

        vector<size_t> get_neighbors(double x, double y, double z, int scale, list<string> methods) const {
            vector<size_t> neighbors;

            for (string method: methods) {
                list<tuple<int, int, int>> candidate;
                if (method == "candidate_1D") {
                    candidate = candidate_1D;
                } else if (method == "candidate_2D") {
                    candidate = candidate_2D;
                } else if (method == "candidate_3D") {
                    candidate = candidate_3D;
                }

                int x_num, y_num, z_num;
                double x_, y_, z_;
                for (auto next: candidate) {
                    tie(x_num, y_num, z_num) = next;
                    x_ = x + x_num * x_step * scale;
                    y_ = y + y_num * y_step * scale;
                    z_ = z + z_num * z_step * scale;
                    if (is_valid_coord(x_, y_, z_)) {
                        neighbors.emplace_back(xyz2flag(x_, y_, z_));
                    }
                }
            }

            return neighbors;
        }

        inline double get_manhattan_cost(size_t loc_flag0, size_t loc_flag1) const {
            /*
             * Here is grid cost
             * */
            int x_grid_0, y_grid_0, z_grid_0;
            tie(x_grid_0, y_grid_0, z_grid_0) = flag2grid(loc_flag0);
            int x_grid_1, y_grid_1, z_grid_1;
            tie(x_grid_1, y_grid_1, z_grid_1) = flag2grid(loc_flag1);
            return compute_manhattan_cost(x_grid_0, y_grid_0, z_grid_0, x_grid_1, y_grid_1, z_grid_1);
        }

        inline double get_euler_cost(size_t loc_flag0, size_t loc_flag1) const {
            /*
             * Here is grid cost
             * */
            int x_grid_0, y_grid_0, z_grid_0;
            tie(x_grid_0, y_grid_0, z_grid_0) = flag2grid(loc_flag0);
            int x_grid_1, y_grid_1, z_grid_1;
            tie(x_grid_1, y_grid_1, z_grid_1) = flag2grid(loc_flag1);
            return compute_euler_cost(x_grid_0, y_grid_0, z_grid_0, x_grid_1, y_grid_1, z_grid_1);
        }

        inline void resample_path(vector<vector<double>> &path, vector<vector<double>> &resample_path, double reso) {
            /*
             * 当x_step, y_step, z_step步长超出物体半径时，以路径点作为管道形状表述将会有严重信息缺失，因此需要减少信息缺陷程度
             * */
            double x0, y0, z0, dist;
            for (int i = 0; i < path.size(); ++i) {
                if (i == 0) {
                    x0 = path[i][0];
                    y0 = path[i][1];
                    z0 = path[i][2];
                    resample_path.emplace_back(vector<double>(path[i]));
                    continue;
                }

                dist = norm2_dist(x0, y0, z0, path[i][0], path[i][1], path[i][2]);
                if (dist < reso) {
                    resample_path.emplace_back(vector<double>(path[i]));
                } else {
                    double num = ceil(dist / reso);
                    double vec_x = path[i][0] - x0;
                    double vec_y = path[i][1] - y0;
                    double vec_z = path[i][2] - z0;

                    for (int j = 0; j < num; ++j) {
                        resample_path.emplace_back(vector<double>{
                                x0 + vec_x * (j / num), y0 + vec_y * (j / num), z0 + vec_z * (j / num)
                        });
                    }
                }

                x0 = path[i][0];
                y0 = path[i][1];
                z0 = path[i][2];
            }
        }
    };

    class StandardGridEnv {
    public:
        double x_grid_length, y_grid_length, z_grid_length;
        int size_of_x, size_of_y, size_of_z;
        double x_init, y_init, z_init;
        double unit_length;

        StandardGridEnv(
                int size_of_x, int size_of_y, int size_of_z,
                double x_init, double y_init, double z_init,
                int x_grid_length, int y_grid_length, int z_grid_length
        ) : size_of_x(size_of_x), size_of_y(size_of_y), size_of_z(size_of_z),
            x_init(x_init), y_init(y_init), z_init(z_init),
            x_grid_length(x_grid_length), y_grid_length(y_grid_length), z_grid_length(z_grid_length) {
            assert(size_of_x <= 1000 & size_of_y <= 1000 & size_of_z <= 1000);
            unit_length = sqrt(pow(x_grid_length, 2.0) + pow(y_grid_length, 2.0) + pow(z_grid_length, 2.0));
        }

        ~StandardGridEnv() {}

        inline size_t xyz2flag(int x_grid, int y_grid, int z_grid) const {
            return z_grid * (size_of_x * size_of_y) + y_grid * size_of_x + x_grid;
        }

        inline tuple<int, int, int> flag2grid(size_t loc_flag) const {
            int z_grid = loc_flag / (size_of_x * size_of_y);
            int y_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) / size_of_x;
            int x_grid = (loc_flag - z_grid * (size_of_x * size_of_y)) % size_of_x;
            return make_tuple(x_grid, y_grid, z_grid);
        }

        inline tuple<double, double, double> flag2xyz(size_t loc_flag) const {
            int x_grid, y_grid, z_grid;
            tie(x_grid, y_grid, z_grid) = flag2grid(loc_flag);
            return make_tuple(
                    x_grid * x_grid_length + x_init,
                    y_grid * y_grid_length + y_init,
                    z_grid * z_grid_length + z_init
            );
        }

        inline tuple<double, double, double> grid2xyz(int x_grid, int y_grid, int z_grid) const {
            return make_tuple(
                    x_grid * x_grid_length + x_init,
                    y_grid * y_grid_length + y_init,
                    z_grid * z_grid_length + z_init
            );
        }

        bool is_valid_coord(double x, double y, double z) const {
            /*
             * 不使用<=或>=的原因在于搜索空间外轮廓不一定为长方体
             * */
            if (x < 0 || x > size_of_x) { return false; }
            if (y < 0 || y > size_of_y) { return false; }
            if (z < 0 || z > size_of_z) { return false; }
            return true;
        }

        vector<size_t>
        get_neighbors(size_t loc_flag, int scale_x, int scale_y, int scale_z, list<string> methods) const {
            int x, y, z;
            tie(x, y, z) = flag2grid(loc_flag);
            return get_neighbors(x, y, z, scale_x, scale_y, scale_z, methods);
        }

        vector<size_t>
        get_neighbors(int x, int y, int z, int scale_x, int scale_y, int scale_z, list<string> methods) const {
            vector<size_t> neighbors;
            for (string method: methods) {
                list<tuple<int, int, int>> candidate;
                if (method == "candidate_1D") {
                    candidate = candidate_1D;
                } else if (method == "candidate_2D") {
                    candidate = candidate_2D;
                } else if (method == "candidate_3D") {
                    candidate = candidate_3D;
                }

                int x_num, y_num, z_num;
                int x_, y_, z_;
                for (auto next: candidate) {
                    tie(x_num, y_num, z_num) = next;
                    x_ = x + x_num * scale_x;
                    y_ = y + y_num * scale_y;
                    z_ = z + z_num * scale_z;
                    if (is_valid_coord(x_, y_, z_)) {
                        neighbors.emplace_back(xyz2flag(x_, y_, z_));
                    }
                }
            }
            return neighbors;
        }

        inline double get_manhattan_cost(size_t loc_flag0, size_t loc_flag1) const {
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
            double x0, y0, z0;
            tie(x0, y0, z0) = grid2xyz(grid_x0, grid_y0, grid_z0);

            double x1, y1, z1;
            tie(x1, y1, z1) = grid2xyz(grid_x1, grid_y1, grid_z1);

            double x2, y2, z2;
            tie(x2, y2, z2) = grid2xyz(grid_x2, grid_y2, grid_z2);

            double radius = circle_3point_radius(x0, y0, z0, x1, y1, z1, x2, y2, z2, is_left);
            if (radius >= ref_radius) {
                return 0.0;
            }

            return pow(1.0 - radius / ref_radius, 2.0) * weight;
        }

    };
}

#endif //MAPF_PIPELINE_GRID_ENV_H
