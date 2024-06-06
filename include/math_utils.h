//
// Created by admin123456 on 2024/5/31.
//

#ifndef MAPF_PIPELINE_MATH_UTILS_H
#define MAPF_PIPELINE_MATH_UTILS_H

#include "math.h"
#include "cfloat"
#include "tuple"
#include "vector"
#include "numeric"

using namespace std;

double norm2_dist(double x0, double y0, double z0, double x1, double y1, double z1) {
    return sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2) + pow(z0 - z1, 2));
}

double point2line_dist(
        double x, double y, double z, double lx0, double ly0, double lz0, double lx1, double ly1, double lz1) {
    double dist_start_point = norm2_dist(x, y, z, lx0, ly0, lz0);
    double dist_end_point = norm2_dist(x, y, z, lx1, ly1, lz1);
    double dist_start_end = norm2_dist(lx0, ly0, lz0, lx1, ly1, lz1);

    if (dist_start_point > dist_end_point && dist_start_point > dist_start_end) {
        return dist_end_point;
    } else if (dist_end_point > dist_start_point && dist_end_point > dist_start_end) {
        return dist_start_point;
    } else {
        double vec_sum = (lx1 - lx0) * (x - lx0) + (ly1 - ly0) * (y - ly0) + (lz1 - lz0) * (z - lz0);
        double cos_dist = vec_sum / dist_start_point;
        double res = sqrt(pow(dist_start_point, 2.0) - pow(cos_dist, 2.0));
        return res;
    }
}

double point2line_projection_dist(
        double x, double y, double z, double lx0, double ly0, double lz0, double lx1, double ly1, double lz1) {
    double dist_start_point = norm2_dist(x, y, z, lx0, ly0, lz0);

    double vec_sum = (lx1 - lx0) * (x - lx0) + (ly1 - ly0) * (y - ly0) + (lz1 - lz0) * (z - lz0);
    double cos_dist = vec_sum / dist_start_point;
    double res = sqrt(pow(dist_start_point, 2.0) - pow(cos_dist, 2.0));
    return res;
}

tuple<double, double, double> point2line_projection_xyz(
        double x, double y, double z, double lx0, double ly0, double lz0, double lx1, double ly1, double lz1) {
    double dist_start_point = norm2_dist(x, y, z, lx0, ly0, lz0);
    double dist_start_end = norm2_dist(lx0, ly0, lz0, lx1, ly1, lz1);

    double vec_sum = (lx1 - lx0) * (x - lx0) + (ly1 - ly0) * (y - ly0) + (lz1 - lz0) * (z - lz0);
    double cos_dist = vec_sum / dist_start_point;

    return make_tuple(lx0 + (lx1 - lx0) * cos_dist / dist_start_end,
                      ly0 + (ly1 - ly0) * cos_dist / dist_start_end,
                      lz0 + (lz1 - lz0) * cos_dist / dist_start_end);
}

bool is_point2line_projection_inner(
        double x, double y, double z, double lx0, double ly0, double lz0, double lx1, double ly1, double lz1) {
    double dist_start_point = norm2_dist(x, y, z, lx0, ly0, lz0);
    double dist_end_point = norm2_dist(x, y, z, lx1, ly1, lz1);
    double dist_start_end = norm2_dist(lx0, ly0, lz0, lx1, ly1, lz1);

    if (dist_start_end > dist_start_point && dist_start_end > dist_end_point) {
        return true;
    } else {
        return false;
    }
}

template<typename T>
T compute_manhattan_cost(T x0, T y0, T z0, T x1, T y1, T z1) {
    return abs(x0 - x1) + abs(y0 - y1) + abs(z0 - z1);
}

template<typename T>
T compute_euler_cost(T x0, T y0, T z0, T x1, T y1, T z1) {
    return sqrt(pow((x0 - x1), 2.0) + pow((y0 - y1), 2.0) + pow((z0 - z1), 2.0));
}

double line2line_cos(double vecx_0, double vecy_0, double vecz_0, double vecx_1, double vecy_1, double vecz_1) {
    double length_0 = sqrt(pow(vecx_0, 2.0) + pow(vecy_0, 2.0) + pow(vecz_0, 2.0));
    double length_1 = sqrt(pow(vecx_1, 2.0) + pow(vecy_1, 2.0) + pow(vecz_1, 2.0));

    if (length_0 == 0 || length_1 == 0){
        return 1.0;
    }
    return (vecx_0 * vecx_1 + vecy_0 * vecy_1 + vecz_0 * vecz_1) / length_0 / length_1;
}

double circle_3point_radius(
        double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2,
        bool is_left = true
) {
    double vec_x0 = x1 - x0;
    double vec_y0 = y1 - y0;
    double vec_z0 = z1 - z0;
    double vec_x1 = x2 - x1;
    double vec_y1 = y2 - y1;
    double vec_z1 = z2 - z1;
    double cos_val = line2line_cos(vec_x0, vec_y0, vec_z0, vec_x1, vec_y1, vec_z1);
    if (cos_val > 1.0 - 1e-2) {
        return DBL_MAX;
    }
    double dist;
    if (is_left) {
        dist = norm2_dist(x0, y0, z0, x1, y1, z1);
    } else {
        dist = min(norm2_dist(x0, y0, z0, x1, y1, z1), norm2_dist(x1, y1, z1, x2, y2, z2));
    }
    return dist / cos_val;
}

double mean(vector<double> data){
    return accumulate(data.begin(), data.end(), 0.0) / data.size();
}

#endif //MAPF_PIPELINE_MATH_UTILS_H
