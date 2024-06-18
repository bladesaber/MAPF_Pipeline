//
// Created by admin123456 on 2024/6/7.
//

#include "math_utils.h"

tuple<double, CellXYZ, CellXYZ> line2line_cross(
        double ax0, double ay0, double az0,
        double ax1, double ay1, double az1,
        double bx0, double by0, double bz0,
        double bx1, double by1, double bz1,
        bool clamp
) {
    Vector3d cross_point_a, cross_point_b;
    double dist;

    Vector3d a0{ax0, ay0, az0};
    Vector3d a1{ax1, ay1, az1};
    Vector3d b0{bx0, by0, bz0};
    Vector3d b1{bx1, by1, bz1};

    Vector3d A = a1 - a0;
    Vector3d B = b1 - b0;
    double magA = A.norm();
    double magB = B.norm();
    Vector3d norm_A = A / magA;
    Vector3d norm_B = B / magB;

    Vector3d cross_vec = norm_A.cross(norm_B);
    double cross_norm_square = cross_vec.squaredNorm();

    if (cross_norm_square == 0) { // If lines are parallel (norm_square=0)
        double d_a0_b0 = norm_A.dot(b0 - a0);
        double d_a0_b1 = norm_A.dot(b1 - a0);
        double extend_dist_in_a_line = min(d_a0_b0, d_a0_b1); // 取d_a0_b0与d_a0_b1中最小，则表明最近点接近a0

        double d_b0_a0 = norm_A.dot(a0 - b0);
        double d_b0_a1 = norm_A.dot(a1 - b0);
        double extend_dist_in_b_line = min(d_b0_a0, d_b0_a1); // 取d_b0_a0与d_b0_a1中最小，则表明最近点接近b0

        dist = sqrt((b0 - a0).squaredNorm() - pow(d_a0_b0, 2.0)); // distance between parallel lines

        if (clamp) {
            extend_dist_in_a_line = min(magA, max(extend_dist_in_a_line, 0.0));
            extend_dist_in_b_line = min(magB, max(extend_dist_in_b_line, 0.0));
        }
        cross_point_a = (a0 + extend_dist_in_a_line * norm_A);
        cross_point_b = (b0 + extend_dist_in_b_line * norm_B);

    } else {
        Matrix3d mat_A;
        mat_A << (b0 - a0).transpose(), norm_B.transpose(), cross_vec.transpose();
        double det_A = mat_A.determinant();

        Matrix3d mat_B;
        mat_B << (b0 - a0).transpose(), norm_A.transpose(), cross_vec.transpose();
        double det_B = mat_B.determinant();

        double extend_dist_in_a_line = det_A / cross_norm_square;
        double extend_dist_in_b_line = det_B / cross_norm_square;

        if (clamp){
            extend_dist_in_a_line = min(magA, max(extend_dist_in_a_line, 0.0));
            extend_dist_in_b_line = min(magB, max(extend_dist_in_b_line, 0.0));

            cross_point_a = (a0 + extend_dist_in_a_line * norm_A);
            cross_point_b = (b0 + extend_dist_in_b_line * norm_B);

            double dot_A = norm_A.dot(cross_point_b - a0);
            dot_A = min(magA, max(dot_A, 0.0));
            cross_point_a = (a0 + dot_A * norm_A);

            double dot_B = norm_B.dot(cross_point_a - b0);
            dot_B = min(magB, max(dot_B, 0.0));
            cross_point_b = (b0 + dot_B * norm_B);


        } else {
            cross_point_a = (a0 + extend_dist_in_a_line * norm_A);
            cross_point_b = (b0 + extend_dist_in_b_line * norm_B);
        }

        dist = (cross_point_a - cross_point_b).norm();
    }

    return make_tuple(
            dist,
            make_tuple(cross_point_a.x(), cross_point_a.y(), cross_point_a.z()),
            make_tuple(cross_point_b.x(), cross_point_b.y(), cross_point_b.z())
    );
}