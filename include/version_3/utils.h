#ifndef MAPF_PIPELINE_UTILS_H
#define MAPF_PIPELINE_UTILS_H

#include "common.h"
#include "kdtree.h"

template<typename T>
void printPointer(T& a, std::string tag);

double fmodr( double x, double y);

double mod2pi(double theta);

double mod2singlePi(double theta);

double rad2degree(double theta);

std::tuple<double, double, double> polar3D_to_vec3D(double alpha, double beta, double length);

std::tuple<double, double> vec3D_to_polar3D(double vec_x, double vec_y, double vec_z);

double point2LineDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
);

double norm2_distance(
    double x0, double y0, double z0,
    double x1, double y1, double z1
);

double point2LineSegmentDistance(
    double lineStart_x, double lineStart_y, double lineStart_z,
    double lineEnd_x, double lineEnd_y, double lineEnd_z,
    double point_x, double point_y, double point_z
);

double round_decimal(double x, int k);

double roundInterval(double x, double interval);

#endif