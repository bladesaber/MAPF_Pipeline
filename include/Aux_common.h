#ifndef MAPF_PIPELINE_AUX_COMMON_H
#define MAPF_PIPELINE_AUX_COMMON_H

#include "iostream"
#include <math.h>
#include <tuple>
#include <map>
#include "string"
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <pybind11/stl.h>

double fmodr( double x, double y)
{
    return x - y * floor(x / y);
}

double mod2pi(double theta)
{
    return fmodr( theta, 2 * M_PI );
}

double mod2singlePi(double theta){
    return fmodr(theta + M_PI, 2 * M_PI ) - M_PI;
}

double rad2degree(double theta){
    return theta / M_PI * 180.0;
}

std::tuple<double, double, double> polar3D_to_vec3D(double alpha, double beta, double length){
    double dz = length * sin(beta);
    double dl = length * cos(beta);
    double dx = dl * cos(alpha);
    double dy = dl * sin(alpha);
    return std::make_tuple(dx, dy, dz);
}

std::tuple<double, double> vec3D_to_polar3D(double vec_x, double vec_y, double vec_z){
    double alpha = atan2(vec_y, vec_x);
    double length = sqrt( pow(vec_x, 2) + pow(vec_y, 2));
    double beta = atan2(vec_z, length);
    return std::make_tuple(alpha, beta);
}

#endif /* MAPF_PIPELINE_AUX_COMMON_H */