#ifndef MAPF_PIPELINE_AUX_UTILS_H
#define MAPF_PIPELINE_AUX_UTILS_H

#include "Aux_common.h"

double fmodr( double x, double y);

double mod2pi(double theta);

double mod2singlePi(double theta);

double rad2degree(double theta);

std::tuple<double, double, double> polar3D_to_vec3D(double alpha, double beta, double length);

std::tuple<double, double> vec3D_to_polar3D(double vec_x, double vec_y, double vec_z);

#endif