#ifndef MAPF_PIPELINE_COMMON_H
#define MAPF_PIPELINE_COMMON_H

#include "iostream"
#include <math.h>
#include <tuple>
#include <map>
#include "string"
#include <ctime>
#include "set"
#include "list"
#include "vector"
#include "algorithm"
#include <climits>
#include <cfloat>
#include "assert.h"

#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define CUSTOM_TYPE1 short unsigned int
typedef CUSTOM_TYPE1 size_ut;

typedef std::vector<size_t> Path;
typedef std::vector<std::tuple<double, double, double, double>> DetailPath;

// x, y, z, radius
typedef std::tuple<double, double, double, double> ConstrainType;

#endif /* MAPF_PIPELINE_COMMON_H */