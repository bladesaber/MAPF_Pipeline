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

#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

typedef std::vector<size_t> Path;
typedef std::vector<std::tuple<double, double, double>> DetailPath;

#endif /* MAPF_PIPELINE_COMMON_H */