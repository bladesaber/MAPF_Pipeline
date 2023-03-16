//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_COMMON_H
#define MAPF_PIPELINE_COMMON_H

#include "iostream"
#include <pybind11/stl.h>
#include "vector"
#include <tuple>
#include <list>
#include <set>
#include <map>
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <cmath>
#include <ctime>

// struct PathEntry
// {
//     int location = -1;
//     PathEntry(int loc = -1) {
//         location = loc;
//     }
// };

// typedef std::vector<PathEntry> Path;
// std::ostream& operator<<(std::ostream& os, const Path& path)
// {
//     for (const auto& state : path)
//     {
//         os << state.location << "->";
//     }
//     return os;
// }

typedef std::vector<size_t> Path;

#endif //MAPF_PIPELINE_COMMON_H
