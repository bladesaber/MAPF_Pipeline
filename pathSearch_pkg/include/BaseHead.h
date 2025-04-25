#pragma once

#include "iostream"
#include <math.h>
#include "numeric"
#include <tuple>
#include <map>
#include "string"
#include <ctime>
#include "set"
#include "list"
#include <vector>
#include "assert.h"
#include <cmath>
#include <unordered_map>
#include <array>
#include "Eigen/Dense"

struct LocXyz
{
    int32_t x, y, z;

    LocXyz() : x(0), y(0), z(0) {}

    LocXyz(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}

    bool operator==(const LocXyz &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    // Optional: Conversion to tuple if needed
    operator std::tuple<uint16_t, uint16_t, uint16_t>() const
    {
        return {x, y, z};
    }

    bool operator!=(const LocXyz &other) const
    {
        return !(*this == other);
    }

    LocXyz operator+(const LocXyz &other) const
    {
        return {x + other.x, y + other.y, z + other.z};
    }

    LocXyz operator-(const LocXyz &other) const
    {
        return {x - other.x, y - other.y, z - other.z};
    }

    // Add less than operator for sorting
    bool operator<(const LocXyz &other) const
    {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }

    friend std::ostream &operator<<(std::ostream &os, const LocXyz &loc)
    {
        os << "(" << loc.x << ", " << loc.y << ", " << loc.z << ")";
        return os;
    }
};

struct CoordHash
{
    std::size_t operator()(const LocXyz &coord) const
    {
        return (static_cast<uint64_t>(coord.x) << 32) ^
               (static_cast<uint64_t>(coord.y) << 16) ^
               static_cast<uint64_t>(coord.z);
    }
};

struct PathData
{
    std::string m_name;
    std::vector<LocXyz> m_path;
    double m_radius;
    double m_cost;

    PathData() = default;
    
    PathData(
        const std::string &name,
        const std::vector<LocXyz> &path,
        const double radius)
    {
        m_name = name;
        m_path = path;
        m_radius = radius;
    }

    ~PathData() = default;

    double estimate_cost()
    {
        LocXyz last_node = m_path[0];
        for (int i = 1; i < m_path.size(); i++)
        {
            LocXyz node = m_path[i];
            m_cost += sqrt(
                (node.x - last_node.x) * (node.x - last_node.x) +
                (node.y - last_node.y) * (node.y - last_node.y) +
                (node.z - last_node.z) * (node.z - last_node.z));
            last_node = node;
        }
        return m_cost;
    }

    size_t get_size()
    {
        return m_path.size();
    }
};

struct PathSetData
{
    std::vector<PathData> m_paths;
    double m_cost = 0.0;

    PathSetData() = default;
    ~PathSetData() = default;

    void add_path(
        const std::string &name,
        const std::vector<LocXyz> &nodes,
        const double radius)
    {
        m_paths.push_back(PathData(name, nodes, radius));
    }

    void estimate_cost()
    {
        m_cost = 0.0;
        for (auto &path : m_paths)
        {
            m_cost += path.estimate_cost();
        }
    }

    size_t get_size()
    {
        size_t size = 0;
        for (auto &path : m_paths)
        {
            size += path.get_size();
        }
        return size;
    }

    double get_max_radius()
    {
        double max_radius = 0.0;
        for (auto &path : m_paths)
        {
            max_radius = std::max(max_radius, path.m_radius);
        }
        return max_radius;
    }
};

struct Conflict
{
    std::string group0;
    LocXyz loc0;
    double radius0;

    std::string group1;
    LocXyz loc1;
    double radius1;

    double conflict_cost;

    Conflict() = default;
    Conflict(
        const std::string &group0, const LocXyz &loc0, const double radius0, 
        const std::string &group1, const LocXyz &loc1, const double radius1, 
        const double conflict_cost)
    {
        this->group0 = group0;
        this->loc0 = loc0;
        this->radius0 = radius0;
        this->group1 = group1;
        this->loc1 = loc1;
        this->radius1 = radius1;
        this->conflict_cost = conflict_cost;
    }
};

struct ConflictSet
{
    std::vector<Conflict> conflicts;
    std::map<std::string, std::vector<std::tuple<size_t, bool>>> conflict_index_map;

    size_t m_conflict_size = 0;
    std::map<std::string, double> m_group_cost;

    ConflictSet() = default;

    ConflictSet(std::vector<std::string> group_names)
    {
        for (auto &group_name : group_names)
        {
            m_group_cost[group_name] = 0.0;
        }
        m_conflict_size = 0;
    }

    ~ConflictSet() = default;

    void add_conflict(const Conflict &conflict)
    {
        conflicts.push_back(conflict);
        conflict_index_map[conflict.group0].push_back(std::make_tuple(conflicts.size() - 1, true));
        conflict_index_map[conflict.group1].push_back(std::make_tuple(conflicts.size() - 1, false));
        m_group_cost[conflict.group0] += conflict.conflict_cost;
        m_group_cost[conflict.group1] += conflict.conflict_cost;
        m_conflict_size++;
    }

    void clear()
    {
        conflicts.clear();
        conflict_index_map.clear();
    }
};

struct Constraint
{
    std::string run_group0;
    LocXyz obs0_loc;
    double obs0_radius;

    std::string run_group1;
    LocXyz obs1_loc;
    double obs1_radius;

    Constraint() = default;
    Constraint(const std::string &run_group0, const LocXyz &obs0_loc, const double obs0_radius, const std::string &run_group1, const LocXyz &obs1_loc, const double obs1_radius)
    {
        this->run_group0 = run_group0;
        this->obs0_loc = obs0_loc;
        this->obs0_radius = obs0_radius;
        this->run_group1 = run_group1;
        this->obs1_loc = obs1_loc;
        this->obs1_radius = obs1_radius;
    }

    friend std::ostream &operator<<(std::ostream &os, const Constraint &constraint)
    {
        os << "run_group0: " << constraint.run_group0 << ", obs0_loc: " << constraint.obs0_loc << ", obs0_radius: " << constraint.obs0_radius << ", run_group1: " << constraint.run_group1 << ", obs1_loc: " << constraint.obs1_loc << ", obs1_radius: " << constraint.obs1_radius;
        return os;
    }
};

struct ObstacleData
{
    LocXyz loc;
    double radius;

    ObstacleData(const LocXyz &loc, const double radius) : loc(loc), radius(radius) {}

    friend std::ostream &operator<<(std::ostream &os, const ObstacleData &obstacle)
    {
        os << "loc: " << obstacle.loc << ", radius: " << obstacle.radius;
        return os;
    }
};

typedef std::vector<ObstacleData> ObstacleList;

struct CbsResult
{
    std::map<std::string, PathSetData> paths;

    CbsResult() = default;
    ~CbsResult() = default;
    
    void add_result(const std::string &group_name, const PathSetData path_set)
    {
        paths[group_name] = path_set;
    }

    void reset()
    {
        paths.clear();
    }
};

struct DynamicObstacleMap
{
    std::map<std::string, std::vector<ObstacleData>> dynamic_obstacle_map;

    DynamicObstacleMap() = default;
    ~DynamicObstacleMap() = default;

    void add_obstacle(const std::string &group_name, const ObstacleData &obstacle)
    {
        dynamic_obstacle_map[group_name].push_back(obstacle);
    }

    const std::vector<ObstacleData>& get_obstacles(const std::string &group_name) const
    {
        return dynamic_obstacle_map.at(group_name);
    }

    bool has_obstacle(const std::string &group_name) const
    {
        return dynamic_obstacle_map.count(group_name) > 0;
    }
};
