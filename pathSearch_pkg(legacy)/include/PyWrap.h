#ifndef PY_WRAP_H
#define PY_WRAP_H

#include "BaseHead.h"
#include "SearchGrid.h"
#include "AstarSearch.h"
#include "ConflictBasedSearch.h"

class PyWrapAstarRunner
{
public:
    PyWrapAstarRunner(std::string name, GridSpace::SearchGrid* static_grid, SearchSpace::BaseSearch* planner)
        : name(name), static_grid(static_grid), planner(planner), grid(static_grid)
    {};

    ~PyWrapAstarRunner() = default;

    void add_dynamic_obstacle(ObstacleData obstacle_data)
    {
        grid.set_obstacle(LocXyz(obstacle_data.loc), obstacle_data.radius);
    }
    
    bool run(const std::vector<LocXyz> starts, const std::vector<LocXyz> goals)
    {
        result.m_path.clear();
        grid.preprocess();
        return planner->search(&grid, starts, goals, result);
    };

    std::string name;
    PathData result;

private:
    GridSpace::SearchGrid* static_grid;
    GridSpace::SearchGridSet grid;

    SearchSpace::BaseSearch* planner;
};

class PyWrapGroupAstarRunner
{
public:
    PyWrapGroupAstarRunner(std::string name, GridSpace::SearchGrid* static_grid, SearchSpace::BaseSearch* planner)
        : name(name), static_grid(static_grid), grid(static_grid), task_set(name), planner(planner), set_planner(*planner)
    {};

    ~PyWrapGroupAstarRunner() = default;

    void add_dynamic_obstacle(ObstacleData obstacle_data)
    {
        grid.set_obstacle(LocXyz(obstacle_data.loc), obstacle_data.radius);
    }

    void add_task(
        const std::string name,
        const std::string start_name,
        const std::string end_name,
        const std::vector<LocXyz> start_locs,
        const std::vector<LocXyz> end_locs,
        const double radius)
    {
        task_set.add_task(name, start_name, end_name, start_locs, end_locs, radius);
    }

    bool run()
    {
        set_planner.set_tasks(task_set);
        grid.preprocess();
        return set_planner.search(&grid, result);
    }

    std::string name;
    PathSetData result;

private:
    GridSpace::SearchGrid* static_grid;
    GridSpace::SearchGridSet grid;

    TaskSpace::PairNodeTaskSet task_set;

    SearchSpace::BaseSearch* planner;
    SearchSpace::SetPairNodePlanner set_planner;
};

class PyWrapCbsRunner
{
public:
    PyWrapCbsRunner(std::string name, GridSpace::SearchGrid* static_grid)
        : name(name), static_grid(static_grid)
    {};

    ~PyWrapCbsRunner(){
        for (auto& [name, planner] : set_planners)
        {
            delete planner;
        }
        for (auto& [name, task_set] : task_sets)
        {
            delete task_set;
        }
        for (auto& [name, dynamic_obstacle_set] : dynamic_obstacle_map)
        {
            dynamic_obstacle_set.clear();
        }
    };

    void add_set(std::string name, SearchSpace::BaseSearch* planner)
    {
        task_sets[name] = new TaskSpace::PairNodeTaskSet(name);
        set_planners[name] = new SearchSpace::SetPairNodePlanner(*planner);
        dynamic_obstacle_map[name] = std::vector<ObstacleData>();
    }

    bool add_dynamic_obstacle(std::string task_name, ObstacleData obstacle_data)
    {
        if (dynamic_obstacle_map.find(task_name) == dynamic_obstacle_map.end())
        {
            return false;
        }
        dynamic_obstacle_map[task_name].push_back(obstacle_data);
        return true;
    }

    void add_task(
        std::string task_set_name, 
        std::string task_name, 
        std::string start_name, 
        std::string end_name, 
        std::vector<LocXyz> start_locs, 
        std::vector<LocXyz> end_locs, 
        double radius)
    {
        task_sets[task_set_name]->add_task(task_name, start_name, end_name, start_locs, end_locs, radius);
    }

    void set_cbs_planner(SearchSpace::ConflictBaseSearch* cbs_planner)
    {
        this->cbs_planner = cbs_planner;
        for (auto& [name, set_planner] : set_planners)
        {
            this->cbs_planner->add_task_set(task_sets[name], set_planner);
        }
    }

    bool init_root_node()
    {
        return cbs_planner->init_root_node(dynamic_obstacle_map, &result);
    }

    bool run()
    {
        return cbs_planner->search(&result);
    }

    bool single_step_search()
    {
        return cbs_planner->single_step_search(&result);
    }

    std::string name;
    CbsResult result;
    
private:
    GridSpace::SearchGrid* static_grid;
    std::map<std::string, std::vector<ObstacleData>> dynamic_obstacle_map;
    
    std::map<std::string, TaskSpace::PairNodeTaskSet*> task_sets;
    std::map<std::string, SearchSpace::SetPairNodePlanner*> set_planners;

    SearchSpace::ConflictBaseSearch* cbs_planner;
};

#endif // PY_WRAP_H