#ifndef PY_WRAP_H
#define PY_WRAP_H

#include "BaseHead.h"
#include "SearchGrid.h"
#include "AstarSearch.h"
#include "ConflictBasedSearch.h"

struct PyWrapAstarStruct
{
    GridSpace::SearchGrid* grid;
    SearchSpace::BaseSearch* search;
    std::vector<LocXyz> path;

    PyWrapAstarStruct(GridSpace::SearchGrid* grid, SearchSpace::BaseSearch* search)
    {
        this->grid = grid;
        this->search = search;
    };
    
    bool run(const std::vector<LocXyz> starts, const std::vector<LocXyz> goals)
    {
        return search->search(grid, starts, goals, path);
    };
};

struct PyWrapCbsStruct
{
    SearchSpace::ConflictBaseSearch* planner;
    CbsResult result;
    DynamicObstacleMap* dynamic_obstacle_map;

    PyWrapCbsStruct(SearchSpace::ConflictBaseSearch* planner, DynamicObstacleMap* dynamic_obstacle_map)
    {
        this->planner = planner;
        this->dynamic_obstacle_map = dynamic_obstacle_map;
    }

    bool run()
    {   
        planner->preprocess_root_node(*dynamic_obstacle_map);
        return planner->search(&result);
    }

    std::tuple<bool, CbsResult> single_step_run()
    {
        planner->preprocess_root_node(*dynamic_obstacle_map);
        return planner->single_step_search();
    }
};

#endif // PY_WRAP_H