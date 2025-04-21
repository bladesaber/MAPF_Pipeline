#ifndef PY_WRAP_H
#define PY_WRAP_H

#include "BaseHead.h"
#include "SearchGrid.h"
#include "AstarSearch.h"

struct PyWrapCbsResult
{
    std::map<std::string, PathSetData> results;

    PyWrapCbsResult() = default;

    // no use & and * for copy PathSetData
    void add_result(const std::string group_name, const PathSetData path_set)
    {
        results[group_name] = path_set;
    }

    ~PyWrapCbsResult()
    {
    }
};

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

#endif // PY_WRAP_H