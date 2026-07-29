#ifndef SET_PLANNER_H
#define SET_PLANNER_H

#include "BaseHead.h"
#include "TaskManager.h"
#include "AstarSearch.h"
#include "SearchGrid.h"

namespace SearchSpace
{
    class SetPlanner
    {
    public:
        SetPlanner(SearchSpace::BaseSearch &planner) : m_planner(planner) {};

        virtual ~SetPlanner() = default;

        void set_tasks(TaskSpace::TaskSet &task_set)
        {
            m_task_set = &task_set;
        }

        virtual void preprocess() = 0;
        virtual void before_task(const TaskSpace::TaskData *task, std::vector<LocXyz> &start_set, std::vector<LocXyz> &end_set) = 0;
        virtual void after_task(const TaskSpace::TaskData *task, const std::vector<LocXyz> &result) = 0;
        virtual bool search(GridSpace::SearchGrid *grid, PathSetData &result_sets) = 0;

    protected:
        TaskSpace::TaskSet *m_task_set;
        SearchSpace::BaseSearch &m_planner;
    };

    class SetPairNodePlanner : public SetPlanner
    {
    public:
        SetPairNodePlanner(SearchSpace::BaseSearch &planner)
            : SetPlanner(planner) {}

        ~SetPairNodePlanner() override = default;

        void set_tasks(TaskSpace::PairNodeTaskSet &task_set)
        {
            m_task_set = &task_set;
        }

        void preprocess() override;
        void before_task(const TaskSpace::TaskData *task, std::vector<LocXyz> &start_set, std::vector<LocXyz> &end_set) override;
        void after_task(const TaskSpace::TaskData *task, const std::vector<LocXyz> &result) override;
        bool search(GridSpace::SearchGrid *grid, PathSetData &result_sets) override;

    protected:
        std::unordered_map<std::string, uint32_t> m_set_map;
        std::vector<std::vector<LocXyz>> m_set_data;
    };

}

#endif // SET_PLANNER_H
