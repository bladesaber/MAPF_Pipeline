#include "SetPlanner.h"

namespace SearchSpace
{
    void SetPairNodePlanner::preprocess()
    {
        m_set_data.clear();
        m_set_map.clear();

        const auto *tasks = static_cast<const std::vector<TaskSpace::TaskPairNodeData> *>(m_task_set->get_tasks());
        for (const auto &task : *tasks)
        {
            if (m_set_map.find(task.start_name) == m_set_map.end())
            {
                m_set_map[task.start_name] = m_set_data.size();
                m_set_data.push_back(std::vector<LocXyz>{task.start_loc});
            }

            if (m_set_map.find(task.end_name) == m_set_map.end())
            {
                m_set_map[task.end_name] = m_set_data.size();
                m_set_data.push_back(std::vector<LocXyz>{task.end_loc});
            }
        }
    }

    void SetPairNodePlanner::before_task(const TaskSpace::TaskData *task, std::vector<LocXyz> &start_set, std::vector<LocXyz> &end_set)
    {
        auto *pair_task = static_cast<const TaskSpace::TaskPairNodeData *>(task);
        start_set = m_set_data[m_set_map[pair_task->start_name]];
        end_set = m_set_data[m_set_map[pair_task->end_name]];
        m_planner.set_radius(pair_task->radius);
    }

    void SetPairNodePlanner::after_task(const TaskSpace::TaskData *task, const std::vector<LocXyz> &result)
    {
        auto *pair_task = static_cast<const TaskSpace::TaskPairNodeData *>(task);

        std::vector<LocXyz> new_set;
        std::vector<LocXyz> prefix_set = m_set_data[m_set_map[pair_task->start_name]];
        std::vector<LocXyz> suffix_set = m_set_data[m_set_map[pair_task->end_name]];

        if (result.size() == 0)
        {
            new_set.insert(new_set.end(), prefix_set.begin(), prefix_set.end());
        }
        else if (result.size() == 2)
        {
            new_set.insert(new_set.end(), prefix_set.begin(), prefix_set.end());
            new_set.insert(new_set.end(), suffix_set.begin(), suffix_set.end());
        }
        else
        {
            new_set.insert(new_set.end(), prefix_set.begin(), prefix_set.end());
            new_set.insert(new_set.end(), result.begin() + 1, result.end() - 1);
            new_set.insert(new_set.end(), suffix_set.begin(), suffix_set.end());
        }

        int new_id = m_set_data.size();
        m_set_data.push_back(new_set);

        for (auto &pair : m_set_map)
        {
            if (pair.second == m_set_map[pair_task->start_name] ||
                pair.second == m_set_map[pair_task->end_name])
            {
                pair.second = new_id;
            }
        }
    }

    bool SetPairNodePlanner::search(GridSpace::SearchGrid *grid, PathSetData &result_sets)
    {
        preprocess();
        const auto *tasks = static_cast<const std::vector<TaskSpace::TaskPairNodeData> *>(m_task_set->get_tasks());
        for (const auto &task : *tasks)
        {
            std::vector<LocXyz> start_set, end_set;
            before_task(&task, start_set, end_set);
            std::vector<LocXyz> result;
            if (!m_planner.search(grid, start_set, end_set, result))
            {
                return false;
            }
            after_task(&task, result);
            result_sets.add_path(task.name, result, task.radius);
        }
        return true;
    }

}
