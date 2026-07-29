#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

#include "BaseHead.h"

namespace TaskSpace
{
    struct TaskData
    {
        std::string name;
        uint32_t priority;
        
        TaskData(const std::string name, const uint32_t priority) : name(name), priority(priority) {}
        virtual ~TaskData() = default;
    };

    struct TaskPairNodeData : public TaskData
    {
        std::string start_name;
        std::vector<LocXyz> start_locs;
        std::string end_name;
        std::vector<LocXyz> end_locs;
        double radius;

        TaskPairNodeData(
            const std::string name,
            const std::string start_name,
            const std::string end_name,
            const std::vector<LocXyz> start_locs,
            const std::vector<LocXyz> end_locs,
            const double radius,
            const uint32_t priority)
            : TaskData(name, priority), start_name(start_name), end_name(end_name), 
            start_locs(start_locs), end_locs(end_locs), radius(radius) {}
    };

    class TaskSet
    { 
    public:
        TaskSet(const std::string name) : name(name) {}
        virtual ~TaskSet() = default;

        virtual void add_task() = 0;
        virtual size_t get_task_size() const = 0;
        virtual const TaskData& get_task(uint32_t priority) const = 0;
        virtual const void* get_tasks() const = 0;

        std::string name;
    };

    class PairNodeTaskSet : public TaskSet
    {
    public:
        PairNodeTaskSet(const std::string name) : TaskSet(name) {}

        void add_task() override {}

        void add_task(
            const std::string name,
            const std::string start_name,
            const std::string end_name,
            const std::vector<LocXyz> start_locs,
            const std::vector<LocXyz> end_locs,
            const double radius)
        {
            auto task = TaskPairNodeData(name, start_name, end_name, start_locs, end_locs, radius, m_tasks.size());
            m_tasks.push_back(task);
            m_max_radius = std::max(m_max_radius, radius);
        }

        void add_landmark_to_task(uint32_t priority, const std::string landmark_name, const LocXyz landmark_loc)
        {
            auto& original_task = m_tasks[priority];

            // Create first task (start to landmark) with same priority
            auto first_task = TaskPairNodeData(
                original_task.name + "_" + landmark_name,
                original_task.start_name,
                landmark_name,
                original_task.start_locs,
                std::vector<LocXyz>{landmark_loc},
                original_task.radius,
                priority);

            // Create second task (landmark to end) with priority + 1
            auto second_task = TaskPairNodeData(
                landmark_name + "_" + original_task.name,
                landmark_name,
                original_task.end_name,
                std::vector<LocXyz>{landmark_loc},
                original_task.end_locs,
                original_task.radius,
                priority + 1);

            // Replace original task with first task
            m_tasks[priority] = first_task;
            
            // First, make space for the new task
            m_tasks.insert(m_tasks.begin() + priority + 1, second_task);

            // Update priorities for all subsequent tasks
            for (size_t i = priority + 2; i < m_tasks.size(); ++i)
            {
                m_tasks[i].priority = i;
            }
        }

        size_t get_task_size() const override {
            return m_tasks.size();
        }

        const TaskPairNodeData& get_task(uint32_t priority) const override {
            return m_tasks[priority];
        }

        const void* get_tasks() const override {
            return &m_tasks;
        }

        double m_max_radius = 0.0;
    private:
        std::vector<TaskPairNodeData> m_tasks;
    };

}

#endif // TASK_MANAGER_H
