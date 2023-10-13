#ifndef MAPF_PIPELINE_GROUP_ASTARSOLVER_H
#define MAPF_PIPELINE_GROUP_ASTARSOLVER_H

#include "eigen3/Eigen/Core"
#include "math.h"

#include "common.h"
#include "AstarSolver.h"
#include "kdtree_xyzrl.h"

namespace PlannerNameSpace {

    Path_XYZRL sampleDetailPath(Path_XYZR &path_xyzr, double stepLength);

    class TaskLeaf {
    public:
        std::set<size_t> locMembers;
        std::set<size_t> path;

        TaskLeaf() {};

        TaskLeaf(size_t loc) {
            locMembers.insert(loc);
            path.insert(loc);
        };

        ~TaskLeaf() {};

        void mergeGroupMembers(TaskLeaf *rhs0, TaskLeaf *rhs1) {
            for (size_t loc: rhs0->locMembers) {
                locMembers.insert(loc);
            }
            for (size_t loc: rhs1->locMembers) {
                locMembers.insert(loc);
            }
        }

        void mergePath(std::set<size_t> &oth_path) {
            for (size_t loc: oth_path) {
                path.insert(loc);
            }
        }

        void mergePath(std::vector<size_t> &oth_path) {
            for (size_t loc: oth_path) {
                path.insert(loc);
            }
        }
    };

    class TaskInfo {
    public:
        std::string tag;
        size_t loc0, loc1;
        double radius0, radius1;
        Path_XYZRL res_path;

        TaskInfo(std::string tag, size_t loc0, double radius0, size_t loc1, double radius1) :
                tag(tag), loc0(loc0), loc1(loc1), radius0(radius0), radius1(radius1) {};

        ~TaskInfo() {};
    };

    class GroupAstarSolver {
    public:
        GroupAstarSolver() {};

        ~GroupAstarSolver() {
            release();
        };

        std::set<size_t> locs;
        std::vector<TaskInfo *> taskTree;

        bool findPath(
                AStarSolver *solver, std::vector<ConstrainType> constraints,
                ConstraintTable &obstacle_table, Instance &instance, double stepLength
        ) {
            std::map<size_t, TaskLeaf *> locLeafsMap;
            for (size_t loc: locs) {
                locLeafsMap[loc] = new TaskLeaf(loc);
            }

            ConstraintTable constrain_table = ConstraintTable();
            for (auto constraint: constraints) {
                constrain_table.insert2CT(constraint);
            }

            Path path;
            Path_XYZR path_xyzr;
            bool success = false;

            size_t loc0, loc1;
            double radius0, radius1;
            double x, y, z, radius;
            for (size_t i = 0; i < taskTree.size(); i++) {
                loc0 = taskTree[i]->loc0;
                loc1 = taskTree[i]->loc1;
                radius0 = taskTree[i]->radius0;
                radius1 = taskTree[i]->radius1;

                std::vector<size_t> start_locs, goal_locs;
                start_locs.assign(locLeafsMap[loc0]->path.begin(), locLeafsMap[loc0]->path.end());
                goal_locs.assign(locLeafsMap[loc1]->path.begin(), locLeafsMap[loc1]->path.end());

                radius = (radius0 + radius1) / 2.0;
                path.clear();
                path = solver->findPath(
                        radius, // radius
                        constrain_table, // constrain_table
                        obstacle_table, // obstacle_table
                        instance, // instance
                        start_locs, // start_locs
                        goal_locs // goal_locs
                );

                if (path.size() == 0) {
                    std::cout << "[Warning]: Search Path " << taskTree[i]->tag << " Fail." << std::endl;
                    break;

                } else {
                    path_xyzr.clear();
                    for (size_t loc: path) {
                        std::tie(x, y, z) = instance.getCoordinate(loc);
                        path_xyzr.emplace_back(std::make_tuple(x, y, z, radius));
                    }
                    taskTree[i]->res_path = sampleDetailPath(path_xyzr, stepLength);
                }

                // -----------
                // restrict terminal point as goal
                double terminal_x, terminal_y, terminal_z;
                std::tie(terminal_x, terminal_y, terminal_z) = instance.getCoordinate(loc0);
                constrain_table.insert2CT(terminal_x, terminal_y, terminal_z, 0.0);
                std::tie(terminal_x, terminal_y, terminal_z) = instance.getCoordinate(loc1);
                constrain_table.insert2CT(terminal_x, terminal_y, terminal_z, 0.0);
                // -----------

                TaskLeaf *taskLeaf = new TaskLeaf();
                taskLeaf->mergeGroupMembers(locLeafsMap[loc0], locLeafsMap[loc1]);
                taskLeaf->mergePath(locLeafsMap[loc0]->path);
                taskLeaf->mergePath(locLeafsMap[loc1]->path);
                taskLeaf->mergePath(path);

                delete locLeafsMap[loc0], locLeafsMap[loc1];
                for (size_t loc: taskLeaf->locMembers) {
                    locLeafsMap[loc] = taskLeaf;
                }

                if (i == taskTree.size() - 1) {
                    success = true;
                    delete taskLeaf;
                }
            }

            if (!success) {
                std::set<size_t> release_set;
                for (auto iter: locLeafsMap) {
                    if (release_set.find(iter.first) != release_set.end()) {
                        continue;
                    }

                    for (size_t loc: iter.second->locMembers) {
                        release_set.insert(loc);
                    }
                    delete iter.second;
                }
                release_set.clear();
                locLeafsMap.clear();

            } else {
                updateLocTree();
            }

            return success;
        }

        KDTree_XYZRL *locTree;
        bool setup_tree = false;

        void updateLocTree() {
            if (setup_tree) {
                delete locTree;
            }

            locTree = new KDTree_XYZRL();
            double x, y, z, radius, length;
            for (TaskInfo *task: taskTree) {
                for (size_t i = 0; i < task->res_path.size(); i++) {
                    std::tie(x, y, z, radius, length) = task->res_path[i];
                    locTree->insertNode(0, x, y, z, radius, length);
                }
            }
            setup_tree = true;
        }

        void copy(std::shared_ptr<GroupAstarSolver> rhs, bool with_path = false) {
            for (TaskInfo *task: rhs->taskTree) {
                TaskInfo *new_obj = new TaskInfo(task->tag, task->loc0, task->radius0, task->loc1, task->radius1);

                if (with_path) {
                    new_obj->res_path = Path_XYZRL(task->res_path);
                }

                this->taskTree.emplace_back(new_obj);
            }
            this->locs = std::set<size_t>(rhs->locs);
        }

        void addTask(std::string tag, size_t loc0, double radius0, size_t loc1, double radius1) {
            locs.insert(loc0);
            locs.insert(loc1);
            taskTree.emplace_back(new TaskInfo(tag, loc0, radius0, loc1, radius1));
        }

        std::vector<Path_XYZRL> getPath() {
            std::vector<Path_XYZRL> taskPaths;
            for (TaskInfo *info: taskTree) {
                taskPaths.emplace_back(info->res_path);
            }
            return taskPaths;
        }

    private:
        void release() {
            for (size_t i = 0; i < taskTree.size(); i++) {
                delete taskTree[i];
            }
            taskTree.clear();

            if (setup_tree) {
                delete locTree;
            }
        }
    };

}
#endif