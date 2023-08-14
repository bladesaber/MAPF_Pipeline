#ifndef MAPF_PIPELINE_CBS_NODE_H
#define MAPF_PIPELINE_CBS_NODE_H

#include "kdtree_xyzrl.h"
#include "conflict.h"
#include "groupAstarSolver.h"

using namespace PlannerNameSpace;

namespace CBSNameSpace {

    class CBSNode {
    public:
        CBSNode(double stepLength): stepLength(stepLength) {};

        ~CBSNode() {
            release();
        }

        std::map<size_t, std::shared_ptr<GroupAstarSolver>> groupAgentMap;
        std::map<size_t, std::shared_ptr<std::vector<ConstrainType>>> constrainsMap;
        std::map<size_t, double> pipe_conflictLength;

        double first_conflictDist = DBL_MAX;
        Conflict firstConflict;
        bool isConflict = false;

        size_t node_id = 0;
        double g_val = 0.0;
        double h_val = 0.0;
        int depth = 0;

        void update_Constrains(size_t groupIdx, const std::vector<ConstrainType> &new_constrains) {
            constrainsMap[groupIdx] = nullptr;
            constrainsMap[groupIdx] = std::make_shared<std::vector<ConstrainType>>(new_constrains);
        }

        bool update_GroupAgentPath(size_t groupIdx, AStarSolver *solver, Instance &instance,
                                   ConstraintTable &obstacle_table) {
            std::shared_ptr<GroupAstarSolver> groupAgent = std::make_shared<GroupAstarSolver>();
            groupAgent->copy(groupAgentMap[groupIdx]);
            groupAgentMap[groupIdx] = nullptr;

            bool success = groupAgent->findPath(
                    solver,
                    *(constrainsMap[groupIdx]),
                    obstacle_table,
                    instance,
                    stepLength
            );
            if (!success) {
                return false;
            }

            groupAgentMap[groupIdx] = groupAgent;
            return true;
        }

        void add_GroupAgent(size_t groupIdx) {
            std::shared_ptr<GroupAstarSolver> groupAgent = std::make_shared<GroupAstarSolver>();
            groupAgentMap[groupIdx] = nullptr;
            groupAgentMap[groupIdx] = groupAgent;
        }

        std::vector<Path_XYZRL> getGroupAgentResPath(size_t groupIdx){
            std::vector<Path_XYZRL> res;
            for (TaskInfo* iter: groupAgentMap[groupIdx]->taskTree) {
                res.emplace_back(Path_XYZRL(iter->res_path));
            }
            return res;
        }

        void addTask_to_GroupAgent(size_t groupIdx, size_t loc0, double radius0, size_t loc1, double radius1) {
            groupAgentMap[groupIdx]->addTask(loc0, radius0, loc1, radius1);
        }

        void copy(CBSNode *rhs) {
            for (auto iter: rhs->groupAgentMap) {
                groupAgentMap[iter.first] = std::shared_ptr<GroupAstarSolver>(iter.second);
            }
            for (auto iter: rhs->constrainsMap) {
                constrainsMap[iter.first] = std::shared_ptr<std::vector<ConstrainType>>(iter.second);
            }
            rectangleExcludeAreas.clear();
            rectangleExcludeAreas = std::vector<std::tuple<double, double, double, double, double, double>>(rhs->rectangleExcludeAreas);
        }

        inline double getFVal() const {
            return g_val + h_val;
        }

        void findFirstPipeConflict();

        void compute_Heuristics();

        void compute_Gval();

        std::vector<ConstrainType> getConstrains(size_t groupIdx) {
            int useCount = constrainsMap[groupIdx].use_count();
            assert(("Constrain Must Be Exist", useCount > 0));
            return std::vector<ConstrainType>(*(constrainsMap[groupIdx]));
        }

        void info(bool with_constrainInfo = false, bool with_pathInfo = false) {
            for (auto groupAgent_iter: groupAgentMap) {
                std::cout << "GroupIdx:" << groupAgent_iter.first << std::endl;

                if (with_constrainInfo) {
                    std::cout << "  constrain size:" << constrainsMap[groupAgent_iter.first]->size() << std::endl;
                }

                if (with_pathInfo) {
                    for (auto task: groupAgent_iter.second->taskTree) {
                        std::cout << "    radius0:" << task->radius0 << std::endl;
                        std::cout << "    radius1:" << task->radius1 << std::endl;
                        std::cout << "    res_path size:" << task->res_path.size() << std::endl;
                    }
                }
            }
        }

        struct compare_node {
            bool operator()(const CBSNode *n1, const CBSNode *n2) const {
                return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
            }
        };

        // TODO need to change to share_ptr
        std::vector<std::tuple<double, double, double, double, double, double>> rectangleExcludeAreas;

        void add_rectangleExcludeArea(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax){
            rectangleExcludeAreas.emplace_back(std::make_tuple(xmin, ymin, zmin, xmax, ymax, zmax));
        }

        void clear_rectangleExcludeArea(){
            rectangleExcludeAreas.clear();
        }

        bool isIn_rectangleExcludeAreas(double x, double y, double z){
            if (rectangleExcludeAreas.size() == 0){
                return false;
            }

            double xmin, ymin, zmin, xmax, ymax, zmax;
            for (int i = 0; i < rectangleExcludeAreas.size(); ++i) {
                std::tie(xmin, ymin, zmin, xmax, ymax, zmax) = rectangleExcludeAreas[i];

                if (x < xmin){
                    continue;
                } else if (x > xmax){
                    continue;
                } else if (y < ymin){
                    continue;
                } else if (y > ymax){
                    continue;
                } else if (z < zmin){
                    continue;
                } else if (z > zmax){
                    continue;
                }

                return true;
            }
            return false;
        }

    private:
        double stepLength;

        void release() {
            for (auto iter: groupAgentMap) {
                iter.second = nullptr;
            }
            groupAgentMap.clear();

            for (auto iter: constrainsMap) {
                iter.second = nullptr;
            }
            constrainsMap.clear();
            pipe_conflictLength.clear();
            rectangleExcludeAreas.clear();
        }
    };

}

#endif