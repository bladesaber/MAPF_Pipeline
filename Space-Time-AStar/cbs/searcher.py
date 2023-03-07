import numpy as np
from typing import Dict, List, Set, Tuple
import multiprocessing as mp
from heapq import heappush, heappop
from itertools import combinations
from copy import deepcopy

from astar.sapce_time_astar import STPlanner
from cbs.utils import CTNode, Constraints, Agent


class CBS_Planner(object):
    def __init__(
            self,
            grid_xmax, grid_ymax,
            robot_radius,
            starts, goals,
    ):
        self.robot_radius = robot_radius
        self.st_planner = STPlanner(grid_xmax, grid_ymax, robot_radius)

        self.agents = []
        for start, goal in zip(starts, goals):
            self.agents.append(Agent(start, goal))

    def plan(
            self,
            max_iter: int = 200,
            low_level_max_iter: int = 100,
            max_process: int = 10,
    ) -> np.ndarray:
        self.low_level_max_iter = low_level_max_iter

        constraints = Constraints()

        solutions = {}
        for agent in self.agents:
            path = self.st_planner.plan(
                agent.start,
                agent.goal,
                constraints.setdefault(agent, dict()),
                terminate_obstacles={},
                max_iter=self.low_level_max_iter,
                debug=True
            )
            solutions[agent] = path

        print('------------------------------')

        open = []
        if all(len(path) != 0 for path in solutions.values()):
            node = CTNode(constraints, solutions)
            open.append(node)

        manager = mp.Manager()
        iter_ = 0
        while open and iter_ < max_iter:
            iter_ += 1

            results = manager.list([])
            processes = []

            for _ in range(max_process if len(open) > max_process else len(open)):
                p = mp.Process(target=self.search_node, args=[heappop(open), results])
                processes.append(p)
                p.start()

                for p in processes:
                    p.join()

                for result in results:
                    if len(result) == 1:
                        return result[0]

                    if result[0]:
                        heappush(open, result[0])

                    if result[1]:
                        heappush(open, result[1])

        return np.array([])

    def search_node(self, best: CTNode, results):
        agent_i, agent_j, time_of_conflict = self.validate_paths(self.agents, best)

        if agent_i is None:
            results.append((best.solution, ))
            print('[Debug]: Success find')
            return

        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict)

        agent_i_path = self.st_planner.plan(
            agent_i.start,
            agent_i.goal,
            object_obstacles=agent_i_constraint[agent_i],
            terminate_obstacles=self.calculate_goal_times(best, agent_i, self.agents),
            max_iter=self.low_level_max_iter,
            debug=True
        )

        if agent_i_path.shape[0] == 0:
            node_i = None
        else:
            solution_i = deepcopy(best.solution)
            solution_i[agent_i] = agent_i_path
            node_i = CTNode(agent_i_constraint, solution_i)

        agent_j_path = self.st_planner.plan(
            agent_j.start,
            agent_j.goal,
            agent_j_constraint[agent_j],
            terminate_obstacles=self.calculate_goal_times(best, agent_j, self.agents),
            max_iter=self.low_level_max_iter,
        )
        if agent_j_path.shape[0] == 0:
            node_j = None
        else:
            solution_j = deepcopy(best.solution)
            solution_j[agent_j] = agent_j_path
            node_j = CTNode(agent_j_constraint, solution_j)

        results.append((node_i, node_j))

    ### ----------------------------------------------
    def validate_paths(self, agents, node: CTNode):
        for agent_i, agent_j in combinations(agents, 2):
            time_of_conflict = self.safe_distance(node.solution, agent_i, agent_j)
            if time_of_conflict == -1:
                continue

            return agent_i, agent_j, time_of_conflict
        return None, None, -1

    def safe_distance(self, solution: Dict[Agent, np.ndarray], agent_i: Agent, agent_j: Agent) -> int:
        for idx, (point_i, point_j) in enumerate(zip(solution[agent_i], solution[agent_j])):
            dist = self.dist(point_i, point_j)
            if dist > 2 * self.robot_radius:
                continue

            return idx
        return -1

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray):
        return np.linalg.norm(point1 - point2, 2)  # L2 norm

    ### ------------------------------------------------------
    def calculate_constraints(
            self,
            node: CTNode,
            constrained_agent: Agent,
            unchanged_agent: Agent,
            time_of_conflict: int
    ) -> Constraints:
        contrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]

        pivot = unchanged_path[time_of_conflict]
        conflict_end_time = time_of_conflict

        dist = self.dist(contrained_path[conflict_end_time], pivot)
        try:
            while dist <= 2 * self.robot_radius:
                conflict_end_time += 1
                dist = self.dist(contrained_path[conflict_end_time], pivot)
        except IndexError:
            pass

        new_constraints = node.constraints.fork(
            constrained_agent, tuple(pivot.tolist()),
            time_of_conflict, conflict_end_time
        )
        return new_constraints

    def calculate_goal_times(self, node: CTNode, agent: Agent, agents: List[Agent]):
        solution = node.solution
        goal_times = dict()
        for other_agent in agents:
            if other_agent == agent:
                continue
            time = len(solution[other_agent]) - 1

            goal_times.setdefault(time, set()).add(tuple(solution[other_agent][time]))

        return goal_times

# if __name__ == '__main__':
#     planner = CBS_Planner(
#         grid_xmax=40, grid_ymax=40, robot_radius=2.0,
#         starts=np.array([
#             [5, 5],
#             [35, 5],
#             [5, 35],
#             [35, 35]
#         ]),
#         goals=np.array([
#             [35, 35],
#             [5, 35],
#             [35, 5],
#             [5, 5]
#         ])
#     )
#     solutions = planner.plan(max_process=4)
#
#     save_paths = []
#     for agent in planner.agents:
#         path = solutions[agent]
#         save_paths.append(path)
#     np.save(
#         'D:/gittest/reference/Space-Time-AStar-master/save_dir/path', np.array(save_paths)
#     )
