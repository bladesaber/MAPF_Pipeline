import numpy as np
from typing import Tuple, List, Dict, Set
from heapq import heappush, heappop

from astar.utils import Grid, State

class STPlanner(object):
    def __init__(
            self,
            grid_xmax, grid_ymax,
            robot_radius,
    ):
        self.grid = Grid(grid_xmax, grid_ymax)
        self.neighbour_table = self.grid.table
        self.robot_radius = robot_radius

    @staticmethod
    def h(start: np.ndarray, goal: np.ndarray):
        return np.linalg.norm(start - goal, 1)  # L1 norm

    @staticmethod
    def l2(start: np.ndarray, goal: np.ndarray):
        return np.linalg.norm(start - goal, 2)  # L2 norm

    def plan(
            self,
            start, goal,
            object_obstacles: Dict[int, Set[Tuple[int, int]]],
            terminate_obstacles: Dict[int, Set[Tuple[int, int]]],
            max_iter: int = 500,
            debug: bool = False
    ) -> np.ndarray:
        dynamic_obstacles = {}
        for time_key, obstacle_v in object_obstacles.items():
            ### obstacle_v is set
            dynamic_obstacles[time_key] = np.array(list(obstacle_v))

        def dynamic_check(grid_pos: np.ndarray, time: int) -> bool:
            conflict_list = []
            for obstacle in dynamic_obstacles.setdefault(time, np.array([])):
                is_conflict = self.l2(grid_pos, obstacle) > 2 * self.robot_radius
                conflict_list.append(is_conflict)
            return all(conflict_list)

        def terminate_check(grid_pos: np.ndarray, time: int) -> bool:
            nonlocal terminate_obstacles
            for timestamp, obstacles in terminate_obstacles.items():
                flag = True
                if time >= timestamp:
                    flag = all(self.l2(grid_pos, obstacle) > 2 * self.robot_radius for obstacle in obstacles)
                if not flag:
                    return False
            return True

        s = State(start, 0, 0, self.h(start, goal))
        open_set = [s]
        closed_set = set()
        came_from = {}

        iter_ = 0
        while open_set and iter_ < max_iter:
            iter_ += 1

            current_state = open_set[0]  # Smallest element in min-heap
            if current_state.pos_equal_to(goal):
                if debug:
                    print('STA*: Path found after {0} iterations'.format(iter_))
                return self.reconstruct_path(came_from, current_state)

            closed_set.add(heappop(open_set))
            epoch = current_state.time + 1

            for neighbour in self.neighbour_table[tuple(current_state.pos)]:
                neighbour_state = State(neighbour, epoch, current_state.g_score + 1, self.h(neighbour, goal))

                if neighbour_state in closed_set:
                    continue

                # Avoid obstacles
                if not dynamic_check(neighbour, epoch) or not terminate_check(neighbour, epoch):
                    continue

                # Add to open set
                if neighbour_state not in open_set:
                    came_from[neighbour_state] = current_state
                    heappush(open_set, neighbour_state)

        if debug:
            print('STA*: Open set is empty, no path found.')
        return np.array([])

    def reconstruct_path(self, came_from: Dict[State, State], current: State) -> np.ndarray:
        total_path = [current.pos]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current.pos)
        return np.array(total_path[::-1])
