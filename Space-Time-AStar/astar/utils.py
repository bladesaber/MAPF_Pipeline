from typing import Tuple
import numpy as np

class Grid(object):
    directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, grid_xmax, grid_ymax):
        ys, xs = np.meshgrid(np.arange(0, grid_xmax), np.arange(0, grid_ymax))
        self.grid = np.concatenate((xs[:, :, np.newaxis], ys[:, :, np.newaxis]), axis=-1)

        self.table = {}
        for y in range(grid_ymax):
            for x in range(grid_xmax):
                neighbours = []
                for dx, dy in self.directions:
                    new_y, new_x = y + dy, x + dx,
                    if new_x >= 0 and new_x < grid_xmax and new_y >= 0 and new_y < grid_ymax:
                        neighbours.append(self.grid[new_y, new_x])
                self.table[tuple(self.grid[y, x])] = np.array(neighbours)

class State:

    def __init__(self, pos: np.ndarray, time: int, g_score, h_score):
        self.pos = pos
        self.time = time
        self.g_score = g_score
        self.f_score = g_score + h_score

    def __hash__(self) -> int:
        concat = str(self.pos[0]) + str(self.pos[1]) + '0' + str(self.time)
        return int(concat)

    def pos_equal_to(self, pos: np.ndarray) -> bool:
        return np.array_equal(self.pos, pos)

    def __lt__(self, other: 'State') -> bool:
        return self.f_score < other.f_score

    def __eq__(self, other: 'State') -> bool:
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return 'State(pos=[' + str(self.pos[0]) + ', ' + str(self.pos[1]) + '], ' \
               + 'time=' + str(self.time) + ', fscore=' + str(self.f_score) + ')'

    def __repr__(self):
        return self.__str__()


