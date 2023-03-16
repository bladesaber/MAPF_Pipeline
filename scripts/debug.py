from build import mapf_pipeline

import numpy as np
import pandas as pd
import random
from typing import List
import matplotlib.pyplot as plt
import open3d

# debug pring functions
# mapf_pipeline.testPring_vector([1, 2, 3])
# mapf_pipeline.testPring_list([1, 2, 3])
# mapf_pipeline.testPring_map({'a': 10, 'b':20})
# mapf_pipeline.testPring_pair(('a', 10))
# mapf_pipeline.testPring_tuple(('a', 10))

# debug Instance
# instance = mapf_pipeline.Instance(30, 25)
# instance.print()
# curr = np.random.randint(0, instance.map_size)
# print("curr: %f" % (curr))
# row = instance.getRowCoordinate(curr)
# col = instance.getColCoordinate(curr)
# print("row: %f col: %f" % (row, col))
# (row, col) = instance.getCoordinate(curr)
# print("loc:%f, row:%f, col:%f" % (curr, row, col))
# neighbours = instance.getNeighbors(curr)
# print("neighbours", neighbours)
# for neighbour in neighbours:
#     (row, col) = instance.getCoordinate(neighbour)
#     print('neighbour:%f, row:%f, col:%f' % (neighbour, row, col))
# curr = instance.linearizeCoordinate(row, col)
# print("curr: %f" % (curr))
# curr = instance.linearizeCoordinate((row, col))
# print("curr: %f" % (curr))

# debug ConstraintTable
# constraint_table = mapf_pipeline.ConstraintTable()
# constraint_table.insert2CT(10)
# constraint_table.insert2CT(20)
# constraint_table.insert2CAT(30)
# constraint_table.insert2CAT(30)
# constraint_table.insert2CAT(40)
# constraints = [
#     (0, 1, 1, mapf_pipeline.constraint_type.VERTEX),
#     (0, 2, 2, mapf_pipeline.constraint_type.VERTEX),
#     (0, 3, 3, mapf_pipeline.constraint_type.VERTEX),
# ]
# constraint_table.insertConstrains2CT(constraints)
# path = [1, 2, 3, 4]
# constraint_table.insertPath2CAT(path)
# path = [3, 4, 5, 6]
# constraint_table.insertPath2CAT(path)
# ct = constraint_table.getCT()
# print(ct)
# cat = constraint_table.getCAT()
# print(cat)

# debug SpaceTimeAStar
# num_rows, num_cols = 30, 30
# xs, ys = np.meshgrid(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1))
# map = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
# map = map.reshape((-1, 2))

# instance = mapf_pipeline.Instance(num_rows, num_cols)
# astar = mapf_pipeline.SpaceTimeAStar(0)
# start_yx = (random.randint(0, num_rows), random.randint(0, num_cols))
# goal_yx = (random.randint(0, num_rows), random.randint(0, num_cols))
# print(start_yx, goal_yx)
# path: List = astar.findPath(paths={}, constraints={}, instance=instance, start_state=start_yx, goal_state=goal_yx)
# print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
# print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT))
# paths_xy = []
# for loc in path:
#     (row, col) = instance.getCoordinate(loc)
#     paths_xy.append([col, row])
# paths_xy = np.array(paths_xy)

# plt.scatter(map[:, 0], map[:, 1], s=2.0)
# plt.scatter([start_yx[1], goal_yx[1]], [start_yx[0], goal_yx[0]], s=10.0, c='g')
# plt.plot(paths_xy[:, 0], paths_xy[:, 1], c='r')
# plt.show()

# debug SpaceTimeAStar 3D
num_rows, num_cols, num_z = 20, 20, 20
instance = mapf_pipeline.Instance3D(num_rows, num_cols, num_z)
start_yxz = (random.randint(0, num_rows), random.randint(0, num_cols), random.randint(0, num_z))
goal_yxz = (random.randint(0, num_rows), random.randint(0, num_cols), random.randint(0, num_z))

astar = mapf_pipeline.SpaceTimeAStar(0)
print("Starting ...")
path: List = astar.findPath(paths={}, constraints={}, instance=instance, start_state=start_yxz, goal_state=goal_yxz)
print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT))
# paths_xyz = []
# for loc in path:
#     (row, col, z) = instance.getCoordinate(loc)
#     paths_xyz.append([col, row, z])
# paths_xy = np.array(paths_xyz)
# print(paths_xyz)

