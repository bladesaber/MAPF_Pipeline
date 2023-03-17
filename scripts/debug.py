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
instance = mapf_pipeline.Instance(30, 30)
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
    # (row, col) = instance.getCoordinate(neighbour)
    # print('neighbour:%f, row:%f, col:%f' % (neighbour, row, col))
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

#----------------- debug SpaceTimeAStar
load = True

num_rows, num_cols = 30, 30
xs, ys = np.meshgrid(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1))
instance = mapf_pipeline.Instance(num_rows, num_cols)
astar = mapf_pipeline.SpaceTimeAStar(0)

if load:
    map = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/map.npy', allow_pickle=True)
else:
    map = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
    map = map.reshape((-1, 2))

map_idx = np.arange(0, map.shape[0], 1)
if load:
    obs_idx = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/obs_idx.npy', allow_pickle=True)
else:
    obs_idx = np.random.choice(map_idx, size=int(map.shape[0]*0.25))
    np.save('/home/quan/Desktop/MAPF_Pipeline/scripts/map', map)
    np.save('/home/quan/Desktop/MAPF_Pipeline/scripts/obs_idx', obs_idx)

remain_idx = np.setdiff1d(map_idx, obs_idx)
start_idx, goal_idx = np.random.choice(remain_idx, size=2)
if load:
    start_yx = (24, 28)
    goal_yx = (28, 0)
else:
    start_yx = (map[start_idx][1], map[start_idx][0])
    goal_yx = (map[goal_idx][1], map[goal_idx][0])
print(start_yx, goal_yx)

if load:
    constrains_table = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/constrains_table.npy', allow_pickle=True).item()
else:
    obs_constrains = []
    for idx in obs_idx:
        x, y = map[idx, :]
        obs_constrains.append(
            # agent_idx, loc, timestep, type
            (0, instance.linearizeCoordinate((y, x)), 0, mapf_pipeline.constraint_type.VERTEX)
        )
    constrains_table = {
        0: obs_constrains
    }
    np.save('/home/quan/Desktop/MAPF_Pipeline/scripts/constrains_table', constrains_table)

path: List = astar.findPath(paths={}, constraints=constrains_table, instance=instance, start_state=start_yx, goal_state=goal_yx)
print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT))
print(path)
paths_xy = []
for loc in path:
    (row, col) = instance.getCoordinate(loc)
    paths_xy.append([col, row])
paths_xy = np.array(paths_xy)

plt.scatter(map[remain_idx, 0], map[remain_idx, 1], s=1.0)
plt.scatter(map[obs_idx, 0], map[obs_idx, 1], s=10.0, c='r')
plt.scatter([start_yx[1], goal_yx[1]], [start_yx[0], goal_yx[0]], s=20.0, c='g')
if paths_xy.shape[0]>0:
    plt.plot(paths_xy[:, 0], paths_xy[:, 1], c='r')
plt.show()

# #----------------- debug SpaceTimeAStar 3D
# num_rows, num_cols, num_z = 5, 5, 5
# instance = mapf_pipeline.Instance3D(num_rows, num_cols, num_z)

# # curr = np.random.randint(0, instance.map_size)
# # curr = 70
# # print("curr: %f" % (curr))
# # row = instance.getRowCoordinate(curr)
# # col = instance.getColCoordinate(curr)
# # z = instance.getZCoordinate(curr)
# # print("row:%f, col:%f, Z:%f" % (row, col, z))
# # (row, col, z) = instance.getCoordinate(curr)
# # print("loc:%f, row:%f, col:%f, Z:%f" % (curr, row, col, z))
# # instance.printCoordinate(curr)
# # neighbours = instance.getNeighbors(curr)
# # print("neighbours", neighbours)
# # for neighbour in neighbours:
# #     (row, col, z) = instance.getCoordinate(neighbour)
# #     print('neighbour:%f, row:%f, col:%f Z:%f' % (neighbour, row, col, z))
# #     instance.printCoordinate(neighbour)

# # curr = instance.linearizeCoordinate(row, col, z)
# # print("curr: %f" % (curr))
# # curr = instance.linearizeCoordinate((row, col, z))
# # print("curr: %f" % (curr))

# # start_yxz = (random.randint(0, num_rows), random.randint(0, num_cols), random.randint(0, num_z))
# # goal_yxz = (random.randint(0, num_rows), random.randint(0, num_cols), random.randint(0, num_z))
# start_yxz = (1, 5, 2)
# goal_yxz = (3, 1, 1)
# print("start node: ", start_yxz, " goal node: ", goal_yxz)
# print("start node: ", instance.linearizeCoordinate(start_yxz), " goal node: ", instance.linearizeCoordinate(goal_yxz))

# astar = mapf_pipeline.SpaceTimeAStar(0)
# print("Starting ...")
# path: List = astar.findPath(paths={}, constraints={}, instance=instance, start_state=start_yxz, goal_state=goal_yxz)
# # print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
# # print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT))
# # print(path)
# # paths_xyz = []
# # for loc in path:
# #     (row, col, z) = instance.getCoordinate(loc)
# #     paths_xyz.append([col, row, z])
# # paths_xy = np.array(paths_xyz)
# # print(paths_xyz)

