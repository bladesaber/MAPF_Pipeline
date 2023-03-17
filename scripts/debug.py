from build import mapf_pipeline

import numpy as np
import pandas as pd
import random
from typing import List
import matplotlib.pyplot as plt
import open3d

def debug_pybind():
    mapf_pipeline.testPring_vector([1, 2, 3])
    mapf_pipeline.testPring_list([1, 2, 3])
    mapf_pipeline.testPring_map({'a': 10, 'b':20})
    mapf_pipeline.testPring_pair(('a', 10))
    mapf_pipeline.testPring_tuple(('a', 10))

def debug_instance2D():
    instance = mapf_pipeline.Instance(30, 30)
    instance.print()
    curr = np.random.randint(0, instance.map_size)
    print("curr: %f" % (curr))
    row = instance.getRowCoordinate(curr)
    col = instance.getColCoordinate(curr)
    print("row: %f col: %f" % (row, col))
    (row, col) = instance.getCoordinate(curr)
    print("loc:%f, row:%f, col:%f" % (curr, row, col))
    neighbours = instance.getNeighbors(curr)
    print("neighbours", neighbours)
    for neighbour in neighbours:
        (row, col) = instance.getCoordinate(neighbour)
        print('neighbour:%f, row:%f, col:%f' % (neighbour, row, col))
    curr = instance.linearizeCoordinate(row, col)
    print("curr: %f" % (curr))
    curr = instance.linearizeCoordinate((row, col))
    print("curr: %f" % (curr))

def debug_ConstraintTable():
    constraint_table = mapf_pipeline.ConstraintTable()
    constraint_table.insert2CT(10)
    constraint_table.insert2CT(20)
    constraint_table.insert2CAT(30)
    constraint_table.insert2CAT(30)
    constraint_table.insert2CAT(40)
    constraints = [
        (0, 1, 1, mapf_pipeline.constraint_type.VERTEX),
        (0, 2, 2, mapf_pipeline.constraint_type.VERTEX),
        (0, 3, 3, mapf_pipeline.constraint_type.VERTEX),
    ]
    constraint_table.insertConstrains2CT(constraints)
    path = [1, 2, 3, 4]
    constraint_table.insertPath2CAT(path)
    path = [3, 4, 5, 6]
    constraint_table.insertPath2CAT(path)
    ct = constraint_table.getCT()
    print(ct)
    cat = constraint_table.getCAT()
    print(cat)

def debug_instance3D():
    instance = mapf_pipeline.Instance3D(30, 30, 30)

    start_yxz, goal_yxz = (21, 0, 12), (19, 0, 19)
    start_loc = instance.linearizeCoordinate(start_yxz)
    goal_loc = instance.linearizeCoordinate(goal_yxz)

    h_val = instance.getManhattanDistance(start_loc, goal_loc)
    print(h_val)
    h_val = instance.getManhattanDistance(start_yxz, goal_yxz)
    print(h_val)

if __name__ == '__main__':
    debug_instance3D()