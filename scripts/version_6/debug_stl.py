import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build import mapf_pipeline

def debug_constrainTable():
    constrainTable = mapf_pipeline.ConstraintTable()

    constrainTable.insert2CT((1.0, 1.0, 0.0, 0.5))
    constrainTable.insert2CT(2.0, 2.0, 0.0, 0.5)

    # isConflict = constrainTable.isConstrained(x=2.5, y=2.0, z=0.0, radius=0.1)
    isConflict = constrainTable.isConstrained(
        lineStart_x=0.0, lineStart_y=0.0, lineStart_z=0.0,
        lineEnd_x=0.0, lineEnd_y=2.0, lineEnd_z=0.0, radius=0.1
    )
    print(isConflict)

def debug_sampleDetailPath():
    path = [
        (0.0, 0.0, 0.0, 0.5),
        (0.0, 2.0, 0.0, 1.0),
        (3.0, 2.0, 0.0, 0.5),
    ]
    path = mapf_pipeline.sampleDetailPath(path_xyzr=path, stepLength=0.5)
    print(path)

def debug_conflict():
    conflict = mapf_pipeline.Conflict(0, 1.0, 1.0, 1.0, 0.5, 1, 2.0, 2.0, 2.0, 0.5)
    conflict.conflictExtend()
    print(conflict.constrain1)
    print(conflict.constrain2)

def debug_astar():
    instance = mapf_pipeline.Instance(5, 5, 1)
    start_loc = instance.linearizeCoordinate(x=3, y=0, z=0)
    # goal_locs = [instance.linearizeCoordinate(x=0, y=4, z=0)]
    goal_locs = [
        instance.linearizeCoordinate(x=1, y=1, z=0),
        instance.linearizeCoordinate(x=2, y=4, z=0)
    ]
    goal_xy = []
    for goal_loc in goal_locs:
        (x, y, z) = instance.getCoordinate(goal_loc)
        goal_xy.append([x, y])
    goal_xy = np.array(goal_xy)
    print(goal_xy)

    xs, ys = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    astar_solver = mapf_pipeline.AStarSolver(False, True)
    pathIdxs = astar_solver.findPath(
        radius=0.5, constraints=[], instance=instance, start_loc=start_loc, goal_locs=goal_locs
    )
    path_xy = []
    for idx in pathIdxs:
        (x, y, z) = instance.getCoordinate(idx)
        path_xy.append([x, y])
    path_xy = np.array(path_xy)

    plt.plot(path_xy[:, 0], path_xy[:, 1], '*-', c='r')
    plt.scatter(goal_xy[:, 0], goal_xy[:, 1], s=30.0, c='g')
    plt.show()

def debug_multiSolver():
    instance = mapf_pipeline.Instance(10, 10, 1)
    astarSolver = mapf_pipeline.AStarSolver(False, True)
    multiSolver = mapf_pipeline.MultiObjs_GroupSolver()

    locs_xyz = np.array([
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0]
    ])
    locs = []
    for (x, y, z) in locs_xyz:
        locs.append(instance.linearizeCoordinate(x, y, z))

    # links = multiSolver.getSequence_miniumSpanningTree(instance, locs)
    multiSolver.insert_objs(locs, radius_list=[0.5, 0.5, 0.5, 0.5], instance=instance)
    success = multiSolver.findPath(astarSolver, constraints=[], instance=instance, stepLength=0.5)
    print(success)

    for obj in multiSolver.objectiveMap:
        print('PathIdx: ',obj.pathIdx)
        print('  start_loc:', obj.start_loc)
        print('  start_xyz:', instance.getCoordinate(obj.start_loc))
        print('  goal_locs:', obj.goal_locs)
        print('  radius:', obj.radius)
        print('  fixed_end:', obj.fixed_end)
        print('  res_path:', obj.res_path)

    xs, ys = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    for obj in multiSolver.objectiveMap:
        res_path = np.array(obj.res_path)

        if res_path.shape[0] > 0:
            plt.plot(res_path[:, 0], res_path[:, 1], '*-', c='r')

    plt.scatter(locs_xyz[:, 0], locs_xyz[:, 1], s=20.0, c='g')

    plt.show()

print('Finish')