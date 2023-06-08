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
    start_locs = [
        instance.linearizeCoordinate(x=3, y=0, z=0),
        instance.linearizeCoordinate(x=1, y=0, z=0)
    ]
    goal_locs = [
        instance.linearizeCoordinate(x=0, y=4, z=0),
        instance.linearizeCoordinate(x=3, y=3, z=0),
    ]

    ### --------------------------------
    goal_xys = []
    for loc in goal_locs:
        (x, y, z) = instance.getCoordinate(loc)
        goal_xys.append([x, y])
    goal_xys = np.array(goal_xys)

    start_xys = []
    for loc in start_locs:
        (x, y, z) = instance.getCoordinate(loc)
        start_xys.append([x, y])
    start_xys = np.array(start_xys)
    ### -------------------------------


    xs, ys = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    astar_solver = mapf_pipeline.AStarSolver(False, True)
    pathIdxs = astar_solver.findPath(
        radius=0.5, 
        constraints=[
            # (3, 1, 0, 0.0),
            # (2, 0, 0, 0.0)
        ], 
        instance=instance, start_locs=start_locs, goal_locs=goal_locs
    )
    print('Time Cost: ', astar_solver.runtime_search)

    path_xy = []
    for idx in pathIdxs:
        (x, y, z) = instance.getCoordinate(idx)
        path_xy.append([x, y])
    path_xy = np.array(path_xy)

    plt.plot(path_xy[:, 0], path_xy[:, 1], '*-', c='b')
    plt.scatter(goal_xys[:, 0], goal_xys[:, 1], s=30.0, c='g')
    plt.scatter(start_xys[:, 0], start_xys[:, 1], s=30.0, c='r')
    plt.show()

def debug_groupSolver():
    instance = mapf_pipeline.Instance(10, 10, 1)
    astarSolver = mapf_pipeline.AStarSolver(False, True)
    groupSolver = mapf_pipeline.SpanningTree_GroupSolver()

    locs_xyz = np.array([
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0]
    ])
    print(locs_xyz)
    pipeMap = {}
    for (x, y, z) in locs_xyz:
        loc = instance.linearizeCoordinate(x, y, z)
        pipeMap[loc] = 0.5

    ### ------------------------------------------------
    groupSolver.insertPipe(pipeMap, instance=instance)
    # for task in groupSolver.task_seq:
    #     print(task.link_sign0, task.link_sign1)
    #     # print(task.radius0, task.radius1)

    success = groupSolver.findPath(
        astarSolver, 
        constraints=[], 
        instance=instance, 
        stepLength=1.0
    )
    print(success)

    xs, ys = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    for task in groupSolver.task_seq:
        res_path = np.array(task.res_path)

        if res_path.shape[0] > 0:
            plt.plot(res_path[:, 0], res_path[:, 1], '*-', c='g')

    plt.scatter(locs_xyz[:, 0], locs_xyz[:, 1], s=60.0, c='r')

    plt.show()

def debug_spanningTree():
    instance = mapf_pipeline.Instance(10, 10, 1)
    multiSolver = mapf_pipeline.SpanningTree_GroupSolver()

    locs_xyz = np.array([
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0],
        [np.random.randint(0, 10, 1)[0], np.random.randint(0, 10, 1)[0], 0]
    ])
    # locs_xyz = np.array([
    #     [8, 7, 0],
    #     [0, 5, 0],
    #     [5, 5, 0],
    #     [0, 6, 0]
    # ])
    locs = []
    for (x, y, z) in locs_xyz:
        locs.append(instance.linearizeCoordinate(x, y, z))

    links = multiSolver.getSequence_miniumSpanningTree(instance, locs)
    print(locs)
    print(locs_xyz)
    print(links)

debug_groupSolver()

print('Finish')