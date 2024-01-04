import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline
from ..mapf_pipeline_py.spanTree_TaskAllocator import MiniumSpanTree_TaskRunner
from ..mapf_pipeline_py.spanTree_TaskAllocator import MiniumDistributeTree_TaskRunner

def debug_constrainTable():
    constrainTable = mapf_pipeline.ConstraintTable()

    # constrainTable.insert2CT((1.0, 1.0, 0.0, 0.5))
    constrainTable.insert2CT(1.0, 1.0, 0.0, 0.5)
    # constrainTable.insert2CT(2.0, 2.0, 0.0, 0.5)

    isConflict = constrainTable.isConstrained(x=1.0, y=1.0, z=0.0, radius=0.5)
    # isConflict = constrainTable.isConstrained(
    #     lineStart_x=2.0, lineStart_y=2.0, lineStart_z=0.0,
    #     lineEnd_x=3.0, lineEnd_y=3.0, lineEnd_z=0.0, radius=0.1
    # )
    print(isConflict)

def debugAstar():
    num_of_x, num_of_y, num_of_z = 10, 10, 1
    instance = mapf_pipeline.Instance(num_of_x, num_of_y, num_of_z)

    start_xyzs = np.array([
        [1, 0, 0]
    ])
    goal_xyzs = np.array([
        [5, 5, 0]
    ])

    start_locs = []
    for x, y, z in start_xyzs:
        loc = instance.linearizeCoordinate(x, y, z)
        start_locs.append(loc)

    goal_locs = []
    for x, y, z in goal_xyzs:
        loc = instance.linearizeCoordinate(x, y, z)
        goal_locs.append(loc)

    xs, ys = np.meshgrid(np.arange(0, num_of_x, 1), np.arange(0, num_of_y, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    constrain_table = mapf_pipeline.ConstraintTable()
    obstacle_table = mapf_pipeline.ConstraintTable()

    obstacle_xyzr = np.array([
        [1, 5, 0, 0.5],
    ])
    for x, y, z, radius in obstacle_xyzr:
        obstacle_table.insert2CT(x, y, z, radius)

    solver = mapf_pipeline.AStarSolver(with_AnyAngle=False, with_OrientCost=True)
    pathIdxs = solver.findPath(
        radius=0.5,
        constraint_table=constrain_table,
        obstacle_table=obstacle_table,
        instance=instance,
        start_locs=start_locs,
        goal_locs=goal_locs
    )
    print('PathSize: ', len(pathIdxs))

    if len(pathIdxs) != 0:
        path_xy = []
        for idx in pathIdxs:
            (x, y, z) = instance.getCoordinate(idx)
            path_xy.append([x, y])
        path_xy = np.array(path_xy)

        plt.plot(path_xy[:, 0], path_xy[:, 1], '*-', c='b')

    plt.scatter(goal_xyzs[:, 0], goal_xyzs[:, 1], s=30.0, c='g')
    plt.scatter(start_xyzs[:, 0], start_xyzs[:, 1], s=30.0, c='r')
    plt.scatter(obstacle_xyzr[:, 0], obstacle_xyzr[:, 1], s=60.0, c='b')
    plt.show()

def debugGroupAstar():
    num_of_x, num_of_y, num_of_z = 10, 10, 1
    instance = mapf_pipeline.Instance(num_of_x, num_of_y, num_of_z)

    loc_xyzs = np.random.randint(0, 10, size=(10, 3))
    loc_xyzs[:, 2] = 0
    # loc_xyzs = np.array([
    #     [0, 3, 0],
    #     [3, 0, 0],
    #     [6, 8, 0],
    #     [3, 9, 0],
    # ])
    print(loc_xyzs)

    locs = []
    for x, y, z in loc_xyzs:
        loc = instance.linearizeCoordinate(x, y, z)
        locs.append(loc)

    solver = mapf_pipeline.GroupAstarSolver()

    allocator = MiniumDistributeTree_TaskRunner()
    for i, xyz in enumerate(loc_xyzs):
        allocator.add_node(i, xyz)
    res_list = allocator.getTaskTrees()

    taskTrees = []
    for i, j in res_list:
        taskTrees.append({
            'loc0': locs[i], 'radius0': 0.5, 'loc1': locs[j], 'radius1': 0.5
        })

    # taskTrees = [
    #     {'loc0': locs[0], 'radius0': 0.5, 'loc1': locs[1], 'radius1': 0.5},
    #     {'loc0': locs[2], 'radius0': 0.5, 'loc1': locs[3], 'radius1': 0.5},
    #     {'loc0': locs[1], 'radius0': 0.5, 'loc1': locs[2], 'radius1': 0.5},
    # ]
    for task in taskTrees:
        solver.addTask(**task)

    obstacle_table = mapf_pipeline.ConstraintTable()
    constraints = [
        # (3, 1, 0, 0.0),
        # (2, 0, 0, 0.0)
    ]

    cell_solver = mapf_pipeline.AStarSolver(with_AnyAngle=False, with_OrientCost=True)
    status = solver.findPath(
        solver=cell_solver, constraints=constraints, obstacle_table=obstacle_table, instance=instance, stepLength=1.0
    )
    print('Status:', status)

    ### ------ Vis
    xs, ys = np.meshgrid(np.arange(0, num_of_x, 1), np.arange(0, num_of_y, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')
    colors = np.random.uniform(size=(len(taskTrees), 3))
    if status:
        for i, taskInfo in enumerate(solver.taskTree):
            path_xyzrl = np.array(taskInfo.res_path)
            print('pathSize: ', path_xyzrl.shape[0])
            plt.plot(path_xyzrl[:, 0], path_xyzrl[:, 1], '*-', c=colors[i, :])
    plt.scatter(loc_xyzs[:, 0], loc_xyzs[:, 1], s=30.0, c='g')
    plt.show()

def debugAstar_2():
    num_of_x, num_of_y, num_of_z = 10, 10, 1
    instance = mapf_pipeline.Instance(num_of_x, num_of_y, num_of_z)

    start_xyzs = np.array([
        [1, 0, 0]
    ])
    goal_xyzs = np.array([
        [5, 5, 0]
    ])

    start_locs = []
    for x, y, z in start_xyzs:
        loc = instance.linearizeCoordinate(x, y, z)
        start_locs.append(loc)

    goal_locs = []
    for x, y, z in goal_xyzs:
        loc = instance.linearizeCoordinate(x, y, z)
        goal_locs.append(loc)

    xs, ys = np.meshgrid(np.arange(0, num_of_x, 1), np.arange(0, num_of_y, 1))
    map_xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1).reshape((-1, 2))
    plt.scatter(map_xys[:, 0], map_xys[:, 1], s=3.0, c='b')

    constrain_table = mapf_pipeline.ConstraintTable()
    obstacle_table = mapf_pipeline.ConstraintTable()

    obstacle_xyzr = np.array([
        [1, 5, 0, 0.5],
    ])
    for x, y, z, radius in obstacle_xyzr:
        obstacle_table.insert2CT(x, y, z, radius)

    solver = mapf_pipeline.AStarSolver(with_AnyAngle=False, with_OrientCost=True)
    pathIdxs = solver.findPath(
        radius=0.5,
        constraint_table=constrain_table,
        obstacle_table=obstacle_table,
        instance=instance,
        start_locs=start_locs,
        goal_locs=goal_locs
    )
    print('PathSize: ', len(pathIdxs))

    if len(pathIdxs) != 0:
        path_xy = []
        for idx in pathIdxs:
            (x, y, z) = instance.getCoordinate(idx)
            path_xy.append([x, y])
        path_xy = np.array(path_xy)

        plt.plot(path_xy[:, 0], path_xy[:, 1], '*-', c='b')

    plt.scatter(goal_xyzs[:, 0], goal_xyzs[:, 1], s=30.0, c='g')
    plt.scatter(start_xyzs[:, 0], start_xyzs[:, 1], s=30.0, c='r')
    plt.scatter(obstacle_xyzr[:, 0], obstacle_xyzr[:, 1], s=60.0, c='b')
    plt.show()

if __name__ == '__main__':
    # debug_constrainTable()
    debugAstar()
    # debugGroupAstar()
