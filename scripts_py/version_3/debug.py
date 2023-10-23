import numpy as np
import math
import matplotlib.pyplot as plt

from build import mapf_pipeline

def debug_instance():
    instance = mapf_pipeline.Instance(30, 30, 30)

    # instance.info()

    curr = np.random.randint(0, instance.map_size)
    print("curr: %d" % (curr))
    x = instance.getXCoordinate(curr)
    y = instance.getYCoordinate(curr)
    z = instance.getZCoordinate(curr)
    print("X:%d Y:%d Z:%d" % (x, y, z))

    (x, y, z) = instance.getCoordinate(curr)
    print("loc:%d, x:%d, y:%d z:%d" % (curr, x, y, z))

    curr = instance.linearizeCoordinate(0, 0, 0)
    neighbours = instance.getNeighbors(curr)
    print("neighbours", neighbours)
    for neighbour in neighbours:
        (x, y, z) = instance.getCoordinate(neighbour)
        print('neighbour:%d, x:%d, y:%d z:%d' % (neighbour, x, y, z))

    # curr = instance.linearizeCoordinate(x, y, z)
    # print("curr: %f" % (curr))
    # curr = instance.linearizeCoordinate((x, y, z))
    # print("curr: %f" % (curr))

def debug_ConstraintTable():
    constraint_table = mapf_pipeline.ConstraintTable()
    constraint_table.insert2CT(10, 0.5)
    constraint_table.insert2CT(20, 0.5)
    constraint_table.insert2CT(30, 0.5)
    constraint_table.insert2CT(30, 1.5)
    constraint_table.insert2CT(40, 1.0)
    
    ct = constraint_table.getCT()
    print(ct)

    print('conflict accepte: %d free accept: %d' % 
        (constraint_table.isConstrained(10), constraint_table.isConstrained(11))
    )

def debug_utils():
    dist = mapf_pipeline.point2LineSegmentDistance(
        0., 0., 0.,
        9., 14., 0.,
        1., 2., 0.
    )
    print(dist)

def debug_KDtree():
    path = [
        (0., 0., 0.),
        (0., 0., 1.),
        (0., 0., 2.),
        (0., 1., 2.),
        (0., 2., 2.),
        (1., 2., 2.),
        (2., 2., 2.)
    ]

    tree = mapf_pipeline.KDTreeWrapper()
    tree.insertPath(path)

    tree.nearest(0.0, 2.1, 2.1)

### ---------------------------------------------------
cond_2DParams = {
    'x': 30,
    'y': 30,
    'z': 1,
    'obs': 0.15,
    'radius': 0.5,

    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/version_3/map',
    'load': True,
}

def debug_AngleAstar2D():
    instance = mapf_pipeline.Instance(cond_2DParams['x'], cond_2DParams['y'], cond_2DParams['z'])

    if not cond_2DParams['load']:
        save_cond = {}
        if np.random.uniform(0.0, 1.0) > 0.5:
            start_xyz = (
                np.random.randint(low=0, high=cond_2DParams['x']),
                0, 
                0
            )

            goal_xyz = (
                np.random.randint(low=0, high=cond_2DParams['x']),
                cond_2DParams['y'] - 1,
                0
            )

        else:
            start_xyz = (
                0,
                np.random.randint(low=0, high=cond_2DParams['y']),
                0
            )

            goal_xyz = (
                cond_2DParams['x'] - 1,
                np.random.randint(low=0, high=cond_2DParams['y']),
                0
            )

        start_loc = instance.linearizeCoordinate(start_xyz)
        goal_loc = instance.linearizeCoordinate(goal_xyz)

        obs_locs = np.random.choice(
            np.arange(0, instance.map_size, 1), size=int(instance.map_size * cond_2DParams['obs']), replace=False
        )
        obs_locs = np.setdiff1d(obs_locs, np.array([start_loc, goal_loc]))

        save_cond['start_loc'] = start_loc
        save_cond['goal_loc'] = goal_loc
        save_cond['obs_loc'] = obs_locs

        np.save(cond_2DParams['save_path'], save_cond)
    
    else:
        save_cond = np.load(cond_2DParams['save_path']+'.npy', allow_pickle=True).item()
        start_loc = save_cond['start_loc']
        goal_loc = save_cond['goal_loc']
        obs_locs = save_cond['obs_loc']

        start_xyz = instance.getCoordinate(start_loc)
        goal_xyz = instance.getCoordinate(goal_loc)

    # new_obs_loc = []
    # for loc in obs_locs:
    #     (x, y, z) = instance.getCoordinate(loc)
    #     if x>6 or y>6:
    #         continue
    #     new_obs_loc.append(loc)
    # obs_locs = new_obs_loc

    constraints = []
    for loc in obs_locs:
        constraints.append((loc, cond_2DParams['radius']))
    
    model = mapf_pipeline.AngleAStar(cond_2DParams['radius'])
    
    print("start pos: ", start_xyz)
    print("goal_pos: ", goal_xyz)
    path = model.findPath(
        constraints = constraints,
        instance = instance,
        start_state = start_xyz,
        goal_state = goal_xyz
    )
    print("num_expanded:%d, num_generated:%d" % (model.num_expanded, model.num_generated))
    print("runtime_search:%f" % (model.runtime_search))
    
    plt.scatter(
        [start_xyz[0], goal_xyz[0]], 
        [start_xyz[1], goal_xyz[1]
    ], s=10.0, c='r')

    obs_np = []
    for loc in obs_locs:
        (x, y, z) = instance.getCoordinate(loc)
        obs_np.append([x, y])
    obs_np = np.array(obs_np)
    plt.scatter(obs_np[:, 0], obs_np[:, 1], s=20.0, c='b')

    # path_np = []
    # for loc in path:
    #     (x, y, z) = instance.getCoordinate(loc)
    #     print((x, y, z))
    #     path_np.append([x, y])
    # path_np = np.array(path_np)
    # if path_np.shape[0]>0:
    #     plt.plot(path_np[:, 0], path_np[:, 1], '-*', color='g')
    
    cbs_planner = mapf_pipeline.CBS()
    if len(path)>0:
        detail_path = cbs_planner.sampleDetailPath(path, instance, 0.5)
        detail_path = np.array(detail_path)
        plt.plot(detail_path[:, 0], detail_path[:, 1], '-*', color='g')

    plt.xlim(-1, cond_2DParams['x'] + 1)
    plt.ylim(-1, cond_2DParams['y'] + 1)

    plt.show()

if __name__ == '__main__':
    # debug_instance()
    # debug_ConstraintTable()

    # debug_AngleAstar2D()
    # debug_utils()

    debug_KDtree()

    pass
