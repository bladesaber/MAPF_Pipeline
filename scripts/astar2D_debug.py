from build import mapf_pipeline

import numpy as np
import pandas as pd
import random
from typing import List
import matplotlib.pyplot as plt

conditional_params = {
    'row': 30,
    'col': 30,
    'obs': 0.25,
    'load_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/map.npy',
    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/map',

    'load': True,
}

def getMap(instance, load=False):
    if not load:
        xs, ys = np.meshgrid(np.arange(0, conditional_params['col'], 1), np.arange(0, conditional_params['row'], 1))
        map = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
        map = map.reshape((-1, 2))

        map_idx = np.arange(0, map.shape[0], 1)
        obs_idx = np.random.choice(map_idx, size=int(map.shape[0]*0.25))
        remain_idx = np.setdiff1d(map_idx, obs_idx)

        start_idx, goal_idx = np.random.choice(remain_idx, size=2)
        start_yx = (map[start_idx][1], map[start_idx][0])
        goal_yx = (map[goal_idx][1], map[goal_idx][0])

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

        np.save(
            conditional_params['save_path'], 
            {
                'map': map,
                'obs_idx': obs_idx,
                'start_yx': start_yx,
                'goal_yx': goal_yx,
                'constrains_table': constrains_table
            }
        )

    else:
        load_data = np.load(conditional_params['load_path'], allow_pickle=True).item()
        map = load_data['map']
        obs_idx = load_data['obs_idx']
        map_idx = np.arange(0, map.shape[0], 1)
        remain_idx = np.setdiff1d(map_idx, obs_idx)
        start_yx = load_data['start_yx']
        goal_yx = load_data['goal_yx']
        constrains_table = load_data['constrains_table']

    return (map, remain_idx, obs_idx), (start_yx, goal_yx), constrains_table

def main():
    instance = mapf_pipeline.Instance(conditional_params['row'], conditional_params['col'])
    astar = mapf_pipeline.SpaceTimeAStar(0)
    # astar.focus_optimal = True
    # astar.focus_w = 1.05
    
    (map, remain_idx, obs_idx), (start_yx, goal_yx), constrains_table = getMap(instance, load=conditional_params['load'])
    path: List = astar.findPath(
        paths={}, constraints=constrains_table, instance=instance, start_state=start_yx, goal_state=goal_yx
    )
    print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
    print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (
        astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT
    ))
    
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


if __name__ == '__main__':
    main()