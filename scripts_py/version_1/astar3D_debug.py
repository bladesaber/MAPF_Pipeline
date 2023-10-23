import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

from build import mapf_pipeline
from scripts_py.visulizer import VisulizerVista

cond_params = {
    'row': 50,
    'col': 50,
    'z': 50,
    'obs': 0.0,
    'load_path': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/map.npy',
    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/map',

    'load': False,
}

def getMap(instance, load=False):
    if not load:
        map_size = cond_params['row'] * cond_params['col'] * cond_params['z']
        obs_size = int(map_size * cond_params['obs'])

        obs_dict = {}
        constrains_table = {
            0: []
        }
        while True:
            if len(obs_dict.keys()) >= obs_size:
                break

            obs_yxz = (
                np.random.randint(0, cond_params['row']),
                np.random.randint(0, cond_params['col']),
                np.random.randint(0, cond_params['z'])
            )
            loc = instance.linearizeCoordinate(obs_yxz)
            if loc in obs_dict.keys():
                continue

            obs_dict[loc] = obs_yxz
            constrains_table[0].append(
                (0, loc, 0, mapf_pipeline.constraint_type.VERTEX)
            )

        while True:
            start_yxz = [
                np.random.randint(0, cond_params['row']),
                np.random.randint(0, cond_params['col']),
                np.random.randint(0, cond_params['z'])
            ]
            cut = np.random.uniform(0.0, 1.0)
            if cut <= 0.33:
                start_yxz[0] = 0
            elif cut < 0.66 and cut > 0.33:
                start_yxz[1] = 0
            else:
                start_yxz[2] = 0

            loc = instance.linearizeCoordinate(start_yxz)
            if loc not in obs_dict.keys():
                start_yxz = tuple(start_yxz)
                break

        while True:
            goal_yxz = [
                np.random.randint(0, cond_params['row']),
                np.random.randint(0, cond_params['col']),
                np.random.randint(0, cond_params['z'])
            ]
            cut = np.random.uniform(0.0, 1.0)
            if cut <= 0.33:
                goal_yxz[0] = 0
            elif cut < 0.66 and cut > 0.33:
                goal_yxz[1] = 0
            else:
                goal_yxz[2] = 0

            loc = instance.linearizeCoordinate(goal_yxz)
            if loc in obs_dict.keys():
                continue
            
            if start_yxz != goal_yxz:
                goal_yxz = tuple(goal_yxz)
                break
        
        np.save(
            cond_params['save_path'], 
            {
                'obs_dict': obs_dict,
                'start_yxz': start_yxz,
                'goal_yxz': goal_yxz,
                'constrains_table': constrains_table
            }
        )
    else:
        load_data = np.load(cond_params['load_path'], allow_pickle=True).item()
        obs_dict = load_data['obs_dict']
        start_yxz = load_data['start_yxz']
        goal_yxz = load_data['goal_yxz']
        constrains_table = load_data['constrains_table']

    return (start_yxz, goal_yxz), obs_dict, constrains_table

def test_single():
    instance = mapf_pipeline.Instance3D(cond_params['row'], cond_params['col'], cond_params['z'])
    astar = mapf_pipeline.SpaceTimeAStar(0)

    print("Starting ......")
    (start_yxz, goal_yxz), obs_dict, constrains_table = getMap(instance, cond_params['load'])
    print("start:", start_yxz, ' goal:', goal_yxz)
    path: List = astar.findPath(
        paths={}, constraints=constrains_table, instance=instance, start_state=start_yxz, goal_state=goal_yxz
    )
    print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
    print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (
        astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT
    ))

    path_xyz = []
    for loc in path:
        (row, col, z) = instance.getCoordinate(loc)
        path_xyz.append([col, row, z])
    path_xyz = np.array(path_xyz)
    # print(path_xyz)

    vis = VisulizerVista()
    obs_xyzs = []
    for key, (y, x, z) in obs_dict.items():
        obs_xyzs.append([x, y, z])
    obs_xyzs = np.array(obs_xyzs)

    box1_mesh = vis.create_box(xyz=[start_yxz[1], start_yxz[0], start_yxz[2]])
    vis.plot(box1_mesh, color=(0.0, 1.0, 0.0))

    box2_mesh = vis.create_box(xyz=[goal_yxz[1], goal_yxz[0], goal_yxz[2]])
    vis.plot(box2_mesh, color=(1.0, 0.0, 0.0))

    boxs_mesh = vis.create_many_boxs(obs_xyzs, length=1.0)
    vis.plot(boxs_mesh, color=(0.0, 0.0, 0.0))

    tube_mesh = vis.create_tube(path_xyz, radius=0.5)
    vis.plot(tube_mesh, color=(0.1, 0.5, 0.8))

    vis.show()

def test_multitimes():
    instance = mapf_pipeline.Instance3D(cond_params['row'], cond_params['col'], cond_params['z'])
    astar = mapf_pipeline.SpaceTimeAStar(0)
    # astar.focus_optimal = True
    # astar.focus_w = 1.05

    run_time_list = []
    for _ in tqdm(range(100)):
        (start_yxz, goal_yxz), obs_dict, constrains_table = getMap(instance, False)

        start_time = time.time()
        path: List = astar.findPath(
            paths={}, constraints=constrains_table, instance=instance, start_state=start_yxz, goal_state=goal_yxz
        )
        print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
        print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f, pyhton_time:%f" % (
            astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT, time.time()-start_time
        ))

        if len(path) > 0:
            run_time_list.append(astar.runtime_search)

    sns.displot(run_time_list)
    plt.show()

if __name__ == '__main__':
    # test_single()
    test_multitimes()
