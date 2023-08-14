import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
from copy import copy
import os

from build import mapf_pipeline
from scripts.visulizer import VisulizerVista

setting = np.load('/home/admin123456/Desktop/work/application/debug/record_0.npy', allow_pickle=True).item()
obstacle_df: pd.DataFrame = pd.read_csv(setting['obstacle_file'], index_col=0)
constraints = setting['constraints']
taskTree = setting['taskTree']
computed_res = setting['computed_resPaths']

task = taskTree[0]

### -------------------
instance = mapf_pipeline.Instance(setting['num_of_x'], setting['num_of_y'], setting['num_of_z'])
solver = mapf_pipeline.AStarSolver(with_AnyAngle=False, with_OrientCost=True)

print(task)
constrain_table = mapf_pipeline.ConstraintTable()
obstacle_table = mapf_pipeline.ConstraintTable()

for x, y, z, radius in constraints:
    constrain_table.insert2CT(x, y, z, radius)

for _, row in obstacle_df.iterrows():
    obstacle_table.insert2CT(row.x, row.y, row.z, row.radius)

radius = 1.575
pathIdxs = solver.findPath(
    radius=radius,
    constraint_table=constrain_table,
    obstacle_table=obstacle_table,
    instance=instance,
    start_locs=[
        task['loc0']
    ],
    goal_locs=[
        task['loc1']
    ]
)
path_xyzs = []
for idx in pathIdxs:
    (x, y, z) = instance.getCoordinate(idx)
    path_xyzs.append([x, y, z])
path_xyzs = np.array(path_xyzs)
# print(path_xyzs)

path_xyzs_r = np.array([
    [10, 26, 23],
    [11, 26, 23],
    [12, 26, 23],
    [13, 26, 23],
    [13, 26, 24],
    [13, 26, 25],
    [13, 26, 26],
    [13, 26, 27],
    [13, 26, 28],
    [13, 26, 29],
    [13, 26, 30],
    [13, 26, 31],
    [13, 26, 32],
    [13, 26, 33],
    [13, 26, 34],
    [13, 26, 35],

    [13, 25, 35],
    [13, 24, 35],
    [13, 24, 36],
    [12, 24, 36],
    [11, 24, 36],
    [10, 24, 36],
    [9, 24, 36],
    [8, 24, 36],
    [7, 24, 36],
])

def getCost(xyzs):
    cost = 0.0
    for i in range(xyzs.shape[0]):
        dif_cost = 0.0

        if i == 0:
            continue

        dif_cost += 1
        if i == 1:
            cost += dif_cost
            continue

        xyz0 = xyzs[i - 2]
        xyz1 = xyzs[i - 1]
        xyz2 = xyzs[i]

        if xyz1[0] - xyz0[0] != xyz2[0] - xyz1[0]:
            dif_cost += 1

        elif xyz1[1] - xyz0[1] != xyz2[1] - xyz1[1]:
            dif_cost += 1

        elif xyz1[2] - xyz0[2] != xyz2[2] - xyz1[2]:
            dif_cost += 1

        cost += dif_cost

    print(cost)

getCost(path_xyzs)
getCost(path_xyzs_r)

### ---------------------
# path_xyzrl = np.array(computed_res[0])

vis = VisulizerVista()

# tube_mesh = vis.create_tube(path_xyzrl[:, :3], radius=radius)
# line_mesh = vis.create_line(path_xyzrl[:, :3])
# vis.plot(tube_mesh, color=(0.0, 1.0, 0.0), opacity=0.6)
# vis.plot(line_mesh, color=(1.0, 0., 0.), opacity=1.0)

tube2_mesh = vis.create_tube(path_xyzs[:, :3], radius=radius)
line2_mesh = vis.create_line(path_xyzs[:, :3])
vis.plot(tube2_mesh, color=(0.0, 1.0, 0.0), opacity=0.6)
vis.plot(line2_mesh, color=(0.0, 0., 1.), opacity=1.0)

for x, y, z, radius in constraints:
    constraint_mesh = vis.create_sphere(np.array([x, y, z]), radius)
    vis.plot(constraint_mesh, (1.0, 0.0, 0.0), opacity=0.85)

obstacle_xyzs = obstacle_df[obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
# obstacle_xyzs = obstacle_df[['x', 'y', 'z']].values
obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

vis.show()
