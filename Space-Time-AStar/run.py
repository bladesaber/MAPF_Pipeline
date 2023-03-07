import math
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import argparse
import yaml

from cbs.searcher import CBS_Planner
from cbs.smooth import bezier_smooth
from cbs.visulizer import create_tunnel_3d, create_line_3d, sample_tunnel

def parse_arg():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--yaml_path', type=str,
                        default='D:/gittest/reference/Space-Time-AStar-master/save_dir/test.yaml')
    args = parser.parse_args()
    return args

def run_search_path(
        grid_xmax:int, grid_ymax:int, robot_radius:float,
        starts: np.array, goals:np.array, save_path, radius_scale=1.5
):
    planner = CBS_Planner(
        grid_xmax=grid_xmax, grid_ymax=grid_ymax, robot_radius=robot_radius * radius_scale,
        starts=starts, goals=goals
    )
    solutions = planner.plan(max_process=5)

    if len(solutions) == 0:
        return

    save_paths = []
    for agent in planner.agents:
        path = solutions[agent]
        save_paths.append(path)
    np.save(save_path, np.array(save_paths))

class StepVis(object):
    def __init__(self, solutions, radius, colors, grid_xmax, grid_ymax):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        smooth_step = 100
        max_z = 0
        self.obj_pcd = {}
        for idx, (solution, color) in enumerate(zip(solutions, colors)):
            self.obj_pcd[idx] = {}

            self.obj_pcd[idx]['pcd'] = o3d.geometry.PointCloud()
            # self.vis.add_geometry(self.obj_pcd[idx]['pcd'])

            head = np.tile(solution[0: 1], (10, 1))
            tail = np.tile(solution[-2: -1], (10, 1))
            solution = np.concatenate([head, solution, tail], axis=0)
            solution = np.concatenate([solution, np.arange(0, solution.shape[0], 1).reshape((-1, 1))], axis=1)
            # self.obj_pcd[idx]['solution'] = solution

            smooth_solution = bezier_smooth(solution, smooth_step)
            self.obj_pcd[idx]['smooth_solution'] = smooth_solution

            smooth_pcd_list = sample_tunnel(smooth_solution, radius)
            self.obj_pcd[idx]['smooth_pcd_list'] = smooth_pcd_list

            self.obj_pcd[idx]['color'] = color
            max_z = max(smooth_solution[:, 2].max(), max_z)

        self.max_step = smooth_step - 1
        self.step = 0

        grid = np.array([
            [radius/2.0, radius/2.0],
            [radius/2.0, grid_ymax + radius/2.0],
            [grid_xmax + radius/2.0, radius/2.0],
            [grid_xmax + radius/2.0, grid_ymax + radius/2.0]
        ])
        level_list = []
        for i in range(math.ceil(max_z)):
            level = np.concatenate([
                grid, np.ones((grid.shape[0], 1)) * i
            ], axis=1)
            level_list.append(level)
        background_pcd = np.concatenate(level_list, axis=0)

        self.background_pcd_o3d = o3d.geometry.PointCloud()
        self.background_pcd_o3d.points = o3d.utility.Vector3dVector(background_pcd)
        self.background_pcd_o3d.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0., 0., 0.]]), (background_pcd.shape[0], 1))
        )
        self.vis.add_geometry(self.background_pcd_o3d)

        self.vis.register_key_callback(ord(','), self.step_visulize)
        self.vis.run()
        self.vis.destroy_window()

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.step < self.max_step:
            for key in self.obj_pcd.keys():
                new_pcd = self.obj_pcd[key]['smooth_pcd_list'][self.step]
                pcd_o3d: o3d.geometry.PointCloud = self.obj_pcd[key]['pcd']
                color = self.obj_pcd[key]['color']

                new_pcds = np.concatenate([np.asarray(pcd_o3d.points), new_pcd], axis=0)
                new_colors = np.tile(color.reshape((1, -1)), (new_pcds.shape[0], 1))
                pcd_o3d.points = o3d.utility.Vector3dVector(new_pcds)
                pcd_o3d.colors = o3d.utility.Vector3dVector(new_colors)

                if self.step == 0:
                    self.vis.add_geometry(pcd_o3d)
                else:
                    self.vis.update_geometry(pcd_o3d)

            self.step += 1

def smooth_and_vis(paths, colors, radius, grid_xmax, grid_ymax):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=960, height=720)
    #
    # for xys, color in zip(paths, colors):
    #     head = np.tile(xys[0: 1], (10, 1))
    #     tail = np.tile(xys[-2: -1], (10, 1))
    #     xys = np.concatenate([head, xys, tail], axis=0)
    #     xyzs = np.concatenate([xys, np.arange(0, xys.shape[0], 1).reshape((-1, 1))], axis=1)
    #
    #     smooth_xyzs = bezier_smooth(xyzs, 100)
    #
    #     # smooth_line = create_line_3d(smooth_xyzs, np.array([0.0, 1.0, 0.0]))
    #     # vis.add_geometry(smooth_line)
    #     pcd_o3d, axes_list = create_tunnel_3d(smooth_xyzs, radius=radius, color=color)
    #     vis.add_geometry(pcd_o3d)
    #
    # vis.run()
    # vis.destroy_window()

    vis = StepVis(paths, radius, colors, grid_xmax, grid_ymax)

if __name__ == '__main__':
    args = parse_arg()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        starts_pos = np.array(data['STARTS'])
        goals_pos = np.array(data['GOALS'])
        radius = data['RADIUS']
        radius_scale = data['RADIUS_SCALE']
        grid_xmax = data['GRID_XMAX']
        grid_ymax = data['GRID_YMAX']
        save_dir = data['SAVE_DIR']

    ### find solution
    # solution_save = os.path.join(save_dir, 'solution')
    # run_search_path(
    #     grid_xmax=grid_xmax, grid_ymax=grid_ymax,
    #     robot_radius=radius,
    #     starts=starts_pos,
    #     goals=goals_pos,
    #     save_path=solution_save,
    #     radius_scale=radius_scale
    # )

    ### visulize
    solution_save = os.path.join(save_dir, 'solution.npy')
    solutions = np.load(solution_save, allow_pickle=True)
    colors = np.random.uniform(0.0, 1.0, size=(solutions.shape[0], 3))
    smooth_and_vis(solutions, colors, radius, grid_xmax, grid_ymax)
