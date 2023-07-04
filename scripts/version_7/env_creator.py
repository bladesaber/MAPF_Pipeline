import open3d as o3d
import os
import numpy as np
import pandas as pd
import json
import math
from sklearn.neighbors import KDTree

from scripts.version_6.env_cfg import pipe_config, obstacle_config, env_config

from scripts.version_6 import o3d_helper

def create_mesh(mesh_dir):
    for idx, cfg in enumerate(obstacle_config):
        if cfg['type'] == 'Z_support':
            mesh = o3d_helper.create_Z_support_mesh(
                cfg['position'][0], cfg['position'][1], cfg['position'][2],
                cfg['radius'], cfg['height']
            )

        elif cfg['type'] == 'Z_screw':
            mesh = o3d_helper.create_Z_screw_mesh(
                cfg['position'][0], cfg['position'][1], cfg['position'][2],
                cfg['radius'], cfg['height']
            )

        elif cfg['type'] == 'X_valve':
            mesh = o3d_helper.create_X_valve_mesh(
                cfg['position'][0], cfg['position'][1], cfg['position'][2],
                cfg['radius'], cfg['height']
            )
        elif cfg['type'] == 'Z_valve':
            mesh = o3d_helper.create_Z_valve_mesh(
                cfg['position'][0], cfg['position'][1], cfg['position'][2],
                cfg['radius'], cfg['height']
            )

        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, '%s.stl'%cfg['name']), mesh)

def create_StaticObstacle(env_cfg, save_dir):
    pcd_list = []
    for info in env_cfg['static_obstacle']:
        path = os.path.join(env_cfg['mesh_dir'], info['file'])
        mesh:o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()

        pcd:o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(number_of_points=info['sample_num'])
        pcd_np = np.asarray(pcd.points)
        pcd_list.append(pcd_np)
    
    pcd_list = np.concatenate(pcd_list, axis=0)
    obs_df = pd.DataFrame(pcd_list, columns=['x', 'y', 'z'])
    obs_df['radius'] = 0.0
    obs_df['tag'] = 'mesh'

    save_file = os.path.join(save_dir, 'obs_df.csv')
    obs_df.to_csv(save_file)
    env_cfg['static_obs_pcd'] = save_file
    return env_cfg

def create_GridEnv(env_cfg, save_dir):
    obs_df = pd.read_csv(env_cfg['static_obs_pcd'], index_col=0)
    obs_df[['x', 'y', 'z', 'radius']] = obs_df[['x', 'y', 'z', 'radius']] / env_cfg['resolution']
    save_file = os.path.join(save_dir, 'grid_obs_df.csv')
    obs_df.to_csv(save_file)
    env_cfg['static_grid_obs_pcd'] = save_file

    for pipeConfig in env_cfg['pipeConfig']:
        radius = []
        for pipe in pipeConfig['pipe']:
            pipe['grid_position'] = [
                int(pipe['position'][0] // env_cfg['resolution']),
                int(pipe['position'][1] // env_cfg['resolution']),
                int(pipe['position'][2] // env_cfg['resolution'])
            ]
            pipe['grid_radius'] = pipe['radius'] / env_cfg['resolution']
            radius.append(pipe['radius'])
        pipeConfig['grid_radius'] = np.mean(radius) / env_cfg['resolution']
    
    return env_cfg

def create_Grid_WallObstacle(env_cfg, save_dir):
    wall_resolution = np.inf
    for pipeConfig in env_cfg['pipeConfig']:
        wall_resolution = min(wall_resolution, pipeConfig['grid_radius'])
    wall_resolution = wall_resolution * 0.55

    xmax = env_cfg['x'] - 1
    ymax = env_cfg['y'] - 1
    zmax = env_cfg['z'] - 1

    xWallStep = math.ceil(xmax / wall_resolution)
    yWallStep = math.ceil(ymax / wall_resolution)
    zWallStep = math.ceil(zmax / wall_resolution)

    xWall = np.linspace(0, xmax, xWallStep)
    yWall = np.linspace(0, ymax, yWallStep)
    zWall = np.linspace(0, zmax, zWallStep)

    ys, zs = np.meshgrid(yWall, zWall)
    yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
    xs, zs = np.meshgrid(xWall, zWall)
    xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
    xs, ys = np.meshgrid(xWall, yWall)
    xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))
    
    xmin_wall = np.concatenate([np.zeros((yzs.shape[0], 1)), yzs], axis=1)
    xmax_wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmax, yzs], axis=1)

    ymin_wall = np.concatenate([xzs[:, :1], np.zeros((xzs.shape[0], 1)), xzs[:, -1:]], axis=1)
    ymax_wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymax, xzs[:, -1:]], axis=1)

    zmin_wall = np.concatenate([xys, np.zeros((xys.shape[0], 1))], axis=1)
    zmax_wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmax], axis=1)

    wall_np = np.concatenate([
        xmin_wall, xmax_wall,
        ymin_wall, ymax_wall,
        zmin_wall, zmax_wall
    ], axis=0)

    delete_idxs = []
    kd_tree = KDTree(wall_np.copy())
    for pipeConfig in env_cfg['pipeConfig']:
        for pipe in pipeConfig['pipe']:
            idxs = kd_tree.query_radius(np.array([[
                pipe['grid_position'][0],
                pipe['grid_position'][1],
                pipe['grid_position'][2]
            ]]), r=pipeConfig['grid_radius']*1.025)[0]
            
            delete_idxs.extend(list(idxs))

    select_bool = np.ones(wall_np.shape[0]).astype(np.bool)
    select_bool[delete_idxs] = False
    wall_np = wall_np[select_bool]

    obs_df = pd.DataFrame(wall_np, columns=['x', 'y', 'z'])
    obs_df['radius'] = 0.0
    obs_df['tag'] = 'wall'

    print("Shape:", obs_df.shape, ' WallResol:', wall_resolution)

    save_file = os.path.join(save_dir, 'wall_obs_df.csv')
    obs_df.to_csv(save_file)
    env_cfg['wall_obs_pcd'] = save_file

    return env_cfg

def show_env(env_cfg, is_grid, with_wall_obs):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=720, width=960)

    if is_grid:
        obs_df = pd.read_csv(env_cfg['static_grid_obs_pcd'], index_col=0)
    else:
        obs_df = pd.read_csv(env_cfg['static_obs_pcd'], index_col=0)

    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_df[['x', 'y', 'z']].values)
    vis.add_geometry(obs_pcd)

    if with_wall_obs:
        wall_obs_df = pd.read_csv(env_cfg['wall_obs_pcd'], index_col=0)
        wall_obs_pcd = o3d.geometry.PointCloud()
        wall_obs_pcd.points = o3d.utility.Vector3dVector(wall_obs_df[['x', 'y', 'z']].values)
        vis.add_geometry(wall_obs_pcd)

    colors = np.random.uniform(0.0, 1.0, size=(len(pipe_config), 3))
    for pipeConfig in env_cfg['pipeConfig']:
        for pipe in pipeConfig['pipe']:
            if is_grid:
                mesh = o3d_helper.create_create_sphere(
                    pipe['grid_position'][0], pipe['grid_position'][1], pipe['grid_position'][2], pipeConfig['grid_radius']
                )
            else:
                mesh = o3d_helper.create_create_sphere(
                    pipe['position'][0], pipe['position'][1], pipe['position'][2], pipe['radius']
                )

            mesh.paint_uniform_color(colors[pipeConfig['groupIdx']])
            vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()

def main():
    ### Just For Debug
    # create_mesh('/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/mesh')

    ### -------------------------------------------
    json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/env_cfg.json'
    grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/grid_env_cfg.json'

    # with open(json_file, 'r') as f:
    #     env_config = json.load(f)

    # env_config = create_StaticObstacle(
    #     env_cfg=env_config, save_dir='/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir'
    # )

    # env_config = create_GridEnv(env_config, save_dir='/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir')

    # with open(grid_json_file, 'r') as f:
    #     env_config = json.load(f)

    # env_config = create_Grid_WallObstacle(env_cfg=env_config, save_dir='/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir')

    # with open(grid_json_file, 'w') as f:
    #     env_config = json.dump(env_config, f)

    with open(grid_json_file, 'r') as f:
        env_config = json.load(f)

    show_env(env_config, is_grid=True, with_wall_obs=True)

if __name__ == '__main__':
    main()
