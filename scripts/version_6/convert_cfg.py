import open3d as o3d
import os
import numpy as np
import pandas as pd
import json

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

def create_grid_env(env_cfg, save_dir):
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

def show_env(env_cfg, is_grid):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=720, width=960)

    if is_grid:
        obs_df = pd.read_csv(env_cfg['static_grid_obs_pcd'], index_col=0)
    else:
        obs_df = pd.read_csv(env_cfg['static_obs_pcd'], index_col=0)

    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_df[['x', 'y', 'z']].values)
    vis.add_geometry(obs_pcd)

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
    # create_mesh('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/mesh')

    ### -------------------------------------------
    json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/env_cfg.json'
    grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_env_cfg.json'

    # with open(json_file, 'r') as f:
    #     env_config = json.load(f)

    # env_config = create_StaticObstacle(
    #     env_cfg=env_config, save_dir='/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir'
    # )

    # env_config = create_grid_env(env_config, save_dir='/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir')

    # with open(grid_json_file, 'w') as f:
    #     json.dump(env_config, f)

    with open(grid_json_file, 'r') as f:
        env_config = json.load(f)

    show_env(env_config, is_grid=True)

if __name__ == '__main__':
    main()
