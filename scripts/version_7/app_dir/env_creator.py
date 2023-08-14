import open3d as o3d
import os
import numpy as np
import pandas as pd
import json
import math
from sklearn.neighbors import KDTree

from scripts.version_7.app_dir.env_cfg import env_config
from scripts.version_7.app_dir.obstacle_config import obstacle_config
from scripts.visulizer import VisulizerVista

def createBoxWall(xmin, xmax, ymin, ymax, zmin, zmax, xSteps, ySteps, zSteps):
    xWall = np.linspace(xmin, xmax, xSteps)
    yWall = np.linspace(ymin, ymax, ySteps)
    zWall = np.linspace(zmin, zmax, zSteps)

    ys, zs = np.meshgrid(yWall, zWall)
    yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
    xs, zs = np.meshgrid(xWall, zWall)
    xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
    xs, ys = np.meshgrid(xWall, yWall)
    xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))

    xmin_wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmin, yzs], axis=1)
    xmax_wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmax, yzs], axis=1)
    ymin_wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymin, xzs[:, -1:]], axis=1)
    ymax_wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymax, xzs[:, -1:]], axis=1)
    zmin_wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmin], axis=1)
    zmax_wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmax], axis=1)

    wall_pcd = np.concatenate([
        xmin_wall, xmax_wall,
        ymin_wall, ymax_wall,
        zmin_wall, zmax_wall
    ], axis=0)
    wall_pcd = pd.DataFrame(wall_pcd).drop_duplicates().values

    return wall_pcd

def createBoxSolid(xmin, xmax, ymin, ymax, zmin, zmax, xSteps, ySteps, zSteps):
    xs = np.linspace(xmin, xmax, xSteps)
    ys = np.linspace(ymin, ymax, ySteps)
    zs = np.linspace(zmin, zmax, zSteps)
    xs, ys, zs = np.meshgrid(xs, ys, zs)
    xyzs = np.concatenate([
        xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]
    ], axis=-1)
    xyzs = xyzs.reshape((-1, 3))
    xyzs = pd.DataFrame(xyzs).drop_duplicates().values
    return xyzs

def create_obstacleXYZs(env_cfg, obstacle_config, obstacleReso, scale=1.0):
    ### ------ create Obstacle Point Cloud
    obstacles_xyzs = []
    for cfg in obstacle_config:
        x_length = int(cfg['x_length'] * scale)
        y_length = int(cfg['y_length'] * scale)
        z_length = int(cfg['z_length'] * scale)
        x = int(cfg['position'][0] * scale)
        y = int(cfg['position'][1] * scale)
        z = int(cfg['position'][2] * scale)

        cfg['grid_x_length'] = x_length
        cfg['grid_y_length'] = y_length
        cfg['grid_z_length'] = z_length
        cfg['grid_position'] = [x, y, z]

        semi_xLength = x_length / 2.0
        semi_yLength = y_length / 2.0
        semi_zLength = z_length / 2.0

        xSteps = max(math.ceil(x_length / obstacleReso), 2)
        ySteps = max(math.ceil(y_length / obstacleReso), 2)
        zSteps = max(math.ceil(z_length / obstacleReso), 2)

        if cfg['type'] == 'Wall':
            xyzs = createBoxWall(
                x - semi_xLength, x + semi_xLength,
                y - semi_yLength, y + semi_yLength,
                z - semi_zLength, z + semi_zLength,
                xSteps, ySteps, zSteps
            )
        elif cfg['type'] == 'Solid':
            xyzs = createBoxSolid(
                x - semi_xLength, x + semi_xLength,
                y - semi_yLength, y + semi_yLength,
                z - semi_zLength, z + semi_zLength,
                xSteps, ySteps, zSteps
            )

        obstacles_xyzs.append(xyzs)    
    obstacles_xyzs = np.concatenate(obstacles_xyzs, axis=0)

    obstacles_xyzs = pd.DataFrame(obstacles_xyzs, columns=['x', 'y', 'z']).drop_duplicates()
    obstacles_xyzs['tag'] = 'Obstacle'
    obstacles_xyzs['radius'] = 0.0

    ### ------ create Wall Point Cloud
    xmax = int(env_cfg['x'] * scale)
    ymax = int(env_cfg['y'] * scale)
    zmax = int(env_cfg['z'] * scale)
    xSteps = math.ceil(xmax / obstacleReso)
    ySteps = math.ceil(ymax / obstacleReso)
    zSteps = math.ceil(zmax / obstacleReso)

    wall_xyzs = createBoxWall(0.0, xmax, 0.0, ymax, 0.0, zmax, xSteps, ySteps, zSteps)
    wall_xyzs = pd.DataFrame(wall_xyzs, columns=['x', 'y', 'z']).drop_duplicates()
    wall_xyzs['tag'] = 'Wall'
    wall_xyzs['radius'] = 0.0

    ### ------
    df = pd.concat([wall_xyzs, obstacles_xyzs], axis=0, ignore_index=True)
    df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True, keep='first')
    df = df[(df['x'] >= 0) & (df['x'] <= xmax)]
    df = df[(df['y'] >= 0) & (df['y'] <= ymax)]
    df = df[(df['z'] >= 0) & (df['z'] <= zmax)]

    env_cfg['grid_x'] = xmax + 1
    env_cfg['grid_y'] = ymax + 1
    env_cfg['grid_z'] = zmax + 1

    # print(df.shape)
    # mesh = VisulizerVista.create_pointCloud(df[['x', 'y', 'z']].values)
    # mesh.plot()

    return df

def createCondition(env_cfg, obstacle_config):
    scale = env_cfg['scale']

    obs_df = create_obstacleXYZs(
        env_cfg, obstacle_config, obstacleReso=env_cfg['obstacle_resolution'], scale=scale
    )
    obs_xyzs = obs_df[['x', 'y', 'z']].values
    obs_tree = KDTree(obs_xyzs)
    
    delete_idxs = []
    for pipeConfig in env_cfg['pipeConfig']:
        radius = []
        for pipe in pipeConfig['pipe']:
            pipe['grid_position'] = [
                int(pipe['position'][0] * scale),
                int(pipe['position'][1] * scale),
                int(pipe['position'][2] * scale)
            ]
            pipe['grid_radius'] = pipe['radius'] * scale
            radius.append(pipe['grid_radius'])
        pipeConfig['grid_radius'] = np.mean(radius)
        
        for pipe in pipeConfig['pipe']:
            idxs = obs_tree.query_radius(np.array([[
                pipe['grid_position'][0],
                pipe['grid_position'][1],
                pipe['grid_position'][2]
            ]]), r=pipeConfig['grid_radius'] * 1.01)[0]
            delete_idxs.extend(list(idxs))

    select_bool = np.ones(obs_df.shape[0]).astype(bool)
    select_bool[delete_idxs] = False
    obs_df = obs_df[select_bool]

    obs_savePath = os.path.join(env_cfg['projectDir'], 'obstacle.csv')
    obs_df.to_csv(obs_savePath)
    env_cfg['obstacleSavePath'] = obs_savePath

    print('[DEBUG]: Obstacle Num: ', obs_df.shape[0])
    # mesh = VisulizerVista.create_pointCloud(obs_df[['x', 'y', 'z']].values)
    # mesh.plot()

    with open(os.path.join(env_cfg['projectDir'], 'cond.json'), 'w') as f:
        json.dump(env_config, f)

    with open(os.path.join(env_cfg['projectDir'], 'condObs.json'), 'w') as f:
        json.dump(obstacle_config, f)

def show_env(env_cfg):
    vis = VisulizerVista()

    obs_df = pd.read_csv(env_cfg['obstacleSavePath'], index_col=0)

    random_colors = np.random.uniform(0.0, 1.0, size=(len(env_cfg['pipeConfig']), 3))
    for pipeConfig in env_cfg['pipeConfig']:
        for pipe in pipeConfig['pipe']:
            mesh = vis.create_sphere(
                np.array(pipe['grid_position']), radius=pipeConfig['grid_radius']
            )
            vis.plot(mesh, color=random_colors[pipeConfig['groupIdx']])

    wall_mesh = vis.create_pointCloud(obs_df[obs_df['tag'] == 'Wall'][['x', 'y', 'z']].values)
    vis.plot(wall_mesh, color=(0.5, 0.5, 0.5), opacity=1.0)

    obstacle_mesh = vis.create_pointCloud(obs_df[obs_df['tag'] == 'Obstacle'][['x', 'y', 'z']].values)
    vis.plot(obstacle_mesh, color=(1.0, 0.5, 0.25), opacity=1.0)

    vis.ploter.add_axes()

    vis.show()

def main():
    # createCondition(env_config, obstacle_config)

    cond_json_file = '/home/admin123456/Desktop/temptory/application_pipe/cond.json'
    with open(cond_json_file, 'r') as f:
        new_config = json.load(f)
    show_env(new_config)

if __name__ == '__main__':
    main()
