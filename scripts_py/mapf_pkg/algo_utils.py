import numpy as np
import argparse
import json

from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils


def create_search_init_cfg(pipes_cfg: dict):
    """
    pipe_info: {name: {radius, position, direction, group_idx}}
    """
    group_names = {}
    for name in pipes_cfg.keys():
        group_names.setdefault(pipes_cfg[name]['group_idx'], [])
        group_names[pipes_cfg[name]['group_idx']].append(name)

    block_priority = []
    for block_id in range(1):
        block_info = {
            'block_id': block_id,
            'groups': []
        }
        for group_idx in group_names.keys():
            block_info['groups'].append({
                'group_idx': group_idx,
                'names': group_names[group_idx]
            })
        block_priority.append(block_info)
    return block_priority


def discrete_pipe_position(pipes_cfg: dict, grid_min: np.ndarray, length_of_xyz: np.ndarray):
    for pipe_name in pipes_cfg.keys():
        position = np.array(pipes_cfg[pipe_name]['position'])
        discrete_grid = np.round((position - grid_min) / length_of_xyz, 0).astype(int) * length_of_xyz + grid_min
        pipes_cfg[pipe_name]['discrete_position'] = discrete_grid.tolist()
    return pipes_cfg


def create_obs_pcd(pipes_cfg: dict, obs_cfg: dict):
    min_reso = np.inf
    for pipe_name in pipes_cfg.keys():
        min_reso = min(pipes_cfg[pipe_name]['radius'], min_reso)

    obs_pcds = []
    for pipe_name in pipes_cfg.keys():
        # ------ create shell
        shell_pcd = ShapeUtils.create_sphere_pcd(
            center=pipes_cfg[pipe_name]['discrete_position'],
            vector=pipes_cfg[pipe_name]['direction'],
            radius=pipes_cfg[pipe_name]['radius'] * 1.015,
            reso=min_reso * 0.3
            # reso=3.0
        )
        obs_pcds.append(shell_pcd)
    for obs_name in obs_cfg.keys():
        obs_pcds.append(obs_cfg[obs_name]['pcd'])
    obs_pcds = np.concatenate(obs_pcds, axis=0)

    for pipe_name in pipes_cfg.keys():
        obs_pcds = ShapeUtils.remove_obstacle_point_in_pipe(
            obs_pcd=obs_pcds,
            center=pipes_cfg[pipe_name]['discrete_position'],
            direction=pipes_cfg[pipe_name]['direction'],
            radius=pipes_cfg[pipe_name]['radius'],
            tol_scale=1.025,
            is_input=pipes_cfg[pipe_name]['is_input']
        )

    return obs_pcds
