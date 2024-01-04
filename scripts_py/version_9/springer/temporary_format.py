import numpy as np
import json
import os

path = '/home/admin123456/Desktop/work/example7/grid_springer_env_cfg.json'
with open(path, 'r') as f:
    env_cfg = json.load(f)

pipes_cfg = env_cfg['pipe_cfgs']
for group_idx_str in pipes_cfg:
    group_pipes = pipes_cfg[group_idx_str]
    for name in group_pipes.keys():
        pipe_cfg = group_pipes[name]
        pipe_cfg.update({
            "name": name,
            'node_type': 'connector',
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'exclude_edges': []
        })

structors_cfg = env_cfg['obstacle_cfgs']
for name in structors_cfg.keys():
    if name == 'wall':
        continue
    cfg = structors_cfg[name]
    cfg.update({
        "name": name,
        'node_type': 'structor',
        'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
        'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
        'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
        'exclude_edges': {
            'plane_min_conflict': ['x', 'y', 'z'],
            'plane_max_conflict': ['x', 'y', 'z']
        }
    })

with open(path, 'w') as f:
    json.dump(env_cfg, f)
