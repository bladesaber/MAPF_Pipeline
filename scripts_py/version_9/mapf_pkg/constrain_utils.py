import numpy as np
import pandas as pd
from typing import Dict, List

from build import mapf_pipeline


class DynamicHumanConstrain(object):
    def __init__(self, grid_cfg: Dict):
        self.grid_env = mapf_pipeline.DiscreteGridEnv(
            size_of_x=grid_cfg['size_of_x'] + 1,
            size_of_y=grid_cfg['size_of_y'] + 1,
            size_of_z=grid_cfg['size_of_z'] + 1,
            x_init=grid_cfg['grid_min'][0],
            y_init=grid_cfg['grid_min'][1],
            z_init=grid_cfg['grid_min'][2],
            x_grid_length=grid_cfg['x_grid_length'],
            y_grid_length=grid_cfg['y_grid_length'],
            z_grid_length=grid_cfg['z_grid_length']
        )
        self.xyzr_idx = []

    def generate_constrain(self, x: float, y: float, z: float, radius: float, group_idx: int):
        x, y, z = self.grid_env.flag2xyz(self.grid_env.xyz2flag(x, y, z))
        self.xyzr_idx.append([x, y, z, radius, group_idx])

    def generate_straight_line_constrains(
            self, begin_grid: np.ndarray, end_grid: np.ndarray, radius: float, group_idx: int
    ):
        vec = end_grid - begin_grid
        vec_sign = np.sign(vec)
        step = int(np.sum(np.abs(vec)) / np.sum(np.abs(vec_sign)))
        for i in range(step + 1):
            inter_grid = begin_grid + vec_sign * i
            x, y, z = self.grid_env.grid2xyz(
                x_grid=inter_grid[0], y_grid=inter_grid[1], z_grid=inter_grid[2]
            )
            self.xyzr_idx.append([x, y, z, radius, group_idx])

    def generate(self, constrain_list: List[Dict], pipe_cfg: Dict):
        for info in constrain_list:
            if info['type'] == 'insert':
                xyzr = info['xyzr']
                self.generate_constrain(xyzr[0], xyzr[1], xyzr[2], xyzr[3], info['group_idx'])
            elif info['type'] == 'terminate_straight_line_constrains':
                pipe_info = pipe_cfg[info['name']]
                direction = np.array(pipe_info['direction'])
                if not pipe_info['is_input']:
                    direction = direction * -1.0

                begin_xyz = np.array(pipe_info['discrete_position'])
                end_xyz = begin_xyz + direction * info['length']
                begin_grid = np.array(self.grid_env.xyz2grid(
                    x=begin_xyz[0], y=begin_xyz[1], z=begin_xyz[2]
                ))
                end_grid = np.array(self.grid_env.xyz2grid(
                    x=end_xyz[0], y=end_xyz[1], z=end_xyz[2]
                ))
                self.generate_straight_line_constrains(
                    begin_grid, end_grid, pipe_info['radius'], pipe_info['group_idx']
                )

    def to_csv(self, file: str):
        if len(self.xyzr_idx) > 0:
            df = pd.DataFrame(np.array(self.xyzr_idx), columns=['x', 'y', 'z', 'radius', 'group_idx'])
            df['group_idx'] = df['group_idx'].astype(int)
        else:
            df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'group_idx'])
        df.to_csv(file)
