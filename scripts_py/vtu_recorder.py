import os
import shutil

import numpy as np
import pyvista
from tensorboardX import SummaryWriter
from scripts_py.mesh_utils import MeshObj


class VtuRecorder(object):
    def __init__(self, mesh: MeshObj, save_dir: str, name='report', is_binary=True):
        self.name = name
        self.mesh = mesh
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        self.save_dir = save_dir
        self.is_binary = is_binary

    def update_point_data(self, key: str, value: np.ndarray):
        assert value.ndim in [1, 2], "value must be 1D or 2D array"
        if value.ndim == 1:
            assert value.shape[0] == self.mesh.grid.n_points, "value must be the same length as the number of points"
            self.mesh.grid.point_data[key] = value
            
        elif value.ndim == 2:
            assert value.shape[0] == self.mesh.grid.n_points and value.shape[1] == 3, "value must be the same length as the number of points and 3D"
            self.mesh.grid.point_data[key] = value

    def update_cell_data(self, key: str, value: np.ndarray):
        assert value.ndim in [1, 2], "value must be 1D or 2D array"
        if value.ndim == 1:
            assert value.shape[0] == self.mesh.grid.n_cells, "value must be the same length as the number of cells"
            self.mesh.grid.cell_data[key] = value
        elif value.ndim == 2:
            assert value.shape[0] == self.mesh.grid.n_cells and value.shape[1] == 3, "value must be the same length as the number of cells and 3D"
            self.mesh.grid.cell_data[key] = value

    def save(self, step: int):
        self.mesh.grid.save(os.path.join(self.save_dir, f'{self.name}_{step:05d}.vtu'), binary=self.is_binary)
    
    def clear(self, clear_point_data=True, clear_cell_data=True):
        if clear_point_data:
            self.mesh.grid.point_data.clear()
        if clear_cell_data:
            self.mesh.grid.cell_data.clear()

    def get_step_name(self, step: int):
        return f'{self.name}_{step:05d}.vtu'

    def get_step_file_path(self, step: int):
        return os.path.join(self.save_dir, f'{self.name}_{step:05d}.vtu')


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    def write_scalars(self, tag, scalar_dict: dict[str, float], step):
        if len(scalar_dict) == 0:
            return
        self.writer.add_scalars(tag, scalar_dict, step)

