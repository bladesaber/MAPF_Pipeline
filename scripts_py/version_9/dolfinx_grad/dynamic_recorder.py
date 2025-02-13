import numpy as np
import os
import shutil
from tensorboardX import SummaryWriter
from typing import Dict

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj


class VtuRecorder(object):
    def __init__(self, mesh: MeshObj, save_dir: str, is_binary=True):
        self.mesh = mesh
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        self.save_dir = save_dir
        self.is_binary = is_binary

    def add_point_scalar(self, key: str, value: np.ndarray):
        self.mesh.grid.point_data[key] = value

    def add_cell_scalar(self, key: str, value: np.ndarray):
        self.mesh.grid.cell_data[key] = value

    def save(self, step: int):
        self.mesh.grid.save(
            os.path.join(self.save_dir, f'res_{step:05d}.vtu'), binary=self.is_binary)


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    def write_scalars(self, tag, scalar_dict: Dict[str, float], step):
        if len(scalar_dict) == 0:
            return
        self.writer.add_scalars(tag, scalar_dict, step)
