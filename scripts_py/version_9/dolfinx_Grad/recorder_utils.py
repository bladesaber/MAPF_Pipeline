import numpy as np
from dolfinx.io import gmshio, XDMFFile, VTKFile
import dolfinx
from mpi4py import MPI
from tensorboardX import SummaryWriter
from typing import Union, List, Dict
import pyvista
from sklearn.neighbors import KDTree

from .vis_mesh_utils import VisUtils

class XDMFRecorder(object):
    def __init__(self, file: str, comm=None):
        assert file.endswith('.xdmf')
        if comm is None:
            comm = MPI.COMM_WORLD

        self.writter = XDMFFile(comm, file, 'w')

    def write_mesh(self, doamin: dolfinx.mesh.Mesh):
        self.writter.write_mesh(doamin)

    def write_function(self, function: dolfinx.fem.Function, step):
        self.writter.write_function(function, step)

    def close(self):
        self.writter.close()


class VTKRecorder(object):
    def __init__(self, file: str, comm=None):
        assert file.endswith('.pvd')
        if comm is None:
            comm = MPI.COMM_WORLD

        self.writter = VTKFile(comm, file, 'w')

    def write_mesh(self, doamin: dolfinx.mesh.Mesh, step):
        self.writter.write_mesh(doamin, step)

    def write_function(self, function: Union[dolfinx.fem.Function, List[dolfinx.fem.Function]], step):
        self.writter.write_function(function, step)

    def close(self):
        self.writter.close()


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    def write_scalars(self, tag, scalar_dict: Dict, step):
        self.writer.add_scalars(tag, scalar_dict, step)


class RefDataRecorder(object):
    def __init__(self, domain: dolfinx.mesh.Mesh):
        self.domain = domain
        self.vtk_files = []

        self.ref_grid: pyvista.DataSet = None
        self.ref2domain_point, self.domain2ref_point = None, None
        self.ref2domain_cell, self.domain2ref_cell = None, None
        self.fields = []

    def load_vtk_file(self, vtu_file, field: str):
        grid = self.read_vtu(vtu_file)
        if len(self.vtk_files) == 0:
            self.ref_grid = grid
            self.ref2domain_point, self.domain2ref_point, self.ref2domain_cell, self.domain2ref_cell = \
                self.update_idx_map()
        self.vtk_files.append(vtu_file)

        self.ref_grid[field] = np.array(grid[field])
        self.fields.append(field)

    def update_idx_map(self):
        ref2domain_point = []
        ref_tree = KDTree(self.ref_grid.points)
        dist_list, idx_list = ref_tree.query(self.domain.geometry.x, k=1, return_distance=True)
        for dists, idxs in zip(dist_list, idx_list):
            assert np.isclose(dists[0], 0.0)
            ref2domain_point.append(idxs[0])
        ref2domain_point = np.array(ref2domain_point)
        assert np.unique(ref2domain_point).shape == ref2domain_point.shape

        domain2ref_point = []
        domain_tree = KDTree(self.domain.geometry.x)
        dist_list, idx_list = domain_tree.query(self.ref_grid.points, k=1, return_distance=True)
        for dists, idxs in zip(dist_list, idx_list):
            assert np.isclose(dists[0], 0.0)
            domain2ref_point.append(idxs[0])
        domain2ref_point = np.array(domain2ref_point)

        # ----------------
        ref_cell_center_xyzs: np.ndarray = self.ref_grid.cell_centers().points
        grid = VisUtils.convert_to_grid(self.domain)
        domain_cell_center_xyzs: np.ndarray = grid.cell_centers().points

        ref2domain_cell = []
        ref_tree = KDTree(ref_cell_center_xyzs)
        dist_list, idx_list = ref_tree.query(domain_cell_center_xyzs, k=1, return_distance=True)
        for dists, idxs in zip(dist_list, idx_list):
            assert np.isclose(dists[0], 0.0)
            ref2domain_cell.append(idxs[0])
        ref2domain_cell = np.array(ref2domain_cell)
        assert np.unique(ref2domain_cell).shape == ref2domain_cell.shape

        domain2ref_cell = []
        domain_tree = KDTree(domain_cell_center_xyzs)
        dist_list, idx_list = domain_tree.query(ref_cell_center_xyzs, k=1, return_distance=True)
        for dists, idxs in zip(dist_list, idx_list):
            assert np.isclose(dists[0], 0.0)
            domain2ref_cell.append(idxs[0])
        domain2ref_cell = np.array(domain2ref_cell)

        assert ref2domain_point.shape[0] == domain2ref_point.shape[0] == self.ref_grid.number_of_points
        assert ref2domain_cell.shape[0] == domain2ref_cell.shape[0] == self.ref_grid.number_of_cells

        return ref2domain_point, domain2ref_point, ref2domain_cell, domain2ref_cell

    @staticmethod
    def read_vtu(file: str):
        assert file.endswith('.vtu')
        ref_data: pyvista.DataSet = pyvista.read(file)
        return ref_data

    def get_data(self, name, convert2domain=True):
        if convert2domain:
            if self.ref_grid[name].shape[0] == self.ref_grid.number_of_points:
                convert_seq = self.ref2domain_point
            elif self.ref_grid[name].shape[0] == self.ref_grid.number_of_cells:
                convert_seq = self.ref2domain_cell
            else:
                raise NotImplementedError

            if self.ref_grid[name].ndim == 2:
                return self.ref_grid[name][convert_seq, :]
            else:
                return self.ref_grid[name][convert_seq]
        else:
            return self.ref_grid[name]

