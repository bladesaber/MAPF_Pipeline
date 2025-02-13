import os
import shutil
import numpy as np
import pyvista
import subprocess
import collections
from typing import Union, Callable, List, Sequence, Literal

import dolfinx
from dolfinx import geometry
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import ufl


class MeshUtils(object):
    @staticmethod
    def msh_to_XDMF(msh_file: str, xdmf_file: str, dim: int, name='model'):
        """
        msh_file: .msh, .gmsh2, .gmsh
        if operation fail, please check:
            if dim=3: whether all necessary volumes had been added to physical group
            if dim=2: whether all necessary surfaces had been added to physical group
        params:
            dim: dimension of model
            mesh_dim: dimension of mesh
        """
        assert xdmf_file.endswith('.xdmf')
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)
        mesh.name = name
        cell_tags.name = f"{mesh.name}_cells"
        facet_tags.name = f"{mesh.name}_facets"
        with XDMFFile(mesh.comm, xdmf_file, 'w') as f:
            mesh.topology.create_connectivity(dim - 1, dim)
            f.write_mesh(mesh)
            f.write_meshtags(cell_tags, mesh.geometry)
            f.write_meshtags(facet_tags, mesh.geometry)

    @staticmethod
    def read_msh(msh_file: str, dim: int) -> (dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags):
        assert msh_file.endswith('.msh')
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)
        return mesh, cell_tags, facet_tags

    @staticmethod
    def read_XDMF(xdmf_file: str, name='model') -> (dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags):
        assert xdmf_file.endswith('.xdmf')
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as f:
            domain = f.read_mesh(name=name)
            cell_tags = f.read_meshtags(domain, name=f'{name}_cells')
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

        # only after creating the connectivity, facet_tags can be extracted
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as f:
            facet_tags = f.read_meshtags(domain, name=f'{name}_facets')
        return domain, cell_tags, facet_tags

    @staticmethod
    def save_XDMF(
            xdmf_file: str, domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags
    ):
        """Save后Points的顺序会改变,原因不明"""
        assert xdmf_file.endswith('.xdmf')
        with XDMFFile(domain.comm, xdmf_file, 'w') as f:
            domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
            f.write_mesh(domain)
            f.write_meshtags(cell_tags, domain.geometry)
            f.write_meshtags(facet_tags, domain.geometry)

    @staticmethod
    def get_topology_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim

    @staticmethod
    def get_facet_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim - 1

    @staticmethod
    def mesh_to_vtk(domain: dolfinx.mesh.Mesh, tdim: int = None, entities=None):
        cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim, entities)
        grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
        return grid

    @staticmethod
    def extract_entity_indices(
            domain: dolfinx.mesh.Mesh,
            style: Literal['cell', 'facet'] = 'facet',
            marker: Union[Callable, int] = None,
            mesh_tags: dolfinx.mesh.MeshTags = None
    ):
        """entity indices为fenics内索引，dof为dolfin.fem.function数组内的索引"""
        tdim = domain.topology.dim
        if style == 'facet':
            if marker is None:
                # return whole boundary entity idxs
                domain.topology.create_connectivity(tdim - 1, tdim)
                entities_idxs = dolfinx.mesh.exterior_facet_indices(domain.topology)
            else:
                if isinstance(marker, Callable):
                    entities_idxs = dolfinx.mesh.locate_entities_boundary(domain, tdim - 1, marker=marker)
                elif isinstance(marker, int):
                    entities_idxs = mesh_tags.find(marker)
                else:
                    raise NotImplementedError
        elif style == 'cell':
            entities_idxs = dolfinx.mesh.locate_entities(domain, tdim, marker=marker)

        else:
            raise ValueError('[Error]: Non-valid Style.')
        return entities_idxs

    @staticmethod
    def extract_entity_dofs(V: dolfinx.fem.functionspace, dim: int, entity_idxs: np.ndarray):
        """dof为dolfin.fem.function数组内的索引idx"""
        return dolfinx.fem.locate_dofs_topological(V, entity_dim=dim, entities=entity_idxs)

    @staticmethod
    def define_facet_norm(domain: dolfinx.mesh.Mesh):
        return ufl.FacetNormal(domain)

    @staticmethod
    def define_coordinate(domain: dolfinx.mesh.Mesh):
        return ufl.SpatialCoordinate(domain)

    @staticmethod
    def define_ds(domain: dolfinx.mesh.Mesh, facets_tag: dolfinx.mesh.MeshTags = None):
        """define area of boundary"""
        return ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)

    @staticmethod
    def define_dx(domain: dolfinx.mesh.Mesh):
        """define volume of mesh"""
        return ufl.Measure("dx", domain=domain)

    @staticmethod
    def define_meshTag(
            domain: dolfinx.mesh.Mesh, indices_list: List[np.ndarray], markers_list: List[np.ndarray], dim: int
    ):
        indices = np.concatenate(indices_list, axis=-1).astype(np.int32)
        markers = np.concatenate(markers_list, axis=-1).astype(np.int32)
        mesh_tag = dolfinx.mesh.meshtags(domain, dim, indices, markers)
        return mesh_tag

    # @staticmethod
    # def move(domain: dolfinx.mesh.Mesh, coord_dif: np.ndarray):
    #     assert domain.geometry.x.shape == coord_dif.shape
    #     domain.geometry.x[:] += coord_dif

    @staticmethod
    def cell_indices_of_points(
            mesh: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], pcd_xyz: np.ndarray
    ) -> List[List[int]]:
        if isinstance(mesh, pyvista.UnstructuredGrid):
            cell_idxs = mesh.find_containing_cell(pcd_xyz)  # 只能返回最接近1个cell
        elif isinstance(mesh, dolfinx.mesh.Mesh):
            bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
            cell_candidates: dolfinx.cpp.graph.AdjacencyList_int32 = geometry.compute_collisions_points(
                bb_tree, pcd_xyz
            )
            colliding_cells: dolfinx.cpp.graph.AdjacencyList_int32 = geometry.compute_colliding_cells(
                mesh, cell_candidates, pcd_xyz
            )
            cell_idxs = []
            for i in range(pcd_xyz.shape[0]):
                cell_idxs.append(list(colliding_cells.links(i)))  # 可以返回包含该point的所有cell
        else:
            raise ValueError('[Error]: Non-valid Mesh.')

        return cell_idxs

    @staticmethod
    def cell_connectivity(mesh: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], dim: int = None) -> np.ndarray:
        if isinstance(mesh, pyvista.UnstructuredGrid):
            if dim == 2:
                return mesh.cells.reshape(-1, 4)
            else:
                return mesh.cells.reshape(-1, 5)
        elif isinstance(mesh, dolfinx.mesh.Mesh):
            con = []
            adjacency_list: dolfinx.cpp.graph.AdjacencyList_int32 = mesh.topology.connectivity(mesh.topology.dim, 0)
            for i in range(adjacency_list.num_nodes):
                con.append(adjacency_list.links(i))
            con = np.array(con)
            return con
        else:
            raise ValueError('[Error]: Non-valid Mesh Type.')

    @staticmethod
    def su2_to_msh(su2_file: str, msh_file: str):
        assert msh_file.endswith('.msh') and su2_file.endswith('.su2')
        run_bash = f"gmsh {su2_file} -format msh -o {msh_file} -save"
        subprocess.run(run_bash, shell=True, check=True)


class MeshObj(object):
    def __init__(
            self,
            mesh_file: str = None, mesh_name='model', dim=None,
            domain: dolfinx.mesh.Mesh = None,
            cell_tags: dolfinx.mesh.MeshTags = None,
            facet_tags: dolfinx.mesh.MeshTags = None
    ):
        if mesh_file is not None:
            assert isinstance(dim, int)
            if mesh_file.endswith('.xdmf'):
                domain, cell_tags, facet_tags = MeshUtils.read_XDMF(mesh_file, name=mesh_name)
            elif mesh_file.endswith('.msh'):
                domain, cell_tags, facet_tags = MeshUtils.read_msh(mesh_file, dim)
            else:
                raise ValueError('[Error]: Non-valid Config.')
            self.domain: dolfinx.mesh.Mesh = domain
            self.cell_tags: dolfinx.mesh.MeshTags = cell_tags
            self.facet_tags: dolfinx.mesh.MeshTags = facet_tags
            self.mesh_file = mesh_file

        elif domain is not None:
            self.domain: dolfinx.mesh.Mesh = domain
            self.cell_tags: dolfinx.mesh.MeshTags = cell_tags
            self.facet_tags: dolfinx.mesh.MeshTags = facet_tags

        else:
            raise ValueError('[Error]: Non-valid Mesh definition.')

        self.num_points = self.domain.geometry.x.shape[0]
        self.grid = MeshUtils.mesh_to_vtk(self.domain)
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.connectivity = MeshUtils.cell_connectivity(self.domain)
        cell_counter: collections.Counter = collections.Counter(self.connectivity.flatten().tolist())
        self.occurrences = np.array([cell_counter[i] for i in range(self.num_points)])

    @property
    def facet_dim(self):
        return self.domain.topology.dim - 1

    @property
    def topology_dim(self):
        return self.domain.topology.dim

    @property
    def points(self):
        return self.domain.geometry.x

    def deform(self, coord_dif: np.ndarray):
        assert self.domain.geometry.x.shape == coord_dif.shape
        self.domain.geometry.x[:] += coord_dif

    def is_collision(self):
        cell_idxs = MeshUtils.cell_indices_of_points(self.domain, self.points)
        cell_counts = []
        for idx_list in cell_idxs:
            cell_counts.append(len(idx_list))
        return np.any(cell_counts != self.occurrences)

    @staticmethod
    def estimate_mesh_quality(
            obj: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], quality_measure: str, tdim=None, entities=None
    ):
        """
        quality_measure:
            1. area
            2. radius_ratio
            3. skew            | Fail
            4. volume
            5. max_angle
            6. min_angle
            7. condition
        """
        if isinstance(obj, dolfinx.mesh.Mesh):
            grid = MeshUtils.mesh_to_vtk(obj, tdim, entities)
        else:
            grid = obj
        quality = grid.compute_cell_quality(quality_measure=quality_measure, progress_bar=False)
        return quality['CellQuality']

    def is_valid_volume_change(self, displacement_np: np.ndarray, vol_change: float):
        vol_orig = self.estimate_mesh_quality(self.grid, 'area' if self.tdim == 2 else 'volume')
        self.deform(displacement_np)
        vol_deformed = self.estimate_mesh_quality(self.grid, 'area' if self.tdim == 2 else 'volume')
        self.deform(displacement_np * -1.0)
        vol_dif = vol_deformed / vol_orig
        min_det, max_det = np.min(vol_dif) - 1.0, np.max(vol_dif) - 1.0
        is_valid = (min_det >= -vol_change) and (max_det <= vol_change)
        if not is_valid:
            print(f'[Info]: Non-valid deformation since volume change {min_det:.2f}/{max_det:.2f} too large.')
        return is_valid

    def is_valid_angle(self, displacement_np: np.ndarray, tol_max=160.0, tol_min=20.0):
        self.deform(displacement_np)
        max_angles = self.estimate_mesh_quality(self.grid, 'max_angle')
        min_angles = self.estimate_mesh_quality(self.grid, 'min_angle')
        self.deform(displacement_np * -1.0)
        is_valid = (np.min(min_angles) >= tol_min) and (np.max(max_angles) <= tol_max)
        if not is_valid:
            print(f'[Info]: Non-valid deformation since angle '
                  f'{np.min(min_angles):.2f}/{np.max(max_angles):.2f} out of range.')
        return is_valid

    def deform_safe(
            self, displacement_np: np.ndarray,
            with_vol_check=True, vol_change=0.2,
            with_angle_check=True, angle_tol_max=160., angle_tol_min=20.
    ):
        if with_vol_check:
            if not self.is_valid_volume_change(displacement_np, vol_change=vol_change):
                return False

        if with_angle_check:
            if not self.is_valid_angle(displacement_np, tol_max=angle_tol_max, tol_min=angle_tol_min):
                return False

        self.deform(displacement_np)
        if self.is_collision():
            self.deform(displacement_np * -1.0)
            print('[Info]: Non-valid deformation Since collision.')
            return False
        return True

    def deform_line_search(
            self,
            dire_np: np.ndarray, scales: List[float],
            with_vol_check=True, vol_change=0.2,
            with_angle_check=True, angle_tol_max=160., angle_tol_min=20.
    ):
        for scale in scales:
            is_valid = self.deform_safe(
                dire_np * scale,
                with_vol_check, vol_change,
                with_angle_check, angle_tol_max, angle_tol_min
            )
            if is_valid:
                return True
        return False

    def record_point_data(self, key: str, value: np.ndarray):
        self.grid.point_data[key] = value

    def save_mesh(self, file: str):
        assert file.endswith('.vtu')
        self.grid.save(file, binary=False)


def debug():
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        '/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/pipe_1.msh', dim=2
    )

    # domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, style='wireframe', color='red')
    # plotter.add_mesh(np.asarray(grid.cell_centers().points), color='green')
    # # plotter.add_mesh(pyvista.PointSet(grid.points[dofs, :]), color='red')
    # plotter.show()


if __name__ == '__main__':
    debug()
