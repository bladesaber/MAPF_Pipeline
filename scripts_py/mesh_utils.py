import os
import shutil
from typing import Literal, Union, Callable

import dolfinx
from dolfinx import geometry
from dolfinx.io import XDMFFile  # type: ignore
try:
    if dolfinx.__version__ == '0.10.0':  # type: ignore
        from dolfinx.io import gmsh as gmshio  # type: ignore
    else:
        from dolfinx.io import gmshio  # type: ignore
except ImportError:
    raise ImportError("[Error]: gmshio is not installed.")

from mpi4py import MPI
import ufl
import basix
from basix.ufl import element

import numpy as np
import pyvista
from sklearn.neighbors import KDTree
import collections
import gmsh, meshio

from scripts_py.dolfinx_utils import DolfinUtils


'''
TODO
Dolfinx support the following cell types:
    - triangle
    - quadrilateral
    - tetrahedron
    - hexahedron
please add support for other cell types.
if using quadrilateral or hexahedron, may be degree of velocity and pressure in CFD can be set to 1 ?

TODO
use topology optimization generate fluid path rather than path search
'''


class MeshUtils(object):
    @staticmethod
    def read_msh(msh_file: str, dim: int) -> (dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags):  # type: ignore
        assert msh_file.endswith(".msh")
        domain, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)
        return domain, cell_tags, facet_tags

    @staticmethod
    def msh_to_XDMF(msh_file: str, xdmf_file: str, dim: int, name="model"):
        """
        msh_file: .msh, .gmsh2, .gmsh
        if operation fail, please check:
            if dim=3: whether all necessary volumes had been added to physical group
            if dim=2: whether all necessary surfaces had been added to physical group
        params:
            dim: dimension of model
            mesh_dim: dimension of mesh
        """
        assert xdmf_file.endswith(".xdmf")
        doamin, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim)
        doamin.name = name
        cell_tags.name = f"{doamin.name}_cells"
        facet_tags.name = f"{doamin.name}_facets"
        with XDMFFile(doamin.comm, xdmf_file, "w") as f:
            doamin.topology.create_connectivity(dim - 1, dim)
            f.write_mesh(doamin)
            f.write_meshtags(cell_tags, doamin.geometry)
            f.write_meshtags(facet_tags, doamin.geometry)

    @staticmethod
    def read_XDMF(xdmf_file: str, name="model") -> (dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags):  # type: ignore
        assert xdmf_file.endswith(".xdmf")
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as f:
            domain = f.read_mesh(name=name)
            cell_tags = f.read_meshtags(domain, name=f"{name}_cells")
        domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )
        # only after creating the connectivity, facet_tags can be extracted
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as f:
            facet_tags = f.read_meshtags(domain, name=f"{name}_facets")
        return domain, cell_tags, facet_tags

    @staticmethod
    def save_XDMF(
        xdmf_file: str,
        domain: dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        cell_tags: dolfinx.mesh.MeshTags,
        name="model"
    ):
        """Save后Points的顺序会改变,原因不明"""
        assert xdmf_file.endswith(".xdmf")
        domain.name = name
        cell_tags.name = f"{domain.name}_cells"
        facet_tags.name = f"{domain.name}_facets"
        with XDMFFile(domain.comm, xdmf_file, "w") as f:
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )
            f.write_mesh(domain)
            f.write_meshtags(cell_tags, domain.geometry)
            f.write_meshtags(facet_tags, domain.geometry)

    @staticmethod
    def generate_msh22_file(
        mesh: 'MeshObj', 
        filename: str, 
        face_tags: dict[str, int],
        cell_tags: dict[str, int],
    ):
        element_type = {
            'Line': [1, 2],
            'Face': [2, 2],
            'Cell': [4, 2]
        }
        
        with open(filename, 'w') as f:
            # MeshFormat
            f.write('$MeshFormat\n')
            f.write('2.2 0 8\n')  # Version 2.2, ASCII (0), double precision (8 bytes)
            f.write('$EndMeshFormat\n')
            
            # PhysicalNames
            num_physical_groups = len(face_tags) + len(cell_tags)
            f.write('$PhysicalNames\n')
            f.write(f'{num_physical_groups}\n')
            for name, phys_id in face_tags.items():
                if mesh.tdim == 2:
                    f.write(f'1 {phys_id} "{name}"\n')
                elif mesh.tdim == 3:
                    f.write(f'2 {phys_id} "{name}"\n')
            for name, phys_id in cell_tags.items():
                if mesh.tdim == 2:
                    f.write(f'2 {phys_id} "{name}"\n')
                elif mesh.tdim == 3:
                    f.write(f'3 {phys_id} "{name}"\n')
            f.write('$EndPhysicalNames\n')
            
            # Nodes
            f.write('$Nodes\n')
            f.write(f'{mesh.num_points}\n')
            for i in range(mesh.num_points):
                x, y, z = mesh.points[i]
                f.write(f'{i+1} {x:.8f} {y:.8f} {z:.8f}\n')
            f.write('$EndNodes\n')
            
            # Elements
            exterior_facet_conn = mesh.facets_to_points[mesh.exterior_facet_idxs]
            cell_conn = mesh.cells_to_points
            exterior_facet_phy_group = mesh.facet_tags.values
            cell_phy_group = mesh.cell_tags.values
            
            total_elements = cell_conn.shape[0] + exterior_facet_conn.shape[0]
            f.write('$Elements\n')
            f.write(f'{total_elements}\n')
            
            elm_id = 1        
            for i in range(exterior_facet_conn.shape[0]):
                phys_idx = exterior_facet_phy_group[i]
                conn = exterior_facet_conn[i]
                if mesh.tdim == 2:
                    elem_type = element_type['Line']
                elif mesh.tdim == 3:
                    elem_type = element_type['Face']
                
                nodes_str = ' '.join(str(n + 1) for n in conn)  # be careful add 1 to node index
                f.write(f'{elm_id} {elem_type[0]} {elem_type[1]} {phys_idx} {phys_idx} {nodes_str}\n')
                elm_id += 1
            
            for i in range(cell_conn.shape[0]):
                phys_idx = cell_phy_group[i]
                conn = cell_conn[i]
                if mesh.tdim == 2:
                    elem_type = element_type['Face']
                elif mesh.tdim == 3:
                    elem_type = element_type['Cell']
                
                nodes_str = ' '.join(str(n + 1) for n in conn)  # be careful add 1 to node index
                f.write(f'{elm_id} {elem_type[0]} {elem_type[1]} {phys_idx} {phys_idx} {nodes_str}\n')
                elm_id += 1
            
            f.write('$EndElements\n')

    @staticmethod
    def convert_mesh_to_msh(
        points: np.ndarray, 
        cells_2d: np.ndarray, 
        msh_file: str, 
        temp_dir: str,
        cells_3d: np.ndarray|None=None,
        xdmf_file: str|None=None,
    ):
        gdim = 2
        if cells_2d.shape[1] == 3:
            cell_type_2d = pyvista.CellType.TRIANGLE
        elif cells_2d.shape[1] == 4:
            cell_type_2d = pyvista.CellType.QUAD
        else:
            raise ValueError(f"cells_2d.shape[1] must be 3 or 4, but got {cells_2d.shape[1]}.")
        cell_dict = {cell_type_2d: cells_2d}
        
        if cells_3d is not None:
            gdim = 3
            if cells_3d.shape[1] == 4:
                cell_type_3d = pyvista.CellType.TETRA
            elif cells_3d.shape[1] == 8:
                cell_type_3d = pyvista.CellType.HEXAHEDRON
            else:
                raise ValueError(f"cells_3d.shape[1] must be 4 or 8, but got {cells_3d.shape[1]}.")
            cell_dict[cell_type_3d] = cells_3d
        grid = pyvista.UnstructuredGrid(cell_dict, points)
        
        temp_grid_vtu = os.path.join(temp_dir, 'temp_grid.vtu')
        temp_grid_msh = os.path.join(temp_dir, 'temp_grid.msh')
        
        grid.save(temp_grid_vtu, binary=False)
        mesh = meshio.read(temp_grid_vtu)
        meshio.write(temp_grid_msh, mesh, file_format="gmsh22", binary=False)
        
        # ------ postprocess through gmsh ------
        gmsh.initialize()
        gmsh.merge(temp_grid_msh)
        entity_dimtags: list[tuple[int, int]] = gmsh.model.getEntities(gdim)
        assert len(entity_dimtags) == 1, "Only one volume is allowed"
        
        vol_dim, vol_tag = entity_dimtags[0]
        gmsh.model.addPhysicalGroup(vol_dim, [vol_tag], name='volume')
        # gmsh.model.geo.synchronize()  # must remove
        gmsh.write(msh_file)
        gmsh.finalize()
        
        if xdmf_file is not None:
            MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=gdim)

    @staticmethod
    def get_topology_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim

    @staticmethod
    def get_facet_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim - 1

    @staticmethod
    def get_cell_type(domain: dolfinx.mesh.Mesh) -> Literal[
        dolfinx.mesh.CellType.triangle,  # type: ignore
        dolfinx.mesh.CellType.quadrilateral,  # type: ignore
        dolfinx.mesh.CellType.tetrahedron,  # type: ignore
        dolfinx.mesh.CellType.hexahedron # type: ignore
    ]:
        cell_type = domain.topology.cell_type
        assert cell_type in [
            dolfinx.mesh.CellType.triangle,  # type: ignore
            dolfinx.mesh.CellType.quadrilateral,  # type: ignore
            dolfinx.mesh.CellType.tetrahedron,  # type: ignore
            dolfinx.mesh.CellType.hexahedron # type: ignore
        ], f"cell_type must be one of {dolfinx.mesh.CellType.triangle},\
        {dolfinx.mesh.CellType.quadrilateral},\
        {dolfinx.mesh.CellType.tetrahedron},\
        {dolfinx.mesh.CellType.hexahedron}."
        return cell_type

    @staticmethod
    def mesh_to_vtk(domain: dolfinx.mesh.Mesh, tdim: int|None=None, entities=None):
        cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim, entities)
        cell_type = MeshUtils.get_cell_type(domain)
        if cell_type == dolfinx.mesh.CellType.triangle:
            vista_cell_types = [pyvista.CellType.TRIANGLE] * len(cell_types)
        elif cell_type == dolfinx.mesh.CellType.quadrilateral:
            vista_cell_types = [pyvista.CellType.QUAD] * len(cell_types)
        elif cell_type == dolfinx.mesh.CellType.tetrahedron:
            vista_cell_types = [pyvista.CellType.TETRA] * len(cell_types)
        elif cell_type == dolfinx.mesh.CellType.hexahedron:
            vista_cell_types = [pyvista.CellType.HEXAHEDRON] * len(cell_types)
        else:
            raise ValueError(f"[Error]: Invalid cell type: {cell_type}.")
        grid = pyvista.UnstructuredGrid(cells, vista_cell_types, geometry)        
        return grid

    @staticmethod
    def extract_entity_indices(
        domain: dolfinx.mesh.Mesh,
        style: Literal["cell", "facet"] = "facet",
        marker: Union[Callable, int, None] = None,
        mesh_tags: Union[dolfinx.mesh.MeshTags, None] = None,
    ) -> np.ndarray:
        """
        Args:
            mesh_tags contain indices and values, indices is the entity index, values is group id
        """
        tdim = domain.topology.dim
        if style == "facet":
            if marker is None:
                # return whole boundary entity idxs
                domain.topology.create_connectivity(tdim - 1, tdim)
                entities_idxs = dolfinx.mesh.exterior_facet_indices(domain.topology)
            else:
                if isinstance(marker, Callable):
                    entities_idxs = dolfinx.mesh.locate_entities_boundary(
                        domain, tdim - 1, marker=marker
                    )
                elif isinstance(marker, int):
                    assert mesh_tags is not None
                    entities_idxs = mesh_tags.find(marker)
                else:
                    raise NotImplementedError
        elif style == "cell":
            assert marker is not None
            entities_idxs = dolfinx.mesh.locate_entities(domain, tdim, marker=marker)  # type: ignore
        else:
            raise ValueError("[Error]: Invalid Style.")
        return entities_idxs

    @staticmethod
    def extract_entity_dofs(
        V: dolfinx.fem.FunctionSpace|tuple[dolfinx.fem.FunctionSpace, dolfinx.fem.FunctionSpace], dim: int, entity_idxs: np.ndarray
    ) -> np.ndarray:
        """
        dof为dolfin.fem.function数组内的索引idx
        """
        return dolfinx.fem.locate_dofs_topological(
            V, entity_dim=dim, entities=entity_idxs
        )

    @staticmethod
    def define_facet_norm(domain: dolfinx.mesh.Mesh):
        return ufl.FacetNormal(domain)

    @staticmethod
    def define_coordinate(domain: dolfinx.mesh.Mesh):
        return ufl.SpatialCoordinate(domain)

    @staticmethod
    def define_ds(domain: dolfinx.mesh.Mesh, facets_tag: dolfinx.mesh.MeshTags | None = None):
        """define area of boundary"""
        return ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)

    @staticmethod
    def define_dx(domain: dolfinx.mesh.Mesh):
        """define volume of mesh"""
        return ufl.Measure("dx", domain=domain)

    @staticmethod
    def define_meshTag(
        domain: dolfinx.mesh.Mesh, marker: int, dim: int, location_func: Callable
    ) -> dolfinx.mesh.MeshTags:
        indices = dolfinx.mesh.locate_entities_boundary(domain, dim, location_func)
        markers = np.full(indices.shape[0], fill_value=marker).astype(np.int32)
        mesh_tag = dolfinx.mesh.meshtags(domain, dim, indices, markers)
        return mesh_tag

    @staticmethod
    def cell_indices_of_points(
        mesh: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], pcd_xyz: np.ndarray
    ) -> list[list[int]] | list[int]:
        """
        find the cell indices of points in the mesh
        """
        if isinstance(mesh, pyvista.UnstructuredGrid):
            cell_idxs = mesh.find_containing_cell(pcd_xyz)  # 只能返回最接近1个cell
        elif isinstance(mesh, dolfinx.mesh.Mesh):
            bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
            cell_candidates = geometry.compute_collisions_points(bb_tree, pcd_xyz)
            colliding_cells = geometry.compute_colliding_cells(
                mesh, cell_candidates, pcd_xyz
            )
            cell_idxs = []
            for i in range(pcd_xyz.shape[0]):
                cell_idxs.append(
                    list(colliding_cells.links(i))
                )  # 可以返回包含该point的所有cell
        else:
            raise ValueError("[Error]: Non-valid Mesh.")
        return cell_idxs  # type: ignore

    @staticmethod
    def cell_connectivity(
        mesh: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], dim: int
    ) -> np.ndarray:
        ''' return connectivity of cells '''
        if isinstance(mesh, pyvista.UnstructuredGrid):
            cell_types = list(mesh.cells_dict.keys())
            if dim == 2:
                if (pyvista.CellType.TRIANGLE in cell_types) and (pyvista.CellType.QUAD in cell_types):
                    raise ValueError("[Error]: Mixed triangle and quadrilateral cells.")
                
                if pyvista.CellType.TRIANGLE in cell_types:
                    return mesh.cells.reshape(-1, 4)[:, 1:]
                elif pyvista.CellType.QUAD in cell_types:
                    return mesh.cells.reshape(-1, 5)[:, 1:]
                else:
                    raise ValueError("[Error]: Invalid cell type.")
            else:
                if (pyvista.CellType.TETRA in cell_types) and (pyvista.CellType.HEXAHEDRON in cell_types):
                    raise ValueError("[Error]: Mixed tetrahedron and hexahedron cells.")
                
                if pyvista.CellType.TETRA in cell_types:
                    return mesh.cells.reshape(-1, 5)[:, 1:]
                elif pyvista.CellType.HEXAHEDRON in cell_types:
                    return mesh.cells.reshape(-1, 9)[:, 1:]
                else:
                    raise ValueError("[Error]: Invalid cell type.")
            
        elif isinstance(mesh, dolfinx.mesh.Mesh):
            con = []
            adjacency_list = mesh.topology.connectivity(mesh.topology.dim, 0)
            for i in range(adjacency_list.num_nodes):
                con.append(adjacency_list.links(i))
            con = np.array(con)
            return con
        else:
            raise ValueError("[Error]: Non-valid Mesh Type.")

    @staticmethod
    def facet_connectivity(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
        con = []
        adjacency_list = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
        for i in range(adjacency_list.num_nodes):
            con.append(adjacency_list.links(i))
        con = np.array(con)
        return con

    @staticmethod
    def line_connectivity(mesh: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid]) -> np.ndarray:
        grid = MeshUtils.mesh_to_vtk(mesh) if isinstance(mesh, dolfinx.mesh.Mesh) else mesh
        return grid.extract_all_edges().lines.reshape((-1, 3))[:, 1:3]

    @staticmethod
    def move(domain: dolfinx.mesh.Mesh, coord_dif: np.ndarray):
        assert domain.geometry.x.shape == coord_dif.shape
        domain.geometry.x[:] += coord_dif


class MeshObj(object):
    def __init__(
        self,
        xdmf_file: str| None = None,
        mesh_name="model",
        domain: dolfinx.mesh.Mesh | None = None,
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        facet_tags: dolfinx.mesh.MeshTags | None = None
    ):
        if xdmf_file is not None:
            self.domain, self.cell_tags, self.facet_tags = MeshUtils.read_XDMF(xdmf_file, name=mesh_name)
            self.mesh_file = xdmf_file
        else:
            assert domain is not None, 'domain is required.'
            self.domain: dolfinx.mesh.Mesh = domain
            self.cell_tags: dolfinx.mesh.MeshTags = cell_tags  # type: ignore
            self.facet_tags: dolfinx.mesh.MeshTags = facet_tags  # type: ignore
            self.mesh_file = None
        
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.grid = MeshUtils.mesh_to_vtk(self.domain, self.tdim)
        self.num_points = self.domain.geometry.x.shape[0]
        self.num_cells = self.grid.n_cells

        self.cells_to_points = MeshUtils.cell_connectivity(self.grid, self.tdim)
        cell_counter: collections.Counter = collections.Counter(self.cells_to_points.flatten().tolist())
        self.occurrences = np.array([cell_counter[i] for i in range(self.num_points)])
        self.facets_to_points = MeshUtils.facet_connectivity(self.domain)
        self.exterior_facet_idxs = MeshUtils.extract_entity_indices(self.domain, style='facet')
        
        self.with_vol_check = True
        self.vol_change = 0.2
        self.with_angle_check = True
        self.angle_tol_max = 160.0
        self.angle_tol_min = 20.0

    @property
    def points(self):
        return self.domain.geometry.x
    
    def cell_types(self) -> basix.CellType:
        return self.domain.basix_cell()  # type: ignore
    
    def cell_name(self) -> str:
        return self.domain.topology.cell_name()
    
    def set_points(self, points: np.ndarray):
        assert points.shape == (self.num_points, 3)
        self.domain.geometry.x[:] = points
    
    def deform(self, displacement_np: np.ndarray):
        assert displacement_np.ndim == 2, f"displacement_np.ndim must be 2, but got {displacement_np.ndim}."
        assert self.domain.geometry.x.shape[0] == displacement_np.shape[0], f"self.domain.geometry.x.shape[0] must be {self.domain.geometry.x.shape[0]}, but got {displacement_np.shape[0]}."
        assert displacement_np.shape[1] in [2, 3], f"displacement_np.shape[1] must be 2 or 3, but got {displacement_np.shape[1]}."
        
        if displacement_np.shape[1] == 2:
            displacement_np = np.concatenate([displacement_np, np.zeros((displacement_np.shape[0], 1))], axis=1)
        self.domain.geometry.x[:] += displacement_np

    def is_collision(self) -> dict:
        cell_idxs: list[list[int]] = MeshUtils.cell_indices_of_points(self.domain, self.points)  # type: ignore
        cell_counts = []
        for idx_list in cell_idxs:
            cell_counts.append(len(idx_list))
        is_valid = np.all(cell_counts == self.occurrences)
        return {
            'is_valid': is_valid,
            'log': f'[Info]: Non-valid deformation Since collision.' if not is_valid else ''
        }

    def estimate_mesh_quality(self, obj: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], quality_measure: str) -> np.ndarray:
        """
        quality_measure:
            - area
            - radius_ratio
            - skew (Fail)
            - volume
            - max_angle
            - min_angle
            - condition
        """
        grid = MeshUtils.mesh_to_vtk(obj, self.tdim) if isinstance(obj, dolfinx.mesh.Mesh) else obj
        quality = grid.compute_cell_quality(quality_measure=quality_measure, progress_bar=False, null_value=-1.0)
        return quality['CellQuality']  # type: ignore

    def is_valid_volume_change(self, displacement_np: np.ndarray, vol_change: float) -> dict:
        vol_orig = self.estimate_mesh_quality(self.grid, 'area' if self.tdim == 2 else 'volume')
        self.deform(displacement_np)
        vol_deformed = self.estimate_mesh_quality(self.grid, 'area' if self.tdim == 2 else 'volume')
        self.deform(displacement_np * -1.0)
        
        if np.unique(vol_orig).shape[0] == 1:
            return {'is_valid': False, 'skip': True, 'log': '[Warning]: All cells have -1.0 volume.'}
        elif np.unique(vol_deformed).shape[0] == 1:
            return {'is_valid': False, 'skip': True, 'log': '[Warning]: All cells have -1.0 volume.'}
        
        vol_dif = vol_deformed / vol_orig
        min_det, max_det = np.min(vol_dif) - 1.0, np.max(vol_dif) - 1.0
        is_valid = (min_det >= -vol_change) and (max_det <= vol_change)
        
        return {
            'is_valid': is_valid, 'skip': False,
            'min_det': min_det, 'max_det': max_det,
            'log': f'[Info]: Non-valid deformation since volume change {min_det:.2f}/{max_det:.2f} too large.' if not is_valid else ''
        }

    def is_valid_angle(self, displacement_np: np.ndarray, tol_max=160.0, tol_min=20.0) -> dict:
        self.deform(displacement_np)
        max_angles = self.estimate_mesh_quality(self.grid, 'max_angle')
        min_angles = self.estimate_mesh_quality(self.grid, 'min_angle')
        self.deform(displacement_np * -1.0)
        
        if np.unique(max_angles).shape[0] == 1:
            return {'is_valid': False, 'skip': True, 'log': '[Warning]: All cells have -1.0 max angle.'}
        elif np.unique(min_angles).shape[0] == 1:
            return {'is_valid': False, 'skip': True, 'log': '[Warning]: All cells have -1.0 min angle.'}
        
        is_valid = (np.min(min_angles) >= tol_min) and (np.max(max_angles) <= tol_max)
        return {
            'is_valid': is_valid, 'skip': False,
            'min_angle': np.min(min_angles), 'max_angle': np.max(max_angles),
            'log': f'[Info]: Non-valid deformation since angle '
                  f'{np.min(min_angles):.2f}/{np.max(max_angles):.2f} out of range.' if not is_valid else ''
        }

    def deform_safe(self, displacement_np: np.ndarray) -> dict:
        if self.with_vol_check:
            vol_check = self.is_valid_volume_change(displacement_np, self.vol_change)
            if (not vol_check['is_valid']) and (not vol_check['skip']):
                return vol_check
            # if vol_check['skip']:
            #     print(vol_check['log'])

        if self.with_angle_check:
            angle_check = self.is_valid_angle(displacement_np, tol_max=self.angle_tol_max, tol_min=self.angle_tol_min)
            if (not angle_check['is_valid']) and (not angle_check['skip']):
                return angle_check
            # if angle_check['skip']:
            #     print(angle_check['log'])

        self.deform(displacement_np)
        collision_check = self.is_collision()
        if not collision_check['is_valid']:
            self.deform(displacement_np * -1.0)
            return collision_check
        return {'is_valid': True}

    def deform_line_search(self, direction_np: np.ndarray, scales: list[float]):
        for scale in scales:
            safe_check = self.deform_safe(direction_np * scale)
            if safe_check['is_valid']:
                return safe_check
        return {'is_valid': False, 'log': '[Info]: Non-valid deformation since line search failed.'}

    def record_point_value(self, key: str, value: np.ndarray):
        self.grid.point_data[key] = value
    
    def record_cell_value(self, key: str, value: np.ndarray):
        self.grid.cell_data[key] = value

    def detect_obstacle(self, obs_pcd: np.ndarray, obs_radius: float, unit_lr: float) -> dict:
        tree = KDTree(self.points[:, :self.tdim])
        obs_pcd = obs_pcd[:, :self.tdim]
        
        idxs_list, dists_list = tree.query_radius(obs_pcd, r=obs_radius + unit_lr, return_distance=True)

        conflict_mesh_idxs, conflict_obs_idxs = [], []
        for obs_idx, (mesh_idxs, dists) in enumerate(zip(idxs_list, dists_list)):
            if len(mesh_idxs) == 0:
                continue
            conflict_mesh_idxs.append(np.array(mesh_idxs))
            conflict_obs_idxs.append(np.full(shape=(len(mesh_idxs)), fill_value=obs_idx, dtype=int))

        if len(conflict_mesh_idxs) > 0:
            conflict_mesh_idxs = np.concatenate(conflict_mesh_idxs, axis=-1)
            conflict_obs_idxs = np.concatenate(conflict_obs_idxs, axis=-1)
            return {
                'is_conflict': True,
                'conflict_mesh_idxs': conflict_mesh_idxs,
                'conflict_obs_idxs': conflict_obs_idxs
            }
        else:
            return {'is_conflict': False}

    def save_grid(self, file: str):
        assert file.endswith('.vtu')
        self.grid.save(file)

    def update_grid(self):
        self.grid = MeshUtils.mesh_to_vtk(self.domain, self.tdim)

    def extract_surface_indices(self, marker: int) -> np.ndarray:
        '''
        return indices of surface points of specific marker
        '''
        return MeshUtils.extract_entity_dofs(
            DolfinUtils.point_scalar_function_space(self.domain), self.fdim,
            MeshUtils.extract_entity_indices(
                self.domain, style='facet', mesh_tags=self.facet_tags, marker=marker
            )
        )


def debug_convert_msh22():
    msh_file = "/home/admin123456/Desktop/work/gmsh_test/t36/model.msh"
    xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t36/model.xdmf"
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=2)
    mesh = MeshObj(xdmf_file=xdmf_file)
    MeshUtils.generate_msh22_file(
        mesh=mesh, 
        filename="/home/admin123456/Desktop/work/gmsh_test/t36/model_v2.msh", 
        face_tags={
            "inlet": 22,
            "outlet": 23,
            "wall": 24,
        }, 
        cell_tags={
            "volume": 21
        }
    )


    MeshUtils.msh_to_XDMF(
        msh_file="/home/admin123456/Desktop/work/gmsh_test/t36/model_v2.msh", 
        xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t36/model_v2.xdmf", 
        dim=2
    )
    mesh_v2 = MeshObj(xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t36/model_v2.xdmf")
    inlet_indices = mesh_v2.extract_surface_indices(22)
    outlet_indices = mesh_v2.extract_surface_indices(23)
    wall_indices = mesh_v2.extract_surface_indices(24)

    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh_v2.grid, style='wireframe')
    plotter.add_mesh(pyvista.PointSet(mesh_v2.grid.points[inlet_indices]), color='red', style='points')
    plotter.add_mesh(pyvista.PointSet(mesh_v2.grid.points[outlet_indices]), color='blue', style='points')
    plotter.add_mesh(pyvista.PointSet(mesh_v2.grid.points[wall_indices]), color='green', style='points')
    plotter.show()


if __name__ == "__main__":
    debug_convert_msh22()
