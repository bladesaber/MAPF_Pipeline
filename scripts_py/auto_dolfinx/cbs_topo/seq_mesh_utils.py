import dolfinx
from mpi4py import MPI

import numpy as np
from functools import partial
from itertools import combinations
from sklearn import neighbors
from sklearn.neighbors import KDTree
from tqdm import tqdm
import pyvista

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary


def dist_pcd2center(pcd: np.ndarray, center_point: np.ndarray):
    return np.linalg.norm(pcd - center_point, axis=1)


def location_func(
    x: np.ndarray,
    face_center: np.ndarray,
    face_norm_vector: np.ndarray,
    face_radius: float,
):
    if x.shape[0] in [2, 3]:
        position = x.T
    else:
        position = x
    pcd2center_dist = dist_pcd2center(position, face_center)
    is_on_surface = pcd2center_dist <= face_radius
    return is_on_surface


def generate_mesh(
    boundary_list: list[AutoFluidBoundary],
    left_bottom_corner: np.ndarray,
    right_top_corner: np.ndarray,
    num_cells: list[int],
    xdmf_file: str,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.tetrahedron,
    comm=MPI.COMM_WORLD,
):
    domain = dolfinx.mesh.create_box(
        comm, [left_bottom_corner, right_top_corner], num_cells, cell_type=cell_type
    )

    mesh_tag_indices = []
    mesh_tag_values = []
    wall_marker = None
    wall_count = 0
    for boundary in boundary_list:
        if boundary.boundary_type in ["velocity", "pressure"]:
            mesh_tag = MeshUtils.define_meshTag(
                domain=domain,
                marker=boundary.marker,
                dim=domain.topology.dim - 1,
                location_func=partial(
                    location_func,
                    face_center=boundary.face_center,  # type: ignore
                    face_norm_vector=boundary.face_norm_vector,  # type: ignore
                    face_radius=boundary.face_radius,  # type: ignore
                ),
            )
            mesh_tag_indices.append(mesh_tag.indices)
            mesh_tag_values.append(mesh_tag.values)

        elif boundary.boundary_type == "wall":
            wall_marker = boundary.marker
            wall_count += 1

        else:
            raise ValueError(
                f"[Error]: Invalid Boundary Type: {boundary.boundary_type}."
            )

    assert wall_count == 1, f"[Error]: Only one wall boundary is allowed."

    inout_tag_indices = np.concatenate(mesh_tag_indices)
    inout_tag_values = np.concatenate(mesh_tag_values)

    wall_indices = MeshUtils.extract_entity_indices(domain, style="facet", marker=None)
    wall_indices = np.array(
        list(set(wall_indices.tolist()) - set(inout_tag_indices.tolist()))
    ).astype(np.int32)

    mesh_tag_indices = np.concatenate([inout_tag_indices, wall_indices]).astype(
        np.int32
    )
    mesh_tag_values = np.concatenate(
        [inout_tag_values, np.full_like(wall_indices, wall_marker)]
    ).astype(np.int32)

    grid = MeshUtils.mesh_to_vtk(domain, domain.topology.dim)
    facet_tags = dolfinx.mesh.meshtags(
        domain, domain.topology.dim - 1, mesh_tag_indices, mesh_tag_values
    )
    cell_tags = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim,
        np.arange(0, grid.n_cells, 1).astype(np.int32),
        np.full(shape=(grid.n_cells,), fill_value=9).astype(np.int32),
    )

    MeshUtils.save_XDMF(
        xdmf_file=xdmf_file, domain=domain, facet_tags=facet_tags, cell_tags=cell_tags
    )

    # return MeshObj(xdmf_file=xdmf_file)


class SeqMeshObj(MeshObj):
    def __init__(
        self,
        xdmf_file: str| None,
        mesh_name="model",
        domain: dolfinx.mesh.Mesh | None = None,
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        facet_tags: dolfinx.mesh.MeshTags | None = None,
        cell_mode: bool = False
    ):
        super().__init__(xdmf_file, mesh_name, domain, cell_tags, facet_tags)
        assert MeshUtils.get_cell_type(self.domain) in [
            dolfinx.mesh.CellType.tetrahedron,  # type: ignore
            dolfinx.mesh.CellType.triangle,  # type: ignore
        ], "[Error]: Cell type must be tetrahedron or triangle."
        
        self.cell_mode = cell_mode
        self.points_neighbors: dict[int, list[int]] = {}
        self.points_to_cells: dict[int, list[int]] = {}
        self.cells_neighbors: dict[int, list[int]] = {}
        if self.cell_mode:
            self.seq_dense_tag = np.zeros((self.num_cells,))
        else:
            self.seq_dense_tag = np.zeros((self.num_points,))
    
    @property
    def cell_centers(self):
        return self.grid.cell_centers().points
    
    def create_neighbors_info(self):
        self.points_neighbors.clear()
        self.points_to_cells.clear()
        if self.tdim == 2:
            edge_to_cells = {}
        else:
            face_to_cells = {}
        
        for cell_idx, p_idxs in tqdm(enumerate(self.cells_to_points), desc="Creating points neighbors"):
            for p_idx in p_idxs:
                if p_idx not in self.points_to_cells.keys():
                    self.points_to_cells[p_idx] = []
                self.points_to_cells[p_idx].append(cell_idx)
            
                if p_idx not in self.points_neighbors.keys():
                    self.points_neighbors[p_idx] = []

            for p0, p1 in combinations(p_idxs, 2):
                self.points_neighbors[p0].append(p1)
                self.points_neighbors[p1].append(p0)
                
                # ------ create edge to cells mapping ------
                if self.tdim == 2:
                    edge_min, edge_max = np.sort([p0, p1])
                    edge_tag = f"{edge_min}_{edge_max}"
                    if edge_tag not in edge_to_cells.keys():
                        edge_to_cells[edge_tag] = []
                    edge_to_cells[edge_tag].append(cell_idx)
            
            if self.tdim == 3:
                for p0, p1, p2 in combinations(p_idxs, 3):
                    face_min, face_mid, face_max = np.sort([p0, p1, p2])
                    face_tag = f"{face_min}_{face_mid}_{face_max}"
                    if face_tag not in face_to_cells.keys():
                        face_to_cells[face_tag] = []
                    face_to_cells[face_tag].append(cell_idx)
        
        for p_idx, neighbor_idxs in tqdm(self.points_neighbors.items(), desc="Removing duplicate neighbors"):
            self.points_neighbors[p_idx] = list(set(neighbor_idxs))
        
        self.cells_neighbors.clear()
        if self.tdim == 2:
            for edge_tag, cell_idxs in tqdm(edge_to_cells.items(), desc="Creating cells neighbors"):
                for cell_idx in cell_idxs:
                    if cell_idx not in self.cells_neighbors.keys():
                        self.cells_neighbors[cell_idx] = []
                
                for c0, c1 in combinations(cell_idxs, 2):
                    self.cells_neighbors[c0].append(c1)
                    self.cells_neighbors[c1].append(c0)
                    
        else:
            for face_tag, cell_idxs in tqdm(face_to_cells.items(), desc="Creating cells neighbors"):
                for cell_idx in cell_idxs:
                    if cell_idx not in self.cells_neighbors.keys():
                        self.cells_neighbors[cell_idx] = []
                
                for c0, c1 in combinations(cell_idxs, 2):
                    self.cells_neighbors[c0].append(c1)
                    self.cells_neighbors[c1].append(c0)
        
        for c_idx, neighbor_idxs in tqdm(self.cells_neighbors.items(), desc="Removing duplicate neighbors"):
            self.cells_neighbors[c_idx] = list(set(neighbor_idxs))
    
    def create_seq_tag(self, source_xyzs: np.ndarray, thresholds_map: np.ndarray):
        """
        source_xyzs:
            if self.cell_mode is True, source_xyzs is cell centers
            if self.cell_mode is False, source_xyzs is points
        thresholds_map: [0.0, >0.0], 0.0表示该点不可达, >0.0表示该点可达 
        """
        if self.cell_mode:
            tree = KDTree(self.cell_centers[:, :self.tdim])
        else:
            tree = KDTree(self.points[:, :self.tdim])
        idxs_list = tree.query(source_xyzs, k=1, return_distance=False)
        idxs_list = np.concatenate(idxs_list, axis=-1)
        
        if self.cell_mode:
            seq_dense_tag = np.zeros((self.num_cells,))
        else:
            seq_dense_tag = np.zeros((self.num_points,))
        seq_dense_tag[idxs_list] = 1
        source_idxs: np.ndarray = idxs_list
        layer_tag = 2
        
        while True:
            next_idxs = []
            for idx in source_idxs:
                if self.cell_mode:
                    next_idxs.append(self.cells_neighbors[idx])
                else:
                    next_idxs.append(self.points_neighbors[idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            
            next_idxs = np.unique(next_idxs)
            next_idxs = next_idxs[seq_dense_tag[next_idxs] == 0.0]
            next_idxs = next_idxs[thresholds_map[next_idxs] > 0.0]
            
            if next_idxs.shape[0] == 0:
                assert np.all(seq_dense_tag[thresholds_map > 0.0] > 0), "[Error]: Some reachable points are not reachable."
                break
            
            seq_dense_tag[next_idxs] = layer_tag
            source_idxs = next_idxs
            layer_tag += 1
        
        self.seq_dense_tag = seq_dense_tag
    
    def extract_cluster(self, idx: int, included_thresholds: np.ndarray):
        cluster_idxs = [np.array([idx])]
        source_idxs = np.array([idx])
        
        threshold_np = included_thresholds.copy()
        threshold_np[idx] = 0.0
        
        while True:
            next_idxs = []
            for idx in source_idxs:
                if self.cell_mode:
                    next_idxs.append(self.cells_neighbors[idx])
                else:
                    next_idxs.append(self.points_neighbors[idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            next_idxs = np.unique(next_idxs)
            
            next_idxs = next_idxs[threshold_np[next_idxs] > 0.0]
            if next_idxs.shape[0] == 0:
                break
            
            threshold_np[next_idxs] = 0.0
            source_idxs = next_idxs
            cluster_idxs.append(next_idxs)
        
        return np.concatenate(cluster_idxs, axis=-1)
    
    def extract_all_clusters(self, source_idxs: np.ndarray):
        if self.cell_mode:
            included_thresholds = np.zeros((self.num_cells,))
        else:
            included_thresholds = np.zeros((self.num_points,))
        included_thresholds[source_idxs] = 1.0
        
        clusters = []
        while True:
            idx = np.random.choice(np.where(included_thresholds == 1.0)[0], size=1)[0]
            cluster_idxs = self.extract_cluster(idx, included_thresholds)
            clusters.append(cluster_idxs)
            included_thresholds[cluster_idxs] = 0.0
            
            if np.all(included_thresholds == 0.0):
                break
        
        return clusters

    def draw_seq_tag(self):
        if self.cell_mode:
            self.grid.cell_data["seq_dense_tag"] = self.seq_dense_tag
            cell_centers_mesh: pyvista.PointSet = self.grid.cell_centers()  # type: ignore
            cell_centers_mesh.point_data["seq_dense_tag"] = self.seq_dense_tag
            cell_centers_mesh.warp_by_scalar("seq_dense_tag")
            
            plotter = pyvista.Plotter()
            plotter.add_mesh(cell_centers_mesh, style='surface')
            plotter.add_mesh(self.grid, style='wireframe')
            plotter.show()
            
        else:
            self.grid.point_data["seq_dense_tag"] = self.seq_dense_tag
            self.grid.warp_by_scalar("seq_dense_tag")
            plotter = pyvista.Plotter()
            plotter.add_mesh(self.grid, style='wireframe')
            plotter.show()

    def draw_point_neighbors(self):
        random_p_idx = np.random.randint(0, self.num_points)
        random_p_neighbors = self.points_neighbors[random_p_idx]
        random_p_of_cells = self.points_to_cells[random_p_idx]
        cell_centers = self.grid.cell_centers().points
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.grid, color='black', style='wireframe')
        plotter.add_mesh(
            pyvista.PointSet(np.array([self.points[random_p_idx, :]])), 
            color='red', point_size=10
        )
        plotter.add_mesh(
            pyvista.PointSet(self.points[random_p_neighbors, :]), 
            color='blue', point_size=8
        )
        plotter.add_mesh(
            pyvista.PointSet(cell_centers[random_p_of_cells, :]), 
            color='green', point_size=8
        )
        plotter.show()

    def draw_cell_neighbors(self):
        random_c_idx = np.random.randint(0, self.num_cells)
        random_c_points = self.cells_to_points[random_c_idx]
        random_c_neighbors = self.cells_neighbors[random_c_idx]
        cell_centers = self.grid.cell_centers().points
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.grid, color='black', style='wireframe')
        plotter.add_mesh(pyvista.PointSet(np.array([cell_centers[random_c_idx, :]])), color='red', point_size=8)
        plotter.add_mesh(pyvista.PointSet(self.points[random_c_points, :]), color='blue', point_size=8)
        plotter.add_mesh(pyvista.PointSet(cell_centers[random_c_neighbors, :]), color='green', point_size=8)
        plotter.show()

    def draw_segments(self, segments: list[np.ndarray]):
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.grid, color='black', style='wireframe')
        for segment in segments:
            random_color = (np.random.uniform(0.0, 1.0, size=(3,)) * 255).astype(np.int32)
            if self.cell_mode:
                plotter.add_mesh(pyvista.PointSet(self.cell_centers[segment, :]), color=random_color, point_size=8)
            else:
                plotter.add_mesh(pyvista.PointSet(self.points[segment, :]), color=random_color, point_size=8)
        plotter.show()


def debug_point_mode():
    source_points = np.array([
        [0.0, 0.25],
        [2.25, 2.5],
        [4.75, 2.5]
    ])
    
    mesh = SeqMeshObj(
        cell_mode=False,
        xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t8/test2.xdmf",
    )
    mesh.create_neighbors_info()
    mesh.create_seq_tag(source_points, np.ones((mesh.num_points,)))
    
    # mesh.draw_point_neighbors()
    # mesh.draw_cell_neighbors()
    mesh.draw_seq_tag()
    
    random_point_idxs = np.random.choice(mesh.num_points, size=int(mesh.num_points * 0.5), replace=False)
    segments = mesh.extract_all_clusters(random_point_idxs)
    mesh.draw_segments(segments)


def debug_cell_mode():
    source_cells = np.array([
        [0.0, 0.25],
        [2.25, 2.5],
        [4.75, 2.5]
    ])
    
    mesh = SeqMeshObj(
        cell_mode=True,
        xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t8/test2.xdmf",
    )
    mesh.create_neighbors_info()
    mesh.create_seq_tag(source_cells, np.ones((mesh.num_cells,)))
    mesh.draw_seq_tag()
    
    random_cell_idxs = np.random.choice(mesh.num_cells, size=int(mesh.num_cells * 0.5), replace=False)
    segments = mesh.extract_all_clusters(random_cell_idxs)
    mesh.draw_segments(segments)    


def debug_3D():
    mesh = SeqMeshObj(
        cell_mode=False,
        xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t8/test.xdmf",
    )
    mesh.create_neighbors_info()
    
    # mesh.draw_point_neighbors()
    mesh.draw_cell_neighbors()

if __name__ == "__main__":
    # debug_point_mode()
    # debug_cell_mode()
    debug_3D()
    
    pass