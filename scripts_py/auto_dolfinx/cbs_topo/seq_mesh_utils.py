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
        facet_tags: dolfinx.mesh.MeshTags | None = None
    ):
        super().__init__(xdmf_file, mesh_name, domain, cell_tags, facet_tags)
        assert MeshUtils.get_cell_type(self.domain) in [
            dolfinx.mesh.CellType.tetrahedron,  # type: ignore
            dolfinx.mesh.CellType.triangle,  # type: ignore
        ], \
            "[Error]: Cell type must be tetrahedron or triangle."

        self.points_neighbors: dict[int, list[int]] = {}
        self.points_to_cells: dict[int, list[int]] = {}
        self.cells_neighbors: dict[int, list[int]] = {}
        self.seq_dense_tag: np.ndarray = np.zeros((self.num_points,))
    
    def create_neighbors_info(self):
        self.points_neighbors.clear()
        self.points_to_cells.clear()
        edge_to_cells = {}
        
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
                edge_min, edge_max = np.sort([p0, p1])
                edge_tag = f"{edge_min}_{edge_max}"
                if edge_tag not in edge_to_cells.keys():
                    edge_to_cells[edge_tag] = []
                edge_to_cells[edge_tag].append(cell_idx)
        
        for p_idx, neighbor_idxs in tqdm(self.points_neighbors.items(), desc="Removing duplicate neighbors"):
            self.points_neighbors[p_idx] = list(set(neighbor_idxs))
        
        self.cells_neighbors.clear()
        for edge_tag, cell_idxs in tqdm(edge_to_cells.items(), desc="Creating cells neighbors"):
            for cell_idx in cell_idxs:
                if cell_idx not in self.cells_neighbors.keys():
                    self.cells_neighbors[cell_idx] = []
            
            for c0, c1 in combinations(cell_idxs, 2):
                self.cells_neighbors[c0].append(c1)
                self.cells_neighbors[c1].append(c0)
        
        for c_idx, neighbor_idxs in tqdm(self.cells_neighbors.items(), desc="Removing duplicate neighbors"):
            self.cells_neighbors[c_idx] = list(set(neighbor_idxs))
    
    def create_seq_tag(self, source_points: np.ndarray, point_thresholds: np.ndarray):
        '''
        point_thresholds: [0.0, >0.0], 0.0表示该点不可达, >0.0表示该点可达
        '''
        tree = KDTree(self.points[:, :self.tdim])
        idxs_list = tree.query(source_points, k=1, return_distance=False)
        idxs_list = np.concatenate(idxs_list, axis=-1)
        
        seq_dense_tag: np.ndarray = np.zeros((self.num_points,))
        seq_dense_tag[idxs_list] = 1
        source_idxs: np.ndarray = idxs_list
        layer_tag = 2
        
        while True:
            next_idxs = []
            for p_idx in source_idxs:
                next_idxs.append(self.points_neighbors[p_idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            
            next_idxs = np.unique(next_idxs)
            next_idxs = next_idxs[seq_dense_tag[next_idxs] == 0.0]
            next_idxs = next_idxs[point_thresholds[next_idxs] > 0.0]
            
            if next_idxs.shape[0] == 0:
                assert np.all(seq_dense_tag[point_thresholds > 0.0] > 0), "[Error]: Some reachable points are not reachable."
                break
            
            seq_dense_tag[next_idxs] = layer_tag
            source_idxs = next_idxs
            layer_tag += 1
        
        self.seq_dense_tag = seq_dense_tag
    
    def extract_cluster(self, point_idx: int, include_point_thresholds: np.ndarray):
        cluster_idxs = [np.array([point_idx])]
        source_idxs = np.array([point_idx])
        
        threshold_np = include_point_thresholds.copy()
        threshold_np[point_idx] = 0.0
        
        while True:
            next_idxs = []
            for p_idx in source_idxs:
                next_idxs.append(self.points_neighbors[p_idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            next_idxs = np.unique(next_idxs)
            
            next_idxs = next_idxs[threshold_np[next_idxs] > 0.0]
            if next_idxs.shape[0] == 0:
                break
            
            threshold_np[next_idxs] = 0.0
            source_idxs = next_idxs
            cluster_idxs.append(next_idxs)
        
        return np.concatenate(cluster_idxs, axis=-1)
    
    def extract_all_clusters(self, source_p_idxs: np.ndarray):
        include_point_thresholds = np.zeros((self.num_points,))
        include_point_thresholds[source_p_idxs] = 1.0
        
        clusters = []
        while True:
            p_idx = np.random.choice(np.where(include_point_thresholds == 1.0)[0], size=1)[0]
            cluster_idxs = self.extract_cluster(p_idx, include_point_thresholds)
            clusters.append(cluster_idxs)
            include_point_thresholds[cluster_idxs] = 0.0
            
            if np.all(include_point_thresholds == 0.0):
                break
        
        return clusters

    def draw_seq_tag(self):
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
            plotter.add_mesh(pyvista.PointSet(self.points[segment, :]), color=random_color, point_size=8)
        plotter.show()


def debug():
    source_points = np.array([
        [0.0, 0.25],
        [2.25, 2.5],
        [4.75, 2.5]
    ])
    
    mesh = SeqMeshObj(
        xdmf_file="/home/admin123456/Desktop/work/gmsh_test/t8/test2.xdmf",
    )
    mesh.create_neighbors_info()
    mesh.create_seq_tag(source_points, np.ones((mesh.num_points,)))
    
    # mesh.draw_point_neighbors()
    # mesh.draw_cell_neighbors()
    # mesh.draw_seq_tag()
    
    random_point_idxs = np.random.choice(mesh.num_points, size=int(mesh.num_points * 0.5), replace=False)
    segments = mesh.extract_all_clusters(random_point_idxs)
    mesh.draw_segments(segments)
    

if __name__ == "__main__":
    debug()