import os

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


class SequenceMeshObj(MeshObj):
    
    class Cluster(object):
        def __init__(self, cluster_id: int, included_idxs: np.ndarray, cell_mode: bool):
            self.cluster_id = cluster_id
            self.included_idxs = included_idxs
            self.cell_mode = cell_mode
            self.surface_idxs: np.ndarray
            self.inner_idxs: np.ndarray
    
    
    class ConnectivitySet(object):
        def __init__(self, name: str, included_idx: int, from_cluster_id: int, to_cluster_id: int, from_dist: int, to_dist: int):
            self.name = name
            self.included_idxs: list[int] = [included_idx]
            self.from_cluster_id: int = from_cluster_id
            self.to_cluster_id: int = to_cluster_id
            self.from_dist: list[int] = [from_dist]
            self.to_dist: list[int] = [to_dist]
        
        def assemble(self):
            self.included_idxs = np.array(self.included_idxs) # type: ignore
            self.from_dist = np.array(self.from_dist) # type: ignore
            self.to_dist = np.array(self.to_dist) # type: ignore

        def get_conn_idxs(self, relax_factor: int = -1):
            if relax_factor < 0:
                return self.included_idxs
            
            path_dist = self.from_dist + self.to_dist
            min_path_dist = np.min(path_dist)
            threshold_dist = min_path_dist + relax_factor
            return self.included_idxs[path_dist <= threshold_dist]
        
    
    def __init__(
        self,
        xdmf_file: str| None = None,
        mesh_name="model",
        domain: dolfinx.mesh.Mesh | None = None,
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        facet_tags: dolfinx.mesh.MeshTags | None = None,
        comm=MPI.COMM_WORLD,
    ):
        super().__init__(xdmf_file, mesh_name, domain, cell_tags, facet_tags, comm)
        assert MeshUtils.get_cell_type(self.domain) in [
            dolfinx.mesh.CellType.tetrahedron,  # type: ignore
            dolfinx.mesh.CellType.triangle,  # type: ignore
        ], "[Error]: Cell type must be tetrahedron or triangle."
        
        self.points_to_neighbors: dict[int, list[int]] = None  # type: ignore
        self.cells_to_neighbors: dict[int, list[int]] = None  # type: ignore

    def get_points_to_neighbors(self):
        if self.points_to_neighbors is not None:
            return self.points_to_neighbors
        
        self.points_to_neighbors = {}
        for p_idx in range(self.num_points):
            self.points_to_neighbors[p_idx] = []
        
        for _, p_idxs in tqdm(enumerate(self.get_cell_to_points()), desc="Creating points neighbors"):
            for p0, p1 in combinations(p_idxs, 2):
                self.points_to_neighbors[p0].append(p1)
                self.points_to_neighbors[p1].append(p0)
        
        for p_idx, neighbor_idxs in tqdm(self.points_to_neighbors.items(), desc="Removing duplicate neighbors"):
            self.points_to_neighbors[p_idx] = np.unique(neighbor_idxs).tolist()
        
        return self.points_to_neighbors

    def get_cells_to_neighbors(self):
        if self.cells_to_neighbors is not None:
            return self.cells_to_neighbors
        
        self.cells_to_neighbors = {}
        for cell_idx in range(self.num_cells):
            self.cells_to_neighbors[cell_idx] = []
        
        tag_to_cells = {}
        for cell_idx, p_idxs in tqdm(enumerate(self.get_cell_to_points()), desc="Creating points neighbors"):
            if self.tdim == 2:
                for p0, p1 in combinations(p_idxs, 2):
                    edge_min, edge_max = np.sort([p0, p1])
                    edge_tag = f"{edge_min}_{edge_max}"
                    if edge_tag not in tag_to_cells.keys():
                        tag_to_cells[edge_tag] = []
                    tag_to_cells[edge_tag].append(cell_idx)
            else:
                for p0, p1, p2 in combinations(p_idxs, 3):
                    face_min, face_mid, face_max = np.sort([p0, p1, p2])
                    face_tag = f"{face_min}_{face_mid}_{face_max}"
                    if face_tag not in tag_to_cells.keys():
                        tag_to_cells[face_tag] = []
                    tag_to_cells[face_tag].append(cell_idx)
        
        for edge_tag, cell_idxs in tqdm(tag_to_cells.items(), desc="Creating cells neighbors"):
            for c0, c1 in combinations(cell_idxs, 2):
                self.cells_to_neighbors[c0].append(c1)
                self.cells_to_neighbors[c1].append(c0)
        
        for c_idx, neighbor_idxs in tqdm(self.cells_to_neighbors.items(), desc="Removing duplicate neighbors"):
            self.cells_to_neighbors[c_idx] = np.unique(neighbor_idxs).tolist()
        
        return self.cells_to_neighbors

    def generate_seq_tag(self, source_xyz: np.ndarray, reachable_map: np.ndarray, cell_mode: bool):
        """
        source_xyzs:
            if self.cell_mode is True, source_xyzs is cell centers
            if self.cell_mode is False, source_xyzs is points
        reachable_map: [0.0, >0.0], 0.0表示该点不可达, >0.0表示该点可达
        """
        if cell_mode:
            tree = KDTree(self.cell_centers[:, :self.tdim])
            seq_dense_tag = np.zeros((self.num_cells,))
            conn_neighbors = self.get_cells_to_neighbors()
        else:
            tree = KDTree(self.points[:, :self.tdim])
            seq_dense_tag = np.zeros((self.num_points,))
            conn_neighbors = self.get_points_to_neighbors()
        
        source_idx = tree.query(source_xyz.reshape((1, self.tdim)), k=1, return_distance=False)[0][0]
        seq_dense_tag[source_idx] = 1
        source_idxs: np.ndarray = np.array([source_idx])
        layer_tag = 2
                
        while True:
            next_idxs = []
            for idx in source_idxs:
                next_idxs.append(conn_neighbors[idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            
            next_idxs = np.unique(next_idxs)
            next_idxs = next_idxs[seq_dense_tag[next_idxs] == 0.0]
            next_idxs = next_idxs[reachable_map[next_idxs] > 0.0]
            
            if next_idxs.shape[0] == 0:
                assert np.all(seq_dense_tag[reachable_map > 0.0] > 0), "[Error]: Some reachable points are not reachable."
                break
            
            seq_dense_tag[next_idxs] = layer_tag
            source_idxs = next_idxs
            layer_tag += 1
        
        return seq_dense_tag
    
    def extract_cluster_from_unseen_idx(self, unseen_idx: int, seen_map: np.ndarray, cell_mode: bool):
        if cell_mode:
            conn_neighbors = self.get_cells_to_neighbors()
        else:
            conn_neighbors = self.get_points_to_neighbors()
        
        seen_map_copy = seen_map.copy()
        seen_map_copy[unseen_idx] = 1.0
        
        cluster_idxs = [unseen_idx]
        source_idxs = np.array([unseen_idx])
        while True:
            next_idxs = []
            for idx in source_idxs:
                next_idxs.append(conn_neighbors[idx])
            next_idxs = np.concatenate(next_idxs, axis=-1)
            next_idxs = np.unique(next_idxs)
            next_idxs = next_idxs[seen_map_copy[next_idxs] == 0.0]
            if next_idxs.shape[0] == 0:
                break
            
            seen_map_copy[next_idxs] = 1.0
            source_idxs = next_idxs
            cluster_idxs.extend(next_idxs.tolist())
        
        return np.array(cluster_idxs)
    
    def extract_clusters(self, reachable_map: np.ndarray, cell_mode: bool, contain_idxs: np.ndarray|None = None):
        """reachable_map: [0.0, >0.0], 0.0表示该点不可达, >0.0表示该点可达"""
        if cell_mode:
            seen_map = np.zeros((self.num_cells,))
        else:
            seen_map = np.zeros((self.num_points,))
        seen_map[reachable_map == 0.0] = 1.0  # non-reachable cells set to 1.0, mean has seen them
        
        clusters = []
        while True:
            if contain_idxs is None:
                unseen_idx = np.random.choice(np.where(seen_map == 0.0)[0], size=1)[0]
            else:
                unseen_idx = np.random.choice(contain_idxs, size=1)[0]
            
            cluster_idxs = self.extract_cluster_from_unseen_idx(unseen_idx, seen_map, cell_mode)
            seen_map[cluster_idxs] = 1.0
            
            clusters.append(SequenceMeshObj.Cluster(cluster_id=len(clusters), included_idxs=cluster_idxs, cell_mode=cell_mode))
            if contain_idxs is not None:
                contain_idxs = contain_idxs[seen_map[contain_idxs] == 0.0]
                if len(contain_idxs) == 0:
                    break
            
            if np.all(seen_map == 1.0):
                break
        
        return clusters

    def extract_cluster_feature(self, cluster: 'SequenceMeshObj.Cluster'):
        if cluster.cell_mode:
            conn_neighbors = self.get_cells_to_neighbors()
        else:
            conn_neighbors = self.get_points_to_neighbors()        
        inside_map = np.zeros((self.num_cells,))
        inside_map[cluster.included_idxs] = 1.0
        
        surface_idxs, inner_idxs = [], []
        for ic_idx in cluster.included_idxs:
            neighbor_idxs = conn_neighbors[ic_idx]
            if np.sum(inside_map[neighbor_idxs]) != len(neighbor_idxs):
                surface_idxs.append(ic_idx)
            else:
                inner_idxs.append(ic_idx)
        cluster.surface_idxs = np.array(surface_idxs)
        cluster.inner_idxs = np.array(inner_idxs)

    def compute_distance_to_two_closest_clusters(self, clusters: list['SequenceMeshObj.Cluster'], reachable_map: np.ndarray):
        cell_mode = clusters[0].cell_mode
        if cell_mode:
            seen_map = np.zeros((self.num_cells,))
            conn_neighbors = self.get_cells_to_neighbors()
        else:
            seen_map = np.zeros((self.num_points,))
            conn_neighbors = self.get_points_to_neighbors()
        seen_map[reachable_map == 0.0] = 1.0
        for cluster in clusters:
            seen_map[cluster.included_idxs] = 1.0
        
        cluster_info_dict = {}
        for cluster in clusters:
            cluster_info_dict[cluster.cluster_id] = {
                'seen_map': seen_map.copy(), 'source_idxs': cluster.surface_idxs,
            }
        del seen_map
        
        cell_dist_map = {}
        layer_tag = 1
        while True:
            is_finished = True
            for cluster_id, cluster_info in cluster_info_dict.items():
                source_idxs = cluster_info['source_idxs']
                if len(source_idxs) == 0:
                    continue
                
                seen_map = cluster_info['seen_map']
                
                next_idxs = []
                for idx in source_idxs:
                    next_idxs.append(conn_neighbors[idx])
                next_idxs = np.concatenate(next_idxs, axis=-1)
                
                next_idxs = np.unique(next_idxs)
                next_idxs = next_idxs[seen_map[next_idxs] == 0.0]
                seen_map[next_idxs] = 1.0
                
                valid_idxs = []
                for idx in next_idxs:
                    if idx not in cell_dist_map.keys():
                        cell_dist_map[idx] = {
                            'count': 1,
                            'related_dists': [layer_tag],
                            'related_cluster_ids': [cluster_id]
                        }
                        valid_idxs.append(idx)
                    else:
                        if cell_dist_map[idx]['count'] == 1:
                            cell_dist_map[idx]['count'] += 1
                            cell_dist_map[idx]['related_dists'].append(layer_tag)
                            cell_dist_map[idx]['related_cluster_ids'].append(cluster_id)
                            valid_idxs.append(idx)
                
                cluster_info.update({
                    'seen_map': seen_map,
                    'source_idxs': valid_idxs,
                })
                is_finished = is_finished & (len(valid_idxs) == 0)
            
            layer_tag += 1
            if is_finished:
                break

        connectivity_sets_dict: dict[str, SequenceMeshObj.ConnectivitySet] = {}
        for idx, info in cell_dist_map.items():
            if info['count'] == 1:
                continue
            
            sorted_indices = np.argsort(info['related_cluster_ids'])
            sorted_cluster_ids = np.array(info['related_cluster_ids'])[sorted_indices]
            sorted_dists = np.array(info['related_dists'])[sorted_indices]
            name = '_'.join([f"c{cluster_id}" for cluster_id in sorted_cluster_ids])
                       
            if name not in connectivity_sets_dict.keys():
                connectivity_sets_dict[name] = SequenceMeshObj.ConnectivitySet(
                    name=name, 
                    included_idx=idx, 
                    from_cluster_id=sorted_cluster_ids[0], 
                    to_cluster_id=sorted_cluster_ids[1], 
                    from_dist=sorted_dists[0], 
                    to_dist=sorted_dists[1])
            else:
                connectivity_sets_dict[name].included_idxs.append(idx)
                connectivity_sets_dict[name].from_dist.append(sorted_dists[0])
                connectivity_sets_dict[name].to_dist.append(sorted_dists[1])
        
        for _, connectivity_set in connectivity_sets_dict.items():
            connectivity_set.assemble()
        
        return connectivity_sets_dict
            
    def draw_seq_tag(self, seq_dense_tag: np.ndarray, cell_mode: bool, threshold: float = 0.0, plotter: pyvista.Plotter|None = None):
        with_plot = False
        if plotter is None:
            plotter = pyvista.Plotter()
            with_plot = True
        
        grid = self.grid.copy()
        if threshold == 0.0:
            centers_mesh: pyvista.PointSet = grid.cell_centers() if cell_mode else pyvista.PointSet(grid.points)  # type: ignore
            centers_mesh.point_data["seq_dense_tag"] = seq_dense_tag
            centers_mesh.warp_by_scalar("seq_dense_tag")
            plotter.add_mesh(centers_mesh, style='surface')
        else:
            if cell_mode:
                valid_pcd = grid.cell_centers().points[seq_dense_tag >= threshold]
            else:
                valid_pcd = grid.points[seq_dense_tag >= threshold]
            plotter.add_mesh(pyvista.PointSet(valid_pcd), color='red')
        
        if with_plot:
            plotter.add_mesh(grid, style='wireframe')
            plotter.show()

    def draw_clusters(self, clusters: list['SequenceMeshObj.Cluster'], plotter: pyvista.Plotter|None = None):
        with_plot = False
        if plotter is None:
            plotter = pyvista.Plotter()
            with_plot = True
        
        cell_mode = clusters[0].cell_mode
        
        grid = self.grid.copy()
        for cluster in clusters:
            random_color = (np.random.uniform(0.0, 1.0, size=(3,)) * 255).astype(np.int32)
            pcd = grid.cell_centers().points[cluster.included_idxs, :] if cell_mode else grid.points[cluster.included_idxs, :]
            plotter.add_mesh(pyvista.PointSet(pcd), color=random_color, point_size=8)
        
        if with_plot:
            plotter.add_mesh(grid, color='black', style='wireframe')
            plotter.show()

    def draw_cluster_feature(self, clusters: list['SequenceMeshObj.Cluster'], plotter: pyvista.Plotter|None = None):
        with_plot = False
        if plotter is None:
            plotter = pyvista.Plotter()
            with_plot = True
        
        cell_mode = clusters[0].cell_mode
        
        grid = self.grid.copy()
        for cluster in clusters:
            pcd = grid.cell_centers().points if cell_mode else grid.points
            if len(cluster.surface_idxs) > 0:
                plotter.add_mesh(pyvista.PointSet(pcd[cluster.surface_idxs, :]), color='red')
            if len(cluster.inner_idxs) > 0:
                plotter.add_mesh(pyvista.PointSet(pcd[cluster.inner_idxs, :]), color='blue')
        
        if with_plot:
            plotter.add_mesh(grid, color='black', style='wireframe')
            plotter.show()


def debug_connectivity_layer():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        msh_file=os.path.join(base_dir, 'model.msh'), dim=2
    )
    mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    source_xyz = np.array([0.5, 0.0])
    source_path_1 = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.2, 1.0],
    ])
    source_path_2 = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.8, 1.0],
    ])
    source_path_3 = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.8, 0.8],
        [1.0, 0.8],
    ])
    
    path_pcd = []
    for path in [source_path_1, source_path_2, source_path_3]:
        for i in range(path.shape[0] - 1):
            step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.01)
            path_pcd.append(np.linspace(path[i], path[i + 1], step))
    path_pcd = np.concatenate(path_pcd, axis=0)
    
    # if path_pcd.shape[1] == 2:
    #     path_pcd = np.concatenate((path_pcd, np.zeros((path_pcd.shape[0], 1))), axis=1)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    # plotter.add_mesh(pyvista.PointSet(path_pcd), color='red')
    # plotter.show()
    
    # ------ compute cell reachable map ------
    cell_tree = KDTree(mesh.cell_centers[:, :mesh.tdim])
    nearest_idxs = cell_tree.query_radius(path_pcd, r=0.05, return_distance=False)
    nearest_idxs = nearest_idxs.flatten()
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    cell_reachable_map = np.zeros((mesh.num_cells,))
    cell_reachable_map[nearest_idxs] = 1.0
    seq_dense_tag = mesh.generate_seq_tag(source_xyz, cell_reachable_map, cell_mode=True)
    mesh.draw_seq_tag(seq_dense_tag, cell_mode=True, threshold=0.0)

    # ------ compute point reachable map ------
    point_tree = KDTree(mesh.points[:, :mesh.tdim])
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.05, return_distance=False)
    nearest_idxs = nearest_idxs.flatten()
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    point_reachable_map = np.zeros((mesh.num_points,))
    point_reachable_map[nearest_idxs] = 1.0
    seq_dense_tag = mesh.generate_seq_tag(source_xyz, point_reachable_map, cell_mode=False)
    mesh.draw_seq_tag(seq_dense_tag, cell_mode=False, threshold=1.0)


def debug_extract_clusters():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        msh_file=os.path.join(base_dir, 'model.msh'), dim=2
    )
    mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    source_path_1 = np.array([[0.5, 0.0], [0.5, 0.5], [0.2, 1.0]])
    source_path_2 = np.array([[0.5, 0.0], [0.5, 0.5], [0.8, 1.0]])
    source_path_3 = np.array([[0.5, 0.0], [0.5, 0.5], [0.7, 0.8], [0.5, 1.0]])
    source_path_4 = np.array([[0.5, 0.0], [0.5, 0.5], [0.3, 0.8], [0.5, 1.0]])
    
    path_pcd = []
    for path in [source_path_1, source_path_2, source_path_3, source_path_4]:
        for i in range(path.shape[0] - 1):
            step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.01)
            path_pcd.append(np.linspace(path[i], path[i + 1], step))
    path_pcd = np.concatenate(path_pcd, axis=0)
    
    block_pcd = np.array([[0.5, 0.2], [0.4, 0.9], [0.6, 0.9], [0.63, 0.7]])
    
    """
    # ------ compute cell reachable map ------
    cell_tree = KDTree(mesh.cell_centers[:, :mesh.tdim])
    nearest_idxs = cell_tree.query_radius(path_pcd, r=0.03, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    cell_reachable_map = np.zeros((mesh.num_cells,))
    cell_reachable_map[nearest_idxs] = 1.0    
    
    block_idxs = cell_tree.query_radius(block_pcd, r=0.05, return_distance=False)
    block_idxs = np.concatenate(block_idxs, axis=-1)
    block_idxs = np.unique(block_idxs)
    cell_reachable_map[block_idxs] = 0.0
    
    # # ------ visualize thresholds map ------
    # plotter = pyvista.Plotter()    
    # cell_centers_mesh: pyvista.PointSet = mesh.grid.cell_centers()  # type: ignore
    # cell_centers_mesh.point_data["cell_reachable_map"] = cell_reachable_map
    # cell_centers_mesh.warp_by_scalar("cell_reachable_map")
    # plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    # plotter.add_mesh(cell_centers_mesh, style='surface')
    # plotter.show()
    # # ------ visualize thresholds map ------
    
    clusters = mesh.extract_clusters(cell_reachable_map, cell_mode=True)
    mesh.draw_clusters(clusters)
    """
    
    # ------ compute distance to two closest clusters ------
    point_tree = KDTree(mesh.points[:, :mesh.tdim])
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.03, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    point_reachable_map = np.zeros((mesh.num_points,))
    point_reachable_map[nearest_idxs] = 1.0    
    
    block_idxs = point_tree.query_radius(block_pcd, r=0.05, return_distance=False)
    block_idxs = np.concatenate(block_idxs, axis=-1)
    block_idxs = np.unique(block_idxs)
    point_reachable_map[block_idxs] = 0.0
    
    # # ------ visualize thresholds map ------
    # plotter = pyvista.Plotter()    
    # point_mesh: pyvista.PointSet = pyvista.PointSet(mesh.points)  # type: ignore
    # point_mesh.point_data["point_reachable_map"] = point_reachable_map
    # point_mesh.warp_by_scalar("point_reachable_map")
    # plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    # plotter.add_mesh(point_mesh, style='surface')
    # plotter.show()
    # # ------ visualize thresholds map ------
    
    clusters = mesh.extract_clusters(point_reachable_map, cell_mode=False)
    mesh.draw_clusters(clusters)


def debug_extract_cell_connectivity():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        msh_file=os.path.join(base_dir, 'model.msh'), dim=2
    )
    mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    source_path_1 = np.array([[0.5, 0.0], [0.5, 0.5], [0.2, 1.0]])
    source_path_2 = np.array([[0.5, 0.0], [0.5, 0.5], [0.8, 1.0]])
    source_path_3 = np.array([[0.5, 0.0], [0.5, 0.5], [0.7, 0.8], [0.5, 1.0]])
    source_path_4 = np.array([[0.5, 0.0], [0.5, 0.5], [0.3, 0.8], [0.5, 1.0]])
    
    path_pcd = []
    for path in [source_path_1, source_path_2, source_path_3, source_path_4]:
        for i in range(path.shape[0] - 1):
            step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.01)
            path_pcd.append(np.linspace(path[i], path[i + 1], step))
    path_pcd = np.concatenate(path_pcd, axis=0)
    
    block_pcd = np.array([[0.5, 0.2], [0.4, 0.9], [0.6, 0.9], [0.63, 0.7]])
    
    # ------ compute inner thresholds_map ------
    cell_tree = KDTree(mesh.cell_centers[:, :mesh.tdim])
    nearest_idxs = cell_tree.query_radius(path_pcd, r=0.03, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    inner_reachable_map = np.zeros((mesh.num_cells,))
    inner_reachable_map[nearest_idxs] = 1.0
    
    block_idxs = cell_tree.query_radius(block_pcd, r=0.05, return_distance=False)
    block_idxs = np.concatenate(block_idxs, axis=-1)
    block_idxs = np.unique(block_idxs)
    inner_reachable_map[block_idxs] = 0.0
    
    clusters = mesh.extract_clusters(inner_reachable_map, cell_mode=True)
    for cluster in clusters:
        mesh.extract_cluster_feature(cluster)
    
    # ------ compute outer thresholds_map ------
    nearest_idxs = cell_tree.query_radius(path_pcd, r=0.07, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    outer_reachable_map = np.zeros((mesh.num_cells,))
    outer_reachable_map[nearest_idxs] = 1.0
    
    # ------ compute connectivity map ------
    conn_set_dict = mesh.compute_distance_to_two_closest_clusters(clusters, outer_reachable_map)
    
    # ------ visualize thresholds map ------
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    cell_centers_mesh: pyvista.PointSet = mesh.grid.cell_centers()  # type: ignore
    cell_centers_mesh.point_data["outer_reachable_map"] = outer_reachable_map
    cell_centers_mesh.warp_by_scalar("outer_reachable_map")
    plotter.add_mesh(cell_centers_mesh, style='surface')
    mesh.draw_cluster_feature(clusters, plotter=plotter)
    
    for _, conn_set in conn_set_dict.items():
        random_color = (np.random.uniform(0.0, 1.0, size=(3,)) * 255).astype(np.int32)
        point_mesh = pyvista.PointSet(mesh.grid.cell_centers().points[conn_set.get_conn_idxs(relax_factor=2), :])
        plotter.add_mesh(point_mesh, color=random_color, point_size=8)
    
    plotter.show()


def debug_extract_point_connectivity():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        msh_file=os.path.join(base_dir, 'model.msh'), dim=2
    )
    mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    source_path_1 = np.array([[0.5, 0.0], [0.5, 0.5], [0.2, 1.0]])
    source_path_2 = np.array([[0.5, 0.0], [0.5, 0.5], [0.8, 1.0]])
    source_path_3 = np.array([[0.5, 0.0], [0.5, 0.5], [0.7, 0.8], [0.5, 1.0]])
    source_path_4 = np.array([[0.5, 0.0], [0.5, 0.5], [0.3, 0.8], [0.5, 1.0]])
    
    path_pcd = []
    for path in [source_path_1, source_path_2, source_path_3, source_path_4]:
        for i in range(path.shape[0] - 1):
            step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.01)
            path_pcd.append(np.linspace(path[i], path[i + 1], step))
    path_pcd = np.concatenate(path_pcd, axis=0)
    block_pcd = np.array([[0.5, 0.2], [0.4, 0.9], [0.6, 0.9], [0.63, 0.7]])
    
    # ------ compute inner thresholds_map ------
    point_tree = KDTree(mesh.points[:, :mesh.tdim])
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.03, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    inner_reachable_map = np.zeros((mesh.num_points,))
    inner_reachable_map[nearest_idxs] = 1.0
    
    block_idxs = point_tree.query_radius(block_pcd, r=0.05, return_distance=False)
    block_idxs = np.concatenate(block_idxs, axis=-1)
    block_idxs = np.unique(block_idxs)
    inner_reachable_map[block_idxs] = 0.0
    
    clusters = mesh.extract_clusters(inner_reachable_map, cell_mode=False)
    for cluster in clusters:
        mesh.extract_cluster_feature(cluster)
    
    # ------ compute outer thresholds_map ------
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.07, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    outer_reachable_map = np.zeros((mesh.num_points,))
    outer_reachable_map[nearest_idxs] = 1.0
    
    # ------ compute connectivity map ------
    conn_set_dict = mesh.compute_distance_to_two_closest_clusters(clusters, outer_reachable_map)
    
    # ------ visualize thresholds map ------
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    point_mesh: pyvista.PointSet = pyvista.PointSet(mesh.points)  # type: ignore
    point_mesh.point_data["outer_reachable_map"] = outer_reachable_map
    point_mesh.warp_by_scalar("outer_reachable_map")
    plotter.add_mesh(point_mesh, style='surface')
    mesh.draw_cluster_feature(clusters, plotter=plotter)
    
    for _, conn_set in conn_set_dict.items():
        random_color = (np.random.uniform(0.0, 1.0, size=(3,)) * 255).astype(np.int32)
        point_mesh = pyvista.PointSet(mesh.points[conn_set.get_conn_idxs(relax_factor=0), :])
        plotter.add_mesh(point_mesh, color=random_color, point_size=8)
    
    plotter.show()


def debug_extract_contain_point_connectivity():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'
    domain, cell_tags, facet_tags = MeshUtils.read_msh(
        msh_file=os.path.join(base_dir, 'model.msh'), dim=2
    )
    mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    source_path_1 = np.array([[0.5, 0.0], [0.5, 0.5], [0.2, 1.0]])
    source_path_2 = np.array([[0.5, 0.0], [0.5, 0.5], [0.8, 1.0]])
    source_path_3 = np.array([[0.5, 0.0], [0.5, 0.5], [0.7, 0.8], [0.5, 1.0]])
    source_path_4 = np.array([[0.5, 0.0], [0.5, 0.5], [0.3, 0.8], [0.5, 1.0]])
    
    path_pcd = []
    for path in [source_path_1, source_path_2, source_path_3, source_path_4]:
        for i in range(path.shape[0] - 1):
            step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.01)
            path_pcd.append(np.linspace(path[i], path[i + 1], step))
    path_pcd = np.concatenate(path_pcd, axis=0)
    block_pcd = np.array([[0.5, 0.2], [0.4, 0.9], [0.6, 0.9], [0.63, 0.7]])
    
    # ------ compute inner thresholds_map ------
    point_tree = KDTree(mesh.points[:, :mesh.tdim])
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.03, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    inner_reachable_map = np.zeros((mesh.num_points,))
    inner_reachable_map[nearest_idxs] = 1.0
    
    block_idxs = point_tree.query_radius(block_pcd, r=0.05, return_distance=False)
    block_idxs = np.concatenate(block_idxs, axis=-1)
    block_idxs = np.unique(block_idxs)
    inner_reachable_map[block_idxs] = 0.0
    
    non_related_center = np.array([[0.85, 0.15]])
    non_related_radius = 0.05
    non_related_bool = np.linalg.norm(mesh.points[:, :mesh.tdim] - non_related_center, axis=1) < non_related_radius
    inner_reachable_map[non_related_bool] = 1.0
    
    # ------ compute outer thresholds_map ------
    nearest_idxs = point_tree.query_radius(path_pcd, r=0.07, return_distance=False)
    nearest_idxs = np.concatenate(nearest_idxs, axis=-1)
    nearest_idxs = np.unique(nearest_idxs)
    outer_reachable_map = np.zeros((mesh.num_points,))
    outer_reachable_map[nearest_idxs] = 1.0
    
    non_related_center = np.array([[0.85, 0.15]])
    non_related_radius = 0.1
    non_related_bool = np.linalg.norm(mesh.points[:, :mesh.tdim] - non_related_center, axis=1) < non_related_radius
    outer_reachable_map[non_related_bool] = 1.0
    
    # ------ contain idxs ------
    contain_xy = np.array([
        [0.5, 0.0], 
        [0.2, 1.0], 
        [0.8, 1.0],
        [0.5, 1.0]
    ])
    contain_idxs = point_tree.query(contain_xy, k=1, return_distance=False)
    contain_idxs = np.concatenate(contain_idxs, axis=-1)
    contain_idxs = np.unique(contain_idxs)
    
    # ------ compute connectivity map ------
    clusters = mesh.extract_clusters(inner_reachable_map, cell_mode=False, contain_idxs=contain_idxs)
    for cluster in clusters:
        mesh.extract_cluster_feature(cluster)
    conn_set_dict = mesh.compute_distance_to_two_closest_clusters(clusters, outer_reachable_map)
    
    # ------ visualize thresholds map ------
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe', color='black')
    
    # outer_point_mesh: pyvista.PointSet = pyvista.PointSet(mesh.points)  # type: ignore
    # outer_point_mesh.point_data["outer_reachable_map"] = outer_reachable_map
    # outer_point_mesh.warp_by_scalar("outer_reachable_map")
    # plotter.add_mesh(outer_point_mesh, style='surface')
    
    # inner_point_mesh: pyvista.PointSet = pyvista.PointSet(mesh.points)  # type: ignore
    # inner_point_mesh.point_data["inner_reachable_map"] = inner_reachable_map
    # inner_point_mesh.warp_by_scalar("inner_reachable_map")
    # plotter.add_mesh(inner_point_mesh, style='surface')
    
    mesh.draw_cluster_feature(clusters, plotter=plotter)
    
    for _, conn_set in conn_set_dict.items():
        random_color = (np.random.uniform(0.0, 1.0, size=(3,)) * 255).astype(np.int32)
        point_mesh = pyvista.PointSet(mesh.points[conn_set.get_conn_idxs(relax_factor=0), :])
        plotter.add_mesh(point_mesh, color=random_color, point_size=8)
    
    plotter.show()


if __name__ == "__main__":
    # debug_connectivity_layer()
    # debug_extract_clusters()
    # debug_extract_cell_connectivity()
    # debug_extract_point_connectivity()
    debug_extract_contain_point_connectivity()
    
    pass
        
