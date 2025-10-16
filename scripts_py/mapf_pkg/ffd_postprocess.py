import os
import numpy as np
import pandas as pd
import pyvista
import open3d as o3d
import gdist
from tqdm import tqdm
import tempfile
from sklearn.neighbors import KDTree
import networkx as nx
from typing import Union, Dict

from scripts_py.mapf_pkg.visual_utils import VisUtils


class QuadMesh(object):
    def __init__(self):
        self.pcd: np.ndarray = None
        self.quad_con: np.ndarray = None
        self.con_tag: np.ndarray = None  # same tag mean opposite point
        self.uv_idxs_cat: np.ndarray = None
        self.c_table: pd.DataFrame = None
        self.dense_num: int = None

    def convert_triangle_mesh_to_quad(self, mesh: Union[pyvista.PolyData, pyvista.UnstructuredGrid], mesh_info: Dict):
        tri_con = mesh_info['tri_con']
        self.pcd = np.concatenate([
            mesh_info['pcd'],
            np.asarray(mesh.cell_centers().points)
        ], axis=0)
        center_shift = mesh_info['pcd'].shape[0]

        quad_con = []
        con_tag = []
        for cell_idx in range(mesh.n_cells):
            cell_tri = tri_con[cell_idx, :]

            near_cell_idxs = np.asarray(mesh.cell_neighbors(cell_idx))
            for near_cell_idx in near_cell_idxs:
                near_cell_tri = tri_con[near_cell_idx, :]
                share_nodes = np.intersect1d(cell_tri, near_cell_tri)
                if len(share_nodes) != 2:
                    continue

                quad_con.append([
                    cell_idx + center_shift, share_nodes[0],
                    near_cell_idx + center_shift, share_nodes[1]
                ])
                con_tag.append([0, 1, 0, 1])

        self.quad_con = np.array(quad_con)
        self.con_tag = np.array(con_tag)

    @staticmethod
    def convert_to_flag_df(aggr_df: pd.DataFrame, tol=1e-6):
        """two more columns: [is_unique, unique_idx]"""
        aggr_df['df_idx'] = np.arange(0, aggr_df.shape[0], 1)
        aggr_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        aggr_xyz = aggr_df[['x', 'y', 'z']].values
        unique_bool = ~(np.sum(np.abs(aggr_xyz[1:, :] - aggr_xyz[:-1, :]), axis=1) < tol)
        unique_bool = np.concatenate([np.array([True]), unique_bool], axis=0)
        aggr_df['is_unique'] = unique_bool.astype(int)
        aggr_df['unique_idx'] = aggr_df['is_unique'].cumsum() - 1
        aggr_df.sort_values(by=['df_idx'], inplace=True)
        aggr_df.drop(columns=['df_idx'], inplace=True)
        return aggr_df

    def generate_dense_control_points(self, dense_num: int):
        quad_uv_df = []
        for con_idx, ((node_0, node_1, node_2, node_3), (tag_0, tag_1, tag_2, tag_3)) in enumerate(zip(
                self.quad_con, self.con_tag
        )):
            assert tag_0 == tag_2
            uv_pcd = np.zeros((dense_num, dense_num, 3))

            # U direction
            u_left_seq = np.linspace(self.pcd[node_0, :], self.pcd[node_1, :], dense_num)
            u_right_seq = np.linspace(self.pcd[node_3, :], self.pcd[node_2, :], dense_num)
            for u_i, (left_point, right_point) in enumerate(zip(u_left_seq, u_right_seq)):
                uv_pcd[u_i, :, :] += np.linspace(left_point, right_point, dense_num)

            # V direction
            v_top_seq = np.linspace(self.pcd[node_0, :], self.pcd[node_3, :], dense_num)
            v_down_seq = np.linspace(self.pcd[node_1, :], self.pcd[node_2, :], dense_num)
            for v_i, (top_point, down_point) in enumerate(zip(v_top_seq, v_down_seq)):
                uv_pcd[:, v_i, :] += np.linspace(top_point, down_point, dense_num)

            uv_pcd = uv_pcd * 0.5

            uv_df = pd.DataFrame(uv_pcd.reshape((-1, 3)), columns=['x', 'y', 'z'])
            uv_df['con_idx'] = con_idx
            quad_uv_df.append(uv_df)
        quad_uv_df = pd.concat(quad_uv_df, axis=0, ignore_index=True)
        quad_uv_df = self.convert_to_flag_df(quad_uv_df)

        c_table = quad_uv_df.copy()
        c_table.drop_duplicates(subset=['unique_idx'], inplace=True, keep='last')
        c_table.sort_values(by=['unique_idx'], inplace=True)
        self.c_table = c_table

        quad_uv_idxs_cat = []
        for (con_idx,), sub_df in quad_uv_df.groupby(by=['con_idx']):
            quad_uv_idxs_cat.append(sub_df['unique_idx'].values.reshape((1, dense_num, dense_num)))
        self.uv_idxs_cat = np.concatenate(quad_uv_idxs_cat, axis=0)

        self.dense_num = dense_num

    def find_region_correspond_pcd(self, pcd: np.ndarray):
        """ 不使用点到面距离是因为实际面可能是曲面"""
        uv_df = []
        for region_idx, uv_idxs in enumerate(self.uv_idxs_cat):
            sub_df = pd.DataFrame(uv_idxs.reshape((-1, 1)), columns=['uv_idx'])
            sub_df['region_idx'] = region_idx
            uv_df.append(sub_df)
        uv_df = pd.concat(uv_df, axis=0, ignore_index=True)

        uv_pcd = self.c_table[['x', 'y', 'z']].values[uv_df['uv_idx'].values, :]
        uv_pcd_tree = KDTree(uv_pcd)

        belong_rows = uv_pcd_tree.query(pcd, k=1, return_distance=False)
        belong_rows = np.array(belong_rows).reshape(-1)
        belong_region_idxs = uv_df['region_idx'].values[belong_rows]
        return belong_region_idxs

    def export_quad_pyvista(self) -> pyvista.UnstructuredGrid:
        return pyvista.UnstructuredGrid(pyvista.PolyData(
            self.pcd, np.concatenate([
                np.full((self.quad_con.shape[0], 1), fill_value=4),
                self.quad_con
            ], axis=1)
        ))

    def save(self):
        # todo non-finish
        raise NotImplementedError

    def restore(self):
        # todo non-finish
        raise NotImplementedError


class FfdSurface(object):
    def __init__(self, stl_file: str):
        self.stl_file = stl_file
        mesh: pyvista.PolyData = pyvista.read(stl_file)
        # mesh = self.simplify_mesh(mesh, target_reduction=0.75)
        self.mesh = pyvista.UnstructuredGrid(mesh)
        self.mesh_info = self.extract_mesh_info(self.mesh)

        self.level_set_mesh: pyvista.UnstructuredGrid = None
        self.voronoi_mesh: pyvista.UnstructuredGrid = None
        self.voronoi_mesh_info: Dict = None

        self.quad_model: QuadMesh = None
        self.dense_num: int = None
        self.mesh_pcd_surface_idxs: np.ndarray = None

    @staticmethod
    def uniform_mesh(mesh_file, uniform_sample_num: int, output_file):
        import pymeshlab  # can not use with pyvista
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_file)
        ms.apply_filter('generate_sampling_poisson_disk', samplenum=uniform_sample_num)
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')
        ms.save_current_mesh(output_file, binary=False)

    @staticmethod
    def simplify_mesh(mesh: pyvista.PolyData, target_reduction):
        return mesh.decimate(target_reduction)

    @staticmethod
    def extract_mesh_info(mesh: Union[pyvista.PolyData, pyvista.UnstructuredGrid]):
        pcd = np.asarray(mesh.points).astype(np.float64)
        pcd_idxs = np.arange(0, pcd.shape[0], 1).astype(np.int32)
        tri_con = mesh.cell_connectivity.reshape((-1, 3)).astype(np.int32)

        edge_con = np.concatenate([
            np.concatenate([tri_con[:, 0:1], tri_con[:, 1:2]], axis=1),
            np.concatenate([tri_con[:, 1:2], tri_con[:, 2:3]], axis=1),
            np.concatenate([tri_con[:, 0:1], tri_con[:, 2:3]], axis=1)
        ], axis=0)
        edge_con = np.sort(edge_con, axis=1)
        edge_df = pd.DataFrame(edge_con, columns=['idx0', 'idx1'])
        edge_df.drop_duplicates(subset=['idx0', 'idx1'], inplace=True)
        edge_con = edge_df.values

        return {'pcd': pcd, 'pcd_idxs': pcd_idxs, 'edge_con': edge_con, 'tri_con': tri_con}

    def generate_expand_level_set_mesh(
            self,
            mesh: Union[pyvista.PolyData, pyvista.UnstructuredGrid],
            mesh_info: Dict,
            xyz_delta: np.ndarray, xyz_num: np.ndarray, expand_threshold,
            tolerance=1e-6
    ) -> pyvista.UnstructuredGrid:
        pcd = mesh_info['pcd']

        pcd_min, pcd_max = np.min(pcd, axis=0), np.max(pcd, axis=0)
        orign_xyz = pcd_min - xyz_delta
        space_xyz = ((pcd_max + xyz_delta) - orign_xyz) / (xyz_num - 1.0)
        grid_mesh = pyvista.ImageData(dimensions=xyz_num, spacing=space_xyz, origin=orign_xyz)
        grid_pcd = np.asarray(grid_mesh.points)

        mesh_surface = mesh.extract_surface()
        res_info = grid_mesh.select_enclosed_points(mesh_surface, inside_out=False, tolerance=tolerance)
        inside_idxs = np.nonzero(res_info['SelectedPoints'])[0]

        aud_idxs = []
        surface_tree = KDTree(pcd)
        idxs_list = surface_tree.query_radius(grid_pcd, r=expand_threshold, return_distance=False)
        for grid_pcd_idx, idxs in zip(range(grid_pcd.shape[0]), idxs_list):
            if len(idxs) > 0:
                aud_idxs.append(grid_pcd_idx)
        aud_idxs = np.array(aud_idxs)

        threshold = np.full(np.prod(xyz_num), fill_value=-1.0)
        threshold[inside_idxs] = 1.0
        threshold[aud_idxs] = 1.0
        expand_level_set_mesh = grid_mesh.contour([0.0], threshold, method='marching_cubes')
        self.level_set_mesh = pyvista.UnstructuredGrid(expand_level_set_mesh)
        return self.level_set_mesh

    def generate_Voronoi_triangle(
            self,
            mesh: Union[pyvista.PolyData, pyvista.UnstructuredGrid],
            mesh_info: Dict,
            cluster_num
    ) -> pyvista.UnstructuredGrid:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file = os.path.join(tmp_dir, 'tmp.stl')
            pyvista.save_meshio(save_file, mesh)
            mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(save_file)
            fake_center = np.asarray(mesh_o3d.sample_points_poisson_disk(cluster_num).points)

        pcd = mesh_info['pcd']
        edge_con = mesh_info['edge_con']
        tri_con = mesh_info['tri_con']
        pcd_idxs = mesh_info['pcd_idxs']

        # ------ generate Voronoi graph
        tree = KDTree(pcd)
        voronoi_center_idxs = tree.query(fake_center, k=1, return_distance=False).reshape(-1)

        dist_matrix = []
        for center_idx in tqdm(voronoi_center_idxs):
            dist = gdist.compute_gdist(
                pcd, tri_con,
                source_indices=np.array([center_idx], dtype=np.int32),
                target_indices=pcd_idxs
            )
            dist_matrix.append(dist[np.newaxis, :])
        dist_matrix = np.concatenate(dist_matrix, axis=0).T
        pcd_category = np.argmin(dist_matrix, axis=1)

        # ------ generate triangle
        cate_link_dict = {}
        edge_cate = np.concatenate([
            pcd_category[edge_con[:, 0]].reshape((-1, 1)),
            pcd_category[edge_con[:, 1]].reshape((-1, 1))
        ], axis=1)
        link_idxs = np.nonzero(edge_cate[:, 0] - edge_cate[:, 1])[0]
        for link_idx in tqdm(link_idxs):
            cate0, cate1 = edge_cate[link_idx]
            tag = (min(cate0, cate1), max(cate0, cate1))
            if tag not in cate_link_dict.keys():
                cate_link_dict[tag] = True
        voronoi_dig_edge_con = np.array(list(cate_link_dict.keys()))

        graph = nx.Graph()
        graph.add_edges_from(voronoi_dig_edge_con)
        voronoi_dig_tri_con = np.array([
            circle for circle in nx.enumerate_all_cliques(graph) if len(circle) == 3
        ])

        voronoi_dig_mesh = pyvista.UnstructuredGrid(pyvista.PolyData(
            pcd[voronoi_center_idxs, :],
            np.concatenate([
                np.full((voronoi_dig_tri_con.shape[0], 1), fill_value=3),
                voronoi_dig_tri_con
            ], axis=1)
        ))

        self.voronoi_mesh = voronoi_dig_mesh
        self.voronoi_mesh_info = self.extract_mesh_info(self.voronoi_mesh)
        return self.voronoi_mesh

    @staticmethod
    def solve_maximum_weight_matching_problem(edges: np.ndarray, weight: np.ndarray):
        """how to combine triangle to quad is exactly a “Maximum Matching Problem” """
        graph = nx.Graph()
        graph.add_weighted_edges_from(np.concatenate(
            [edges, weight.reshape((-1, 1))], axis=1
        ))
        return list(nx.max_weight_matching(graph, maxcardinality=True))

    @staticmethod
    def solve_maximum_matching_problem(edges: np.ndarray):
        """how to combine triangle to quad is exactly a “Maximum Matching Problem” """
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return list(nx.maximal_matching(graph))

    @staticmethod
    def merge_triangle_to_quad(
            mesh: Union[pyvista.PolyData, pyvista.UnstructuredGrid],
            mesh_info: Dict,
    ):
        """ Depreciate it is too hard to success"""
        tri_con = mesh_info['tri_con']

        valid_cell_con = []
        # cell_center = np.asarray(mesh.cell_centers().points)
        for cell_idx in range(mesh.n_cells):
            cell_tri = tri_con[cell_idx, :]
            near_cell_idxs = np.asarray(mesh.cell_neighbors(cell_idx))
            near_tri = tri_con[near_cell_idxs, :]
            share_edge_cell_idxs = near_cell_idxs[np.sum(
                ((near_tri - cell_tri[0]) == 0) +
                ((near_tri - cell_tri[1]) == 0) +
                ((near_tri - cell_tri[2]) == 0),
                axis=1
            ) == 2]

            # ------ only for debug
            # vis = VisUtils()
            # vis.plot(mesh, style='wireframe')
            # vis.plot(VisUtils.create_sphere(cell_center[cell_idx], radius=0.2), color=(1., 0., 0.))
            # vis.plot(VisUtils.create_point_cloud(cell_center[share_edge_cell_idxs, :]), color=(0., 1., 0.))
            # vis.show()
            # ------

            share_edge_cell_idxs = share_edge_cell_idxs.reshape((-1, 1))
            valid_cell_con.append(
                np.concatenate([
                    np.full_like(share_edge_cell_idxs, fill_value=cell_idx),
                    share_edge_cell_idxs
                ], axis=1)
            )

        valid_cell_con = np.concatenate(valid_cell_con, axis=0)
        valid_cell_con = np.sort(valid_cell_con, axis=1)
        valid_cell_con = pd.DataFrame(valid_cell_con, columns=['idx0', 'idx1'])
        valid_cell_con.drop_duplicates(subset=['idx0', 'idx1'], inplace=True)
        valid_cell_con = valid_cell_con.values

        # # ------
        # vis = VisUtils()
        # mesh = pyvista.PolyData(
        #     cell_center, np.concatenate([
        #         np.full((valid_cell_con.shape[0], 1), fill_value=2),
        #         valid_cell_con
        #     ], axis=1)
        # )
        # vis.plot(mesh, style='wireframe')
        # vis.show()
        # # ------

        match_list = np.array(FfdSurface.solve_maximum_matching_problem(valid_cell_con))
        is_quad_valid = len(np.unique(match_list)) == mesh.n_cells

        return is_quad_valid, match_list

    def generate_quad_model(self,dense_num: int):
        self.quad_model = QuadMesh()
        self.quad_model.convert_triangle_mesh_to_quad(self.voronoi_mesh, self.voronoi_mesh_info)
        self.quad_model.generate_dense_control_points(dense_num=dense_num)
        self.dense_num = dense_num

    def find_region_correspond_pcd(self):
        self.mesh_pcd_surface_idxs = self.quad_model.find_region_correspond_pcd(self.mesh_info['pcd'])

    def draw(
            self,
            with_orig_mesh=False, with_level_set_mesh=False, with_voronoi_mesh=False,
            with_quad_mesh=False, with_quad_control_pcd=False,
            with_pcd_correspond_quad=False,
            orig_mesh_style='surface', level_set_mesh_style='wireframe', voronoi_mesh_style='wireframe',
            quad_mesh_style='wireframe',
            vis: VisUtils = None
    ):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        if with_orig_mesh:
            vis.plot(self.mesh, style=orig_mesh_style)
        if with_level_set_mesh:
            vis.plot(self.level_set_mesh, style=level_set_mesh_style)
        if with_voronoi_mesh:
            vis.plot(self.voronoi_mesh, style=voronoi_mesh_style)
        if with_quad_mesh:
            vis.plot(self.quad_model.export_quad_pyvista(), style=quad_mesh_style)
            if with_quad_control_pcd:
                c_xyz = self.quad_model.c_table[['x', 'y', 'z']].values
                for uv_idxs in self.quad_model.uv_idxs_cat:
                    reg_pcd = c_xyz[uv_idxs.reshape((-1)), :]
                    vis.plot(VisUtils.create_point_cloud(reg_pcd), color=np.random.uniform(0.0, 1.0, 3))

        if with_pcd_correspond_quad:
            belong_idxs = np.unique(self.mesh_pcd_surface_idxs)
            c_xyz = self.quad_model.c_table[['x', 'y', 'z']].values
            for belong_idx in belong_idxs:
                belong_pcd = self.mesh_info['pcd'][self.mesh_pcd_surface_idxs == belong_idx]

                belong_quad_idxs = self.quad_model.uv_idxs_cat[belong_idx]
                belong_c_xyz = c_xyz[belong_quad_idxs.reshape(-1), :]
                belong_center = np.mean(belong_c_xyz, axis=0)

                subset_pcd = np.concatenate([belong_center.reshape((1, 3)), belong_pcd], axis=0)
                line_set = np.concatenate([
                    np.full((belong_pcd.shape[0], 1), fill_value=2),
                    np.full((belong_pcd.shape[0], 1), fill_value=0),
                    np.arange(0, belong_pcd.shape[0], 1).reshape((-1, 1)) + 1
                ], axis=1)
                vis.plot(VisUtils.create_line(subset_pcd, line_set), color=np.random.uniform(0.0, 1.0, 3))
                # vis.plot(VisUtils.create_point_cloud(belong_pcd), color=np.random.uniform(0.0, 1.0, 3))

        if show_plot:
            vis.show()
