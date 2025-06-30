import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List
import pyvista
from scipy.spatial import Voronoi, voronoi_plot_2d

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils


class ReMeshObj(object):
    def __init__(
            self,
            grid: pyvista.UnstructuredGrid, dim: int,
            bc_dofs: np.ndarray = None, internal_dofs: np.ndarray = None
    ):
        self.grid = grid
        self.dim = dim
        self.num_points = self.grid.n_points

        self.bc_dofs = bc_dofs
        self.internal_dofs = internal_dofs

        self.internal_line_conn: Dict[int, np.ndarray] = {}
        self.points_to_cell: Dict[int, np.ndarray] = {}
        self.cell_conn: np.ndarray = None

        self.update_conn()

    def compute_cell_volume(self, coord: np.ndarray, with_check=False):
        is_valid = True
        if self.dim == 2:
            points_a = coord[self.cell_conn[:, 0], :self.dim]
            points_b = coord[self.cell_conn[:, 1], :self.dim]
            points_c = coord[self.cell_conn[:, 2], :self.dim]
            areas = 0.5 * np.abs(
                points_a[:, 0] * (points_b[:, 1] - points_c[:, 1])
                + points_b[:, 0] * (points_c[:, 1] - points_a[:, 1])
                + points_c[:, 0] * (points_a[:, 1] - points_b[:, 1])
            )
            if with_check:
                is_valid = np.all(areas > 0.0)
            return is_valid, areas
        else:
            points_a = coord[self.cell_conn[:, 0], :self.dim]
            points_b = coord[self.cell_conn[:, 1], :self.dim]
            points_c = coord[self.cell_conn[:, 2], :self.dim]
            points_d = coord[self.cell_conn[:, 3], :self.dim]
            vec_ab = points_b - points_a
            vec_ac = points_c - points_a
            vec_ad = points_d - points_a
            volume = 1. / 6. * np.sum(vec_ab * np.cross(vec_ac, vec_ad, axisa=1), axis=1)
            if with_check:
                is_valid = np.all(volume > 0.0)
            return is_valid, volume

    def update_conn(self):
        self.cell_conn = MeshUtils.cell_connectivity(self.grid, dim=self.dim)

        line_conn = self.grid.extract_all_edges().lines.reshape((-1, 3))[:, 1:3]
        graph = nx.Graph()
        graph.add_edges_from(line_conn)

        internal_line_dict: Dict[int, List[List[int]]] = {}
        for p_idx in self.internal_dofs:
            neg_idxs = list(graph.neighbors(p_idx))
            neg_len = len(neg_idxs)
            if neg_len not in internal_line_dict.keys():
                internal_line_dict[neg_len] = []
            internal_line_dict[neg_len].append([p_idx] + neg_idxs)
        for neg_len, combin_pack in internal_line_dict.items():
            self.internal_line_conn[neg_len] = np.array(combin_pack)

    def laplacian_smoothing(self, max_iteration=1000, tol=1e-2, with_debug=False):
        coord_cur = self.grid.points.copy()[:, :self.dim]
        iteration, is_valid, info = -1, False, '[Error]: laplacian smoothing fail.'
        while True:
            p_idxs, coord_new = [], []
            for neg_len, line_pack in self.internal_line_conn.items():
                new_pos = np.mean(coord_cur[line_pack[:, 1:]], axis=1)
                p_idxs.append(line_pack[:, 0])
                coord_new.append(new_pos)

            p_idxs = np.concatenate(p_idxs, axis=-1)
            coord_new = np.concatenate(coord_new, axis=0)

            # # ----------------------
            # vis = VisUtils()
            # grid_tmp_1 = self.grid.copy()
            # grid_tmp_1.points[:, :self.dim] = coord_cur
            # vis.plot(grid_tmp_1, style='wireframe')
            # grid_tmp_2 = self.grid.copy()
            # grid_tmp_2.points[p_idxs, :self.dim] = coord_new
            # vis.plot(grid_tmp_2, style='wireframe', color=(0., 1., 0.))
            # vis.show()
            # # -----------------------

            error = np.max(np.linalg.norm(coord_cur[p_idxs, :] - coord_new, ord=2, axis=1))
            coord_cur[p_idxs, :] = coord_new

            iteration += 1
            if iteration > max_iteration:
                info = f'[Info]: laplacian smoothing reach max iteration, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

            if error <= tol:
                info = f'[Info]: laplacian smoothing [Info]: converge, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

        if with_debug:
            print(info)

        return {'is_valid': is_valid, 'res_pcd': coord_cur, 'info': info}

    def laplacian_balance_smoothing(self, max_iteration=1000, tol=1e-2, with_debug=False):
        coord_cur = self.grid.points.copy()[:, :self.dim]
        iteration, is_valid, info = -1, False, '[Error]: laplacian smoothing fail.'
        while True:
            is_valid, vol = self.compute_cell_volume(coord_cur, with_check=True)
            if not is_valid:
                info = f'[Info]: laplacian smoothing volume is zero.'
                is_valid = False
                break

            if self.dim == 2:
                cell_df = pd.DataFrame(data={
                    'p_idx0': self.cell_conn[:, 0],
                    'p_idx1': self.cell_conn[:, 1],
                    'p_idx2': self.cell_conn[:, 2],
                    'vol': vol
                })

                # cell_df.sort_values(by=['vol'], inplace=True, ascending=True)
                # weights = np.full(shape=(self.num_points, 3), fill_value=np.inf, dtype=float)
                # for col, p_idxs_name in enumerate(['p_idx0', 'p_idx1', 'p_idx2']):
                #     idxs_vol_df = cell_df[[p_idxs_name, 'vol']].drop_duplicates(subset=[p_idxs_name], keep='first')
                #     weights[idxs_vol_df[p_idxs_name].values, col] = idxs_vol_df['vol'].values
                # weights = np.min(weights, axis=1)

                weights = np.zeros(shape=(self.num_points,))
                weight_count = np.zeros(shape=(self.num_points,))
                for col, p_idxs_name in enumerate(['p_idx0', 'p_idx1', 'p_idx2']):
                    sub_df: pd.Series = cell_df.groupby(by=[p_idxs_name])['vol'].aggregate(func=['sum', 'count'])
                    weights[np.array(sub_df.index)] += sub_df['sum'].values
                    weight_count[np.array(sub_df.index)] += sub_df['count'].values
                weights = weights / weight_count

            else:
                cell_df = pd.DataFrame(data={
                    'p_idx0': self.cell_conn[:, 0],
                    'p_idx1': self.cell_conn[:, 1],
                    'p_idx2': self.cell_conn[:, 2],
                    'p_idx3': self.cell_conn[:, 3],
                    'vol': vol
                })

                # weights = np.full(shape=(self.num_points, 4), fill_value=np.inf, dtype=float)
                # for col, p_idxs_name in enumerate(['p_idx0', 'p_idx1', 'p_idx2', 'p_idx3']):
                #     idxs_vol_df = cell_df[[p_idxs_name, 'vol']].drop_duplicates(subset=[p_idxs_name], keep='first')
                #     weights[idxs_vol_df[p_idxs_name].values, col] = idxs_vol_df['vol'].values
                # weights = np.min(weights, axis=1)

                weights = np.zeros(shape=(self.num_points,))
                weight_count = np.zeros(shape=(self.num_points,))
                for col, p_idxs_name in enumerate(['p_idx0', 'p_idx1', 'p_idx2', 'p_idx3']):
                    sub_df: pd.Series = cell_df.groupby(by=[p_idxs_name])['vol'].aggregate(func=['sum', 'count'])
                    weights[np.array(sub_df.index)] += sub_df['sum'].values
                    weight_count[np.array(sub_df.index)] += sub_df['count'].values
                weights = weights / weight_count

            p_idxs, coord_new = [], []
            for neg_len, line_pack in self.internal_line_conn.items():
                center_weight = weights[line_pack[:, 1:]]
                center_weight = center_weight / np.sum(center_weight, axis=1, keepdims=True)
                new_pos = np.sum(center_weight[:, :, np.newaxis] * coord_cur[line_pack[:, 1:]], axis=1)
                p_idxs.append(line_pack[:, 0])
                coord_new.append(new_pos)

            p_idxs = np.concatenate(p_idxs, axis=-1)
            coord_new = np.concatenate(coord_new, axis=0)

            error = np.max(np.linalg.norm(coord_cur[p_idxs, :] - coord_new, ord=2, axis=1))
            coord_cur[p_idxs, :] = coord_new

            iteration += 1
            if iteration > max_iteration:
                info = f'[Info]: laplacian smoothing reach max iteration, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

            if error <= tol:
                info = f'[Info]: laplacian smoothing [Info]: converge, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

        if with_debug:
            print(info)

        return {'is_valid': is_valid, 'res_pcd': coord_cur, 'info': info}

    def voronoi_smoothing(self, max_iteration=1000, tol=1e-2, with_debug=False):
        """Very terrible performance"""
        raise NotImplementedError

        coord_cur = self.grid.points.copy()[:, :self.dim]
        iteration, is_valid, info = -1, False, '[Error]: laplacian smoothing fail.'
        while True:
            coord_new = []
            vor_graph = Voronoi(coord_cur)
            for inner_idx in self.internal_dofs:
                region_idx = vor_graph.point_region[inner_idx]
                region_vert_idxs = np.array(vor_graph.regions[region_idx])
                region_vert_idxs = region_vert_idxs[region_vert_idxs >= 0]
                coord_new.append(np.mean(vor_graph.vertices[region_vert_idxs, :], axis=0, keepdims=True))
            coord_new = np.concatenate(coord_new, axis=0)

            # # ----------------------
            # vis = VisUtils()
            # grid_tmp_1 = self.grid.copy()
            # grid_tmp_1.points[:, :self.dim] = coord_cur
            # vis.plot(grid_tmp_1, style='wireframe')
            # grid_tmp_2 = self.grid.copy()
            # grid_tmp_2.points[self.internal_dofs, :self.dim] = coord_new
            # vis.plot(grid_tmp_2, style='wireframe', color=(0., 1., 0.))
            # vis.show()
            # # -----------------------

            error = np.max(np.linalg.norm(coord_cur[self.internal_dofs, :] - coord_new, ord=2, axis=1))
            coord_cur[self.internal_dofs, :] = coord_new

            iteration += 1
            if iteration > max_iteration:
                info = f'[Info]: laplacian smoothing reach max iteration, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

            if error <= tol:
                info = f'[Info]: laplacian smoothing [Info]: converge, laplacian smoothing re-mesh error:{error}.'
                is_valid = True
                break

        if with_debug:
            print(info)

        return {'is_valid': is_valid, 'res_pcd': coord_cur, 'info': info}

    def draw_mesh(self, with_dof=False, vis: VisUtils = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True

        vis.plot(self.grid, style='wireframe')

        if with_dof:
            vis.plot(pyvista.PointSet(self.grid.points[self.bc_dofs, :]), color=(0., 0., 1.))
            vis.plot(pyvista.PointSet(self.grid.points[self.internal_dofs, :]), color=(0., 1., 0.))

        if show_vis:
            vis.show()


def debug():
    import dolfinx
    from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshObj

    mesh = MeshObj(mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/model.msh', dim=2)
    bry_cfg = {
        'inlet': {'marker': 35, 'is_free': False},
        'outlet': {'marker': 36, 'is_free': False},
        'wall': {'marker': 37, 'is_free': True},
    }

    func_space = dolfinx.fem.VectorFunctionSpace(mesh.domain, ("CG", 1))
    bc_dofs = []
    for name, cfg in bry_cfg.items():
        bc_dofs.append(
            MeshUtils.extract_entity_dofs(
                func_space, mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=cfg['marker']
                )
            )
        )

    bc_dofs = np.concatenate(bc_dofs, axis=-1)
    internal_dofs = np.array(list(set(np.arange(0, mesh.num_points, 1)) - set(bc_dofs)))

    grid = pyvista.read(
        '/home/admin123456/Desktop/work/topopt_exps/new_test/example04/cache/result_vtu/res_00154.vtu'
    )
    mesh.re_init_points(grid.points)

    re_mesher = ReMeshObj(grid=mesh.grid, dim=mesh.topology_dim, bc_dofs=bc_dofs, internal_dofs=internal_dofs)

    # res_dict = re_mesher.laplacian_smoothing(with_debug=False)
    res_dict = re_mesher.laplacian_balance_smoothing(with_debug=False)
    _, areas = re_mesher.compute_cell_volume(res_dict['res_pcd'])
    print(pd.Series(areas).describe())

    if res_dict['is_valid']:
        res_pcd = res_dict['res_pcd']
        if re_mesher.dim == 2:
            res_pcd = np.concatenate([res_pcd, np.zeros(shape=(res_pcd.shape[0], 1))], axis=1)
        mesh.re_init_points(res_pcd)
        re_mesher.draw_mesh()
    else:
        print(res_dict['info'])


if __name__ == '__main__':
    debug()
