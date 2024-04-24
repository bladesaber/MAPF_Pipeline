import math
import numpy as np
import pyvista
from typing import Union, Callable, Literal
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.cluster import KMeans

import ufl
import dolfinx

from .sparse_utils import ScipySparseUtils
from .lagrange_method.type_database import ShapeDataBase
from .dolfinx_utils import MeshUtils, AssembleUtils


class PyVistaUtils(object):
    @staticmethod
    def save_vtk(grid: pyvista.DataSet, file: str):
        assert file.endswith('.vtk')
        pyvista.save_meshio(file, grid)

    @staticmethod
    def create_grid(coordinates: np.ndarray, connectivity: np.ndarray[np.ndarray[int]]):
        """
        Cell Type:
        * ``EMPTY_CELL = 0``
        * ``VERTEX = 1``
        * ``POLY_VERTEX = 2``
        * ``LINE = 3``
        * ``POLY_LINE = 4``
        * ``TRIANGLE = 5``
        * ``TRIANGLE_STRIP = 6``
        * ``POLYGON = 7``
        * ``PIXEL = 8``
        * ``QUAD = 9``
        * ``TETRA = 10``
        * ``VOXEL = 11``
        * ``HEXAHEDRON = 12``
        * ``WEDGE = 13``
        * ``PYRAMID = 14``
        * ``PENTAGONAL_PRISM = 15``
        * ``HEXAGONAL_PRISM = 16``
        * ``QUADRATIC_EDGE = 21``
        * ``QUADRATIC_TRIANGLE = 22``
        * ``QUADRATIC_QUAD = 23``
        * ``QUADRATIC_POLYGON = 36``
        * ``QUADRATIC_TETRA = 24``
        * ``QUADRATIC_HEXAHEDRON = 25``
        * ``QUADRATIC_WEDGE = 26``
        * ``QUADRATIC_PYRAMID = 27``
        * ``BIQUADRATIC_QUAD = 28``
        * ``TRIQUADRATIC_HEXAHEDRON = 29``
        * ``QUADRATIC_LINEAR_QUAD = 30``
        * ``QUADRATIC_LINEAR_WEDGE = 31``
        * ``BIQUADRATIC_QUADRATIC_WEDGE = 32``
        * ``BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33``
        * ``BIQUADRATIC_TRIANGLE = 34``

        if grid is pyvista.UnstructuredGrid, use cell_dict get connectivity
        """

        if coordinates.shape[1] == 2:
            coordinates_xyz = np.zeros((coordinates.shape[0], 3))
            coordinates_xyz[:, :2] = coordinates
        else:
            coordinates_xyz = coordinates

        cells, cell_types = [], []
        for connectivity_tuple in connectivity:
            if connectivity_tuple.shape[0] == 3:
                cell_type = 5
            else:
                raise NotImplementedError("Unexpected Cell Type")

            cells.append([connectivity_tuple.shape[0]] + list(connectivity_tuple))
            cell_types.append(cell_type)

        cells = np.array(cells).reshape(-1)
        cell_types = np.array(cell_types)

        grid = pyvista.UnstructuredGrid(cells, cell_types, coordinates_xyz)

        return grid

    @staticmethod
    def triangulate(grid: Union[pyvista.PolyData]):
        return grid.triangulate()

    @staticmethod
    def unstructured_grid_to_polydata(grid: pyvista.UnstructuredGrid) -> pyvista.PolyData:
        return grid.extract_surface()

    @staticmethod
    def compute_surface_normal(grid: pyvista.PolyData, cell_normals=True, point_normals=True):
        res_grid = grid.compute_normals(cell_normals=cell_normals, point_normals=point_normals)
        res = {}
        if cell_normals:
            res.update({'cell_normal': res_grid.cell_data['Normals']})
        if point_normals:
            res.update({'point_normal': res_grid.point_data['Normals']})
        return res


class TopoLogyField(object):
    @staticmethod
    def _create_grid(grid_nums, width, bry_shift):
        grid_dim = grid_nums.shape[0]
        grids = np.meshgrid(*[np.arange(0, i, 1) for i in grid_nums])
        grids = [np.expand_dims(grid, axis=grid_dim) for grid in grids]
        grids = np.concatenate(grids, axis=-1)
        grids = grids.reshape((-1, grid_dim)) * width + width * 0.5 + bry_shift
        return grids

    @staticmethod
    def _split_grid(centers: np.ndarray, box_width):
        center_dim = centers.shape[1]
        width = box_width * 0.5
        if center_dim == 2:
            shifts = [
                np.array([width, width]), np.array([-width, width]),
                np.array([width, -width]), np.array([-width, -width])
            ]
        else:
            shifts = [
                np.array([width, width, width]),
                np.array([width, width, -width]),
                np.array([width, -width, width]),
                np.array([width, -width, -width]),
                np.array([-width, width, width]),
                np.array([-width, width, -width]),
                np.array([-width, -width, width]),
                np.array([-width, -width, -width]),
            ]

        new_centers = []
        for shift in shifts:
            new_centers.append(centers + shift)
        new_centers = np.concatenate(new_centers, axis=0)
        return new_centers

    @staticmethod
    def semi_octree_fit(pcd: np.ndarray, bbox_width):
        """
        方法1：根据需求边长产生n个标准中心，KNN中心分类
        方法2：根据需求边长产生n个标准中心，对中心作radius搜索从而去除空中心，再做无监督分类
        """
        if pcd.shape[0] == 1:
            return np.array([0])

        bdy_max = np.max(pcd, axis=0)
        bry_min = np.min(pcd, axis=0)
        bry_dif = bdy_max - bry_min
        pcd_num, pcd_dim = pcd.shape
        tree = KDTree(pcd)

        bbox_num = np.prod(np.ceil(bry_dif / bbox_width))
        scale_times = np.maximum(math.ceil(np.log2(np.power(bbox_num / 1000., 1. / pcd_dim))), 0.0)
        scale_width = bbox_width * np.power(2, scale_times)
        bry_cell_num = np.ceil(bry_dif / scale_width)
        grid_centers = TopoLogyField._create_grid(bry_cell_num, scale_width, bry_min)

        while True:
            r = np.linalg.norm(np.ones(pcd_dim) * scale_width * 0.6, ord=2)  # scale 0.6 since python float error
            neighbor_set = tree.query_radius(grid_centers, r=r)
            valid_centers = []
            for neighbor_idxs, grid_center in zip(neighbor_set, grid_centers):
                if len(neighbor_idxs) == 0:
                    continue
                center_difs = np.abs(pcd[neighbor_idxs, :] - grid_center)
                if np.any(np.max(center_difs, axis=1) <= scale_width / 2.0 * 1.1):  # scale 1.1 since python float error
                    valid_centers.append(grid_center)
            grid_centers = np.array(valid_centers)

            if scale_width <= bbox_width:
                break
            scale_width = scale_width / 2.0
            grid_centers = TopoLogyField._split_grid(grid_centers, scale_width)

        # cause by python float error
        n_clusters = np.minimum(grid_centers.shape[0], pcd.shape[0])
        model = KMeans(n_clusters=n_clusters, init=grid_centers[:n_clusters, :])
        labels = model.fit_predict(pcd)

        # TopoLogyField._grid_2d_plot(pcd, grid_centers, scale_width)
        # TopoLogyField._cluster_2d_plot(pcd, labels)

        return labels

    @staticmethod
    def _grid_2d_plot(pcd: np.ndarray, grid_centers: np.ndarray, bbox_width):
        fig, ax = plt.subplots()
        plt.scatter(pcd[:, 0], pcd[:, 1], c='b', s=1.0)
        plt.scatter(grid_centers[:, 0], grid_centers[:, 1], c='r')
        for xy in grid_centers:
            rect = patches.Rectangle(
                xy - np.array([bbox_width, bbox_width]) * 0.5, bbox_width, bbox_width,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
        plt.show()

    @staticmethod
    def _cluster_2d_plot(pcd: np.ndarray, labels: np.ndarray):
        for label in np.unique(labels):
            group_pcd = pcd[labels == label]
            plt.scatter(group_pcd[:, 0], group_pcd[:, 1], s=20.0)
        plt.show()

    @staticmethod
    def sigmoid_fun(xs, center, c=1.0, invert=False, exp_ops: Callable = np.exp):
        if invert:
            return 1.0 / (1.0 + exp_ops(1.0 * (xs - center) * c))
        else:
            return 1.0 / (1.0 + exp_ops(-1.0 * (xs - center) * c))

    @staticmethod
    def search_sigmoid_parameters(break_range, c=1.0, reso=1e-3):
        assert 0.0 < reso < 1.0

        min_limit = reso
        max_limit = 1.0 - reso
        while True:
            x_min_break = -1.0 * np.log2((1. - min_limit) / min_limit) / c
            x_max_break = -1.0 * np.log2((1. - max_limit) / max_limit) / c
            x_range = x_max_break - x_min_break
            if x_range < break_range:
                break
            else:
                c += 1.0
        return c

    @staticmethod
    def relu_fun(xs, threshold, c, max_ops: Callable = np.maximum, invert=False):
        if invert:
            return max_ops(-1.0 * (xs - threshold), 0.0) * c
        else:
            return max_ops(xs - threshold, 0.0) * c

    @staticmethod
    def softplus_fun(xs, base, center, grad, invert=False, log_ops: Callable = np.log, exp_ops: Callable = np.exp):
        if invert:
            return log_ops(1 + exp_ops((xs + base - center) * -1.0 * grad))
        else:
            return log_ops(1 + exp_ops((xs + base - center) * grad))

    @staticmethod
    def search_softplus_parameters(grad, lower=1e-2, log_ops: Callable = np.log, exp_ops: Callable = np.exp):
        base = log_ops(exp_ops(lower) - 1.) / grad
        return base

    @staticmethod
    def dist_fun(center, coord, is_square=False, power_ops: Callable = np.power, sqrt_ops: Callable = np.sqrt):
        if center.shape[0] == 2:
            dist_square = power_ops(coord[0] - center[0], 2) \
                          + power_ops(coord[1] - center[1], 2)
        elif center.shape[0] == 3:
            dist_square = power_ops(coord[0] - center[0], 2) \
                          + power_ops(coord[1] - center[1], 2) \
                          + power_ops(coord[2] - center[2], 2)
        else:
            raise NotImplementedError

        if not is_square:
            return sqrt_ops(dist_square)
        return dist_square


class RbfTopoLogyField(TopoLogyField):
    """
    1.不是合理的重建方案，只能模仿邻近边界的 sign distance function
    """

    def __init__(self, rbf_fun: Callable = None):
        if rbf_fun is None:
            self.rbf_fun = self.gaussian_rbf

        self.model_dict = {}

    @staticmethod
    def create_sdf_data(pcd: np.ndarray, normals: np.ndarray, min_dist, max_dist, scale_rho):
        pcd_num = pcd.shape[0]
        scale_num = math.floor(pcd_num * scale_rho)
        tree = KDTree(pcd)

        sdf_pcd = pcd
        sdf_values = np.zeros(sdf_pcd.shape[0])

        scale_sdf_pcd, scale_sdf_values = [], []
        for sign in [-1.0, 1.0]:
            for idx in range(scale_num):
                xyz = pcd[idx]
                normal = normals[idx]
                d = max_dist

                sdf_xyz = xyz + sign * d * normal
                near_idx = tree.query([sdf_xyz], k=1, return_distance=False)[0][0]

                while near_idx != idx:
                    d = d * 0.8
                    if d < min_dist:
                        break

                    sdf_xyz = xyz + sign * d * normal
                    near_idx = tree.query([sdf_xyz], k=1, return_distance=False)[0][0]

                if d < min_dist:
                    continue

                scale_sdf_pcd.append(sdf_xyz)
                scale_sdf_values.append(d * sign)

        scale_sdf_pcd = np.array(scale_sdf_pcd)
        scale_sdf_values = np.array(scale_sdf_values)

        sdf_pcd = np.concatenate([sdf_pcd, scale_sdf_pcd], axis=0)
        sdf_values = np.concatenate([sdf_values, scale_sdf_values], axis=0)

        return sdf_pcd, sdf_values

    @staticmethod
    def gaussian_rbf(dist_mat: np.ndarray, c: float = -3.0):
        assert c < 0.0
        return np.exp(c * np.power(dist_mat, 2))

    @staticmethod
    def inverse_rbf(dist_mat: np.ndarray, c: float = 1.0, epsilon: float = 1.0):
        return 1.0 / np.sqrt(np.power(dist_mat / epsilon, 2) + epsilon)

    def fit_sparse_surface(self, pcd: np.ndarray, normals: np.ndarray, min_dist, max_dist, scale_rho, dist_radius):
        """
        Params:
            pcd: point cloud
        """
        sdf_pcd, sdf_values = self.create_sdf_data(pcd, normals, min_dist, max_dist, scale_rho)

        tree = KDTree(sdf_pcd)
        neighbor_set, dist_set = tree.query_radius(sdf_pcd, r=dist_radius, return_distance=True)

        sdf_pcd_num, coord_dim = sdf_pcd.shape
        indptr, indices, values = [0], [], []
        indptr_sum = 0

        if coord_dim == 2:
            extend_row_idxs = np.array([sdf_pcd_num, sdf_pcd_num + 1, sdf_pcd_num + 2])
        else:
            extend_row_idxs = np.array([sdf_pcd_num, sdf_pcd_num + 1, sdf_pcd_num + 2, sdf_pcd_num + 3])

        # ------ Step 1: Distance Mat and Coefficient Mat
        for idx, (neighbor_idxs, dists_np) in enumerate(zip(neighbor_set, dist_set)):
            xyz = sdf_pcd[idx]
            indptr_sum += len(neighbor_idxs) + coord_dim + 1
            indptr.append(indptr_sum)

            row_idxs = np.concatenate([neighbor_idxs, extend_row_idxs], axis=-1)
            indices.append(row_idxs)

            rbf_value = self.rbf_fun(dists_np)
            value_cell = np.concatenate([rbf_value, np.array([1.0]), xyz], axis=-1)
            values.append(value_cell)

        # ------ Step 2: Coefficient Mat Transpose
        for i in range(coord_dim + 1):
            indptr_sum += sdf_pcd_num
            indptr.append(indptr_sum)
            indices.append(np.arange(0, sdf_pcd_num, 1, dtype=int))

            if i == 0:
                value_cell = np.ones(sdf_pcd_num)
            else:
                value_cell = sdf_pcd[:, i - 1]
            values.append(value_cell)

        indices = np.concatenate(indices, axis=-1)
        values = np.concatenate(values, axis=-1)

        a_mat = ScipySparseUtils.create_csr_mat(
            values, indices, indptr, shape=(sdf_pcd_num + coord_dim + 1, sdf_pcd_num + coord_dim + 1)
        )
        b_vec = np.zeros((sdf_pcd_num + coord_dim + 1))
        b_vec[:sdf_pcd_num] = sdf_values

        coffe = ScipySparseUtils.linear_solve(a_mat, b_vec)
        # coffe = np.dot(np.linalg.inv(a_mat.todense()), b_vec)  # dense solve

        self.model_dict.update({
            'method': 'sparse_surface',
            'coffe': coffe.reshape(-1),
            'coord_dim': coord_dim,
            'support_dim': sdf_pcd_num,
            'dist_radius': dist_radius,
            'support_tree': tree,
        })

    def infer_sparse_surface(self, pcd: np.ndarray):
        if not self.model_dict.get('method', False):
            return

        support_tree: KDTree = self.model_dict['support_tree']
        neighbor_set, dist_set = support_tree.query_radius(pcd, r=self.model_dict['dist_radius'], return_distance=True)

        coord_dim = self.model_dict['coord_dim']
        support_dim = self.model_dict['support_dim']

        if coord_dim == 2:
            extend_row_idxs = np.array([support_dim, support_dim + 1, support_dim + 2])
        else:
            extend_row_idxs = np.array([support_dim, support_dim + 1, support_dim + 2, support_dim + 3])

        indptr, indices, values = [0], [], []
        indptr_sum = 0
        for idx, (neighbor_idxs, dists_np) in enumerate(zip(neighbor_set, dist_set)):
            xyz = pcd[idx]
            indptr_sum += len(neighbor_idxs) + coord_dim + 1
            indptr.append(indptr_sum)

            if len(neighbor_idxs) > 0:
                row_idxs = np.concatenate([neighbor_idxs, extend_row_idxs], axis=-1)
                rbf_value = self.rbf_fun(dists_np)
                value_cell = np.concatenate([rbf_value, np.array([1.0]), xyz], axis=-1)
            else:
                row_idxs = extend_row_idxs
                value_cell = np.concatenate([np.array([1.0]), xyz], axis=-1)

            indices.append(row_idxs)
            values.append(value_cell)

        indices = np.concatenate(indices, axis=-1)
        values = np.concatenate(values, axis=-1)

        a_mat = ScipySparseUtils.create_csr_mat(
            values, indices, indptr, shape=(pcd.shape[0], support_dim + coord_dim + 1)
        )

        sdf_value = a_mat.dot(self.model_dict['coffe'])
        # sdf_value = np.array(np.dot(a_mat.todense(), self.model_dict['coffe'].reshape((-1, 1))))

        return sdf_value


class SparsePointsRegularization(TopoLogyField):
    """
    TODO: 优化有抖动，原因不明
    """

    def __init__(
            self,
            shape_problem: ShapeDataBase, cfg: dict, mu: float,
            name='SparsePointsRegularization'
    ):
        self.name = name
        self.cfg = cfg
        self.mu = mu
        self.shape_problem = shape_problem
        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.constant_empty = dolfinx.fem.Constant(self.shape_problem.domain, 0.0)
        self.cost_form: ufl.Form = self.constant_empty * ufl.dx

        if self.cfg['method'] == 'sigmoid_v1':
            if not self.cfg.get('c', False):
                self.cfg['c'] = self.search_sigmoid_parameters(
                    self.cfg['break_range'], c=1.0, reso=self.cfg['reso']
                )

        elif self.cfg['method'] == 'relu_v1':
            # self.cfg['base'] = self.search_softplus_parameters(
            #     grad=self.cfg['c'], lower=self.cfg['lower']
            # )
            pass

        else:
            raise NotImplementedError("[ERROR]: Non-Valid Method")

    def update_expression(self, bbox_infos: dict, own_point_radius):
        if len(bbox_infos) > 0:
            cost_form = 0.0
            for bbox_name in bbox_infos.keys():
                info: dict = bbox_infos[bbox_name]
                bbox_pcd = info['points']
                obs_radius = info['obs_radius']

                center = np.mean(bbox_pcd, axis=0)
                center_to_point = np.max(np.linalg.norm(bbox_pcd - center, ord=2, axis=1))
                radius = center_to_point + obs_radius + own_point_radius

                dist = self.dist_fun(center, self.coord, is_square=False, sqrt_ops=ufl.sqrt)
                if self.cfg['method'] == 'sigmoid_v1':
                    cost_form += self.sigmoid_fun(
                        dist - radius, 0.0,
                        c=self.cfg['c'], invert=True, exp_ops=ufl.exp
                    ) * ufl.dx

                elif self.cfg['method'] == 'relu_v1':
                    # cost_form += self.softplus_fun(
                    #     dist, base=self.cfg['base'], center=radius, grad=self.cfg['c'],
                    #     invert=True, log_ops=ufl.ln, exp_ops=ufl.exp
                    # ) * ufl.dx

                    cost_form += self.relu_fun(
                        dist, radius, c=self.cfg['c'], max_ops=ufl.max_value, invert=True
                    ) * ufl.dx

                else:
                    raise NotImplementedError

                info.update({'obs_center': center, 'obs_conflict_length': radius})

        else:
            cost_form = self.constant_empty * ufl.dx

        self.cost_form = cost_form

    def compute_objective(self):
        assert self.cost_form is not None
        value = self.mu * AssembleUtils.assemble_scalar(dolfinx.fem.form(self.cost_form))
        return value

    def compute_shape_derivative(self):
        shape_form = ufl.derivative(self.mu * self.cost_form, self.coord, self.test_v)
        return shape_form

    def update(self):
        pass

    def update_weight(self, mu):
        self.mu = mu

    @staticmethod
    def _plot_bbox_meshes(bbox_infos, with_points=True, with_conflict_box=False):
        random_colors = np.random.random((len(bbox_infos), 3))
        meshes = []

        for i, bbox_name in enumerate(bbox_infos.keys()):
            info: dict = bbox_infos[bbox_name]
            bbox_pcd = info['points']
            n_points, pcd_ndim = bbox_pcd.shape

            if pcd_ndim == 2:
                bbox_pcd = np.concatenate([bbox_pcd, np.zeros((n_points, 1))], axis=1)

            if with_points:
                mesh = pyvista.PointSet(bbox_pcd)
                mesh.point_data['color'] = np.ones((n_points, 3)) * random_colors[i, :]
                meshes.append(mesh)

            if with_conflict_box:
                if pcd_ndim == 2:
                    mesh = pyvista.Circle(info['obs_conflict_length'])
                    xyz_center = np.zeros((3,))
                    xyz_center[:2] = info['obs_center']
                    mesh.translate(xyz_center, inplace=True)
                else:
                    mesh = pyvista.Sphere(radius=info['obs_conflict_length'], center=info['obs_center'])
                meshes.append(mesh)
        return meshes


class LimitCenterRegularization(TopoLogyField):
    pass


class PoissonSigmoidField(object):
    """
    todo: 是不是改梯度为sigmoid梯度再做poisson更容易呢??
    """
    pass


type_conflict_regulariztions = Union[
    SparsePointsRegularization, LimitCenterRegularization
]
