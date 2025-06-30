import numpy as np
import pandas as pd
from typing import Callable, Dict, Union, Tuple, Literal, List
import torch
import pyvista
from tqdm import tqdm
import dolfinx
import networkx as nx
import ufl
import scipy
from scipy import sparse
from basix.ufl import element

from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils
from scripts_py.version_9.dolfinx_grad.deform_mesh_utils import MeshObj, MeshUtils
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver
from scripts_py.version_9.dolfinx_grad.remesh_utils import ReMeshObj


class SpringTorchMeshObj(MeshObj):
    """
    基于assemble_stiff_matrix与assemble_b_vec的线性代数计算，网格依然变形严重，
    这是由于刚度可以容纳的能量是限定的，这是个失败的模型
    """

    def __init__(
            self,
            grad_norm: float, stiff_cof=1.0,
            mesh_file: str = None, mesh_name='model', dim=None,
            domain: dolfinx.mesh.Mesh = None,
            cell_tags: dolfinx.mesh.MeshTags = None,
            facet_tags: dolfinx.mesh.MeshTags = None,
    ):
        super().__init__(
            mesh_file=mesh_file, mesh_name=mesh_name, dim=dim,
            domain=domain, cell_tags=cell_tags, facet_tags=facet_tags
        )

        self.bc_free_marker = []
        self.bc_free_dofs: np.ndarray = None
        self.bc_fix_marker = []
        self.bc_fix_dofs: np.ndarray = None

        self.internal_dofs: np.ndarray = None
        self.free_dofs: np.ndarray = None
        self.num_fix_points: int = None
        self.num_free_points: int = None

        self.line_conn: np.ndarray = None
        self.line_mean_len = 0.0

        self.bc_line_conn: Dict[int, Dict[str, np.ndarray]] = {}
        self.internal_line_conn: Dict[int, Dict[str, np.ndarray]] = {}

        self.grad_norm = grad_norm
        self.stiff_cof = stiff_cof
        self.structure_dict: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {}
        self.loss_dict: Dict[str, torch.Tensor] = {}
        self.log_dict = {}

        # raise NotImplementedError('[Error]: 失败求解，无法用梯度求解预期结果')

    def set_dofs(
            self,
            bc_fix_dofs: np.ndarray, bc_fix_marker: List[int],
            bc_free_dofs: np.ndarray, bc_free_marker: List[int]
    ):
        self.bc_free_marker = bc_free_marker
        self.bc_free_dofs = np.array(list(set(bc_free_dofs) - set(bc_fix_dofs)))

        self.bc_fix_marker = bc_fix_marker
        self.bc_fix_dofs = bc_fix_dofs

        self.internal_dofs = np.array(list(
            set(np.arange(0, self.num_points, 1)) - set(self.bc_fix_dofs) - set(self.bc_free_dofs)
        ))
        self.free_dofs = np.array(list(
            set(np.arange(0, self.num_points, 1)) - set(self.bc_fix_dofs)
        ))

        self.num_fix_points = self.bc_fix_dofs.shape[0]
        self.num_free_points = self.free_dofs.shape[0]

    def extract_spring_structure(self):
        self.line_conn = self.grid.extract_all_edges().lines.reshape((-1, 3))[:, 1:3]
        self.line_mean_len = np.mean(np.linalg.norm(
            self.points[self.line_conn[:, 0], :] - self.points[self.line_conn[:, 1], :], ord=2, axis=1
        ))

        graph = nx.Graph()
        graph.add_edges_from(self.line_conn)

        # ------ free boundary line connectivity
        bc_free_line_dict: Dict[int, List[List[int]]] = {}
        for p_idx in self.bc_free_dofs:
            neg_idxs = list(graph.neighbors(p_idx))
            neg_len = len(neg_idxs)
            if neg_len not in bc_free_line_dict.keys():
                bc_free_line_dict[neg_len] = []
            bc_free_line_dict[neg_len].append([p_idx] + neg_idxs)  # first indices is itself, the rest is neighbours
        for neg_len, combin_pack in bc_free_line_dict.items():
            pack_np = np.array(combin_pack)
            dist = np.linalg.norm(
                self.points[pack_np[:, 0]][:, np.newaxis, :] - self.points[pack_np[:, 1:]],
                ord=2, axis=-1
            )
            self.bc_line_conn[neg_len] = {'idxs': pack_np, 'dist': dist}

        # ------ internal node line connectivity
        internal_line_dict: Dict[int, List[List[int]]] = {}
        for p_idx in self.internal_dofs:
            neg_idxs = list(graph.neighbors(p_idx))
            neg_len = len(neg_idxs)
            if neg_len not in internal_line_dict.keys():
                internal_line_dict[neg_len] = []
            internal_line_dict[neg_len].append([p_idx] + neg_idxs)
        for neg_len, combin_pack in internal_line_dict.items():
            pack_np = np.array(combin_pack)
            dist = np.linalg.norm(
                self.points[pack_np[:, 0]][:, np.newaxis, :] - self.points[pack_np[:, 1:]],
                ord=2, axis=-1
            )
            self.internal_line_conn[neg_len] = {'idxs': pack_np, 'dist': dist}

    def cal_internal_force(self, with_bc=True, with_internal=True, disp: np.ndarray = None):
        if disp is not None:
            pcd = self.points.copy()[:, :self.topology_dim] + disp
        else:
            pcd = self.points.copy()[:, :self.topology_dim]

        line_cons = []
        if with_bc:
            line_cons.append(self.bc_line_conn)
        if with_internal:
            line_cons.append(self.internal_line_conn)

        force_node = np.zeros(shape=(self.num_points, self.topology_dim))
        for line_con in line_cons:
            for neg_len, line_cfg in line_con.items():
                pack_idxs = line_cfg['idxs']
                vec_raw = pcd[pack_idxs[:, 1:], :] - pcd[pack_idxs[:, 0], :][:, np.newaxis, :]
                vec_len = np.linalg.norm(vec_raw, ord=2, axis=-1)
                vec_norm = vec_raw / vec_len[:, :, np.newaxis]

                force_mag = (vec_len - line_cfg['dist']) * self.stiff_cof
                # force_mag = (vec_len - self.line_mean_len) * self.stiff_cof

                force_agg = np.sum(vec_norm * force_mag[:, :, np.newaxis], axis=1)
                force_node[pack_idxs[:, 0], :] = force_agg

        return force_node

    def assemble_problem(self, force: np.ndarray, disp_init: np.ndarray = None):
        if disp_init is None:
            disp_init = np.random.normal(loc=0.0, scale=1e-8, size=(self.num_points, self.topology_dim))
        self.structure_dict.update({
            'pcd': TorchUtils.np2tensor(self.points[:, :self.topology_dim], require_grad=False),
            'disp': TorchUtils.np2tensor(disp_init, require_grad=True),
            'force': TorchUtils.np2tensor(force, require_grad=False),
        })

    def cal_constant_node_loss(self):
        disp = self.structure_dict['disp']
        loss_fix = torch.pow(disp[self.bc_fix_dofs, :], 2.0)
        # loss_fix = torch.sqrt(loss_fix)  # 以后不要使用sqrt会导致0的nan
        loss_fix = torch.mean(loss_fix) + torch.max(loss_fix)
        self.loss_dict.update({'loss_fix': loss_fix})

    def cal_force_loss(self):
        disp = self.structure_dict['disp']
        force_outer = self.structure_dict['force']
        pcd_new = self.structure_dict['pcd'] + disp

        loss_sum = 0.0
        # loss_max = torch.tensor(0.0)
        for name, line_con in zip(['bc', 'internal'], [self.bc_line_conn, self.internal_line_conn]):
            for neg_len, line_cfg in line_con.items():
                pack_idxs = line_cfg['idxs']

                vec_raw = pcd_new[pack_idxs[:, 1:]] - pcd_new[pack_idxs[:, 0]][:, None, :]  # center -> neighbours
                vec_len = torch.norm(vec_raw, p=2, dim=-1)
                vec_norm = vec_raw / vec_len[:, :, None]

                # ------ dist_cur>dist_nat: pull force to neighbour
                # force_mag = (vec_len - TorchUtils.np2tensor(line_cfg['dist'], require_grad=False)) * self.stiff_cof
                force_mag = (vec_len - self.line_mean_len) * self.stiff_cof

                force_agg = vec_norm * force_mag[:, :, None]
                force_agg = torch.sum(force_agg, dim=1)
                force_agg += force_outer[pack_idxs[:, 0], :]

                loss_scale = torch.norm(force_agg, p=2, dim=1)
                loss_scale = torch.pow(loss_scale, 2.0)

                # loss_max = torch.maximum(torch.max(loss_scale), loss_max)
                loss_sum += torch.sum(loss_scale)

            loss_agg = loss_sum / float(self.free_dofs.shape[0])
            # loss_agg += loss_max
            self.loss_dict.update({f'loss_force_{name}': loss_agg})

    def cal_regularization_loss(self):
        disp = self.structure_dict['disp']
        pcd_new = self.structure_dict['pcd'] + disp

        # line_mean_len = torch.mean(torch.norm(
        #     pcd_new[self.line_conn[:, 0], :] - pcd_new[self.line_conn[:, 1], :], p=2, dim=1
        # ))

        loss = 0.0
        for name, line_con in zip(['bc', 'internal'], [self.bc_line_conn, self.internal_line_conn]):
            for neg_len, line_cfg in line_con.items():
                pack_idxs = line_cfg['idxs']
                vec_raw = pcd_new[pack_idxs[:, 1:]] - pcd_new[pack_idxs[:, 0]][:, None, :]
                vec_len = torch.norm(vec_raw, p=2, dim=-1)

                mag = (vec_len - TorchUtils.np2tensor(line_cfg['dist'], require_grad=False)) * self.stiff_cof
                mag = torch.sum(torch.pow(mag, 2.0), dim=1)

                # mag = vec_len / TorchUtils.np2tensor(line_cfg['dist'], require_grad=False) - 1.0
                # mag = vec_len - line_mean_len

                loss += torch.sum(mag)
        loss = loss / float(self.free_dofs.shape[0])

        self.loss_dict.update({f'loss_reg': loss})

    def compute_loss(self, weight_dict: Dict[str, float]):
        self.cal_constant_node_loss()
        self.cal_force_loss()
        self.cal_regularization_loss()
        loss = 0.0
        for name, loss_t in self.loss_dict.items():
            loss += loss_t * weight_dict[name]
            loss_np = np.sqrt(self.loss_dict[name].detach().numpy())
            self.log_dict.update({name: loss_np})
        return loss

    def fit_disp(self, min_lr_ratio: float, loss_weight: Dict[str, float], max_iter=1000):
        debug_bar = tqdm()
        optimizer = TorchUtils.get_optimizer(
            [self.structure_dict['disp']], lr=1.0, method='sgd_momentum'
        )  # don't use adam here
        scheduler = TorchUtils.get_lr_scheduler(
            optimizer, factor=0.5, patience=500, cooldown=1000, min_lr=min_lr_ratio
        )

        loss_first = None
        iteration = -1
        while True:
            iteration += 1
            lr = scheduler.get_last_lr()[-1]
            loss = self.compute_loss(loss_weight)

            # ------ debug vis
            desc = f'[iter:{iteration}/lr:{lr:.8f}] '
            for loss_n, loss_f in self.log_dict.items():
                desc += f'{loss_n}:{loss_f:.3f} '
            loss_cur = np.sum(list(self.log_dict.values()))
            if loss_first is None:
                loss_first = loss_cur
            loss_rho = loss_cur / loss_first
            desc += f'loss_rho:{loss_rho:.2f}.'
            debug_bar.set_description(desc)
            debug_bar.update(iteration)
            # print(desc)
            # ------

            loss.backward()
            # ------ clip norm 只能缓解问题
            TorchUtils.clip_grad([self.structure_dict['disp']], method='clip_norm', max_norm=self.grad_norm)

            # if iteration % 20 == 0:
            #     with torch.no_grad():
            #         disp_np = TorchUtils.tensor2np(self.structure_dict['disp'], with_copy=True)
            #         grid_orig = self.grid.copy()
            #         grid_new = self.grid.copy()
            #         if self.topology_dim == 2:
            #             disp_np = np.concatenate([disp_np, np.zeros((disp_np.shape[0], 1))], axis=1)
            #         grid_new.points += disp_np
            #         vis = VisUtils()
            #         vis.plot(grid_orig, style='wireframe')
            #         vis.plot(grid_new, style='wireframe', color=(0., 1., 0.))
            #         vis.plot_grad(grid_orig.points, disp_np, mag=1.0, color=(1., 0., 0.))
            #         vis.show()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            if iteration > max_iter:
                break

            if lr <= min_lr_ratio:
                break

        disp = TorchUtils.tensor2np(self.structure_dict['disp'], with_copy=True)
        return disp

    def draw_mesh(self, vis: VisUtils = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True

        vis.plot(self.grid, style='wireframe', color=(0., 0., 0.))
        vis.plot(pyvista.PointSet(self.points[self.bc_free_dofs, :]), color=(0., 1., 0.))
        vis.plot(pyvista.PointSet(self.points[self.bc_fix_dofs, :]), color=(1., 0., 0.))
        vis.plot(pyvista.PointSet(self.points[self.internal_dofs, :]), color=(1., 0., 1.))
        # vis.plot(pyvista.PointSet(self.points[self.free_dofs, :]), color=(0., 1., 1.))

        # colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 0.), (1., 0., 1.), (0., 1., 1.)]
        # for i, (neg_len, line_cfg) in enumerate(self.line_dict.items()):
        #     print(f'neg_len:{neg_len}, color:{colors[i]}.')
        #     vis.plot(pyvista.PointSet(self.points[line_cfg['idxs'][:, 0], :]), color=colors[i])

        if show_vis:
            vis.show()

    def draw_force_graph(self, vis: VisUtils = None, force_outer: np.ndarray = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        vis.plot(self.grid, style='wireframe')
        for line_con in [self.bc_line_conn, self.internal_line_conn]:
            for neg_len, line_cfg in line_con.items():
                pack_idxs = line_cfg['idxs']
                vec_raw = self.points[pack_idxs[:, 1:], :] - self.points[pack_idxs[:, 0], :][:, np.newaxis, :]
                vec_len = np.linalg.norm(vec_raw, ord=2, axis=-1)
                vec_norm = vec_raw / vec_len[:, :, np.newaxis]
                force_mag = (vec_len - line_cfg['dist']) * self.stiff_cof
                force_all = vec_norm * force_mag[:, :, np.newaxis]
                for i in range(neg_len):
                    vis.plot_grad(
                        self.points[pack_idxs[:, 0], :], force_all[:, i, :], mag=1.0, color=(0., 0., 1.), opacity=0.25
                    )
                force_agg = np.sum(force_all, axis=1)
                vis.plot_grad(self.points[pack_idxs[:, 0], :], force_agg, mag=1.0, color=(1., 0., 0.), opacity=0.25)
        if force_outer is not None:
            vis.plot_grad(self.points, force_outer, mag=1.0, color=(0., 1., 0.), opacity=0.15)
        if show_vis:
            vis.show()

    def assemble_stiff_matrix(self):
        df = pd.DataFrame(self.line_conn, columns=['l0', 'l1'])
        df['l0_fix'] = df['l0'].isin(self.bc_fix_dofs)
        df['l1_fix'] = df['l1'].isin(self.bc_fix_dofs)
        fix_con = df['l0_fix'] & df['l1_fix']
        df = df[~fix_con]

        row_idxs, col_idxs = [], []
        for sub_df in tqdm(df.itertuples()):
            if sub_df.l0_fix:
                row_idxs.append(sub_df.l1)
                col_idxs.append(sub_df.l0)
                row_idxs.append(sub_df.l1 + self.num_points)
                col_idxs.append(sub_df.l0 + self.num_points)

            elif sub_df.l1_fix:
                row_idxs.append(sub_df.l0)
                col_idxs.append(sub_df.l1)
                row_idxs.append(sub_df.l0 + self.num_points)
                col_idxs.append(sub_df.l1 + self.num_points)

            else:
                row_idxs.append(sub_df.l0)
                col_idxs.append(sub_df.l1)
                row_idxs.append(sub_df.l0 + self.num_points)
                col_idxs.append(sub_df.l1 + self.num_points)

                row_idxs.append(sub_df.l1)
                col_idxs.append(sub_df.l0)
                row_idxs.append(sub_df.l1 + self.num_points)
                col_idxs.append(sub_df.l0 + self.num_points)

        row_idxs.extend(self.bc_fix_dofs.tolist())
        col_idxs.extend(self.bc_fix_dofs.tolist())
        row_idxs.extend((self.bc_fix_dofs + self.num_points).tolist())
        col_idxs.extend((self.bc_fix_dofs + self.num_points).tolist())

        a_mat = sparse.coo_matrix(
            (np.full(shape=(len(row_idxs),), fill_value=1.0, dtype=float), (row_idxs, col_idxs)),
            shape=(2 * self.num_points, 2 * self.num_points)
        )
        return a_mat

    def assemble_b_vec(self, force: np.ndarray, a_mat: sparse.coo_matrix):
        b_vec = np.concatenate([force[:, 0], force[:, 1]], axis=-1) * -1.0
        (x, state_code) = sparse.linalg.gmres(a_mat, b_vec, maxiter=None)
        error = np.max(np.abs(a_mat.dot(x) - b_vec))
        print(f'[Error]: {error}')
        disp = np.concatenate([
            x[:self.num_points].reshape((-1, 1)),
            x[self.num_points:].reshape((-1, 1))
        ], axis=1)
        return disp


class SpringFeaMeshObj(MeshObj):
    def __init__(
            self,
            mesh_file: str = None, mesh_name='model', dim=None,
            domain: dolfinx.mesh.Mesh = None,
            cell_tags: dolfinx.mesh.MeshTags = None,
            facet_tags: dolfinx.mesh.MeshTags = None,
    ):
        super().__init__(
            mesh_file=mesh_file, mesh_name=mesh_name, dim=dim,
            domain=domain, cell_tags=cell_tags, facet_tags=facet_tags
        )
        self.bc_free_marker = []
        self.bc_free_dofs: np.ndarray = None
        self.bc_fix_marker = []
        self.bc_fix_dofs: np.ndarray = None

        self.internal_dofs: np.ndarray = None
        self.free_dofs: np.ndarray = None
        self.num_fix_points: int = None
        self.num_free_points: int = None

        self.vector_point_element = element("CG", self.domain.basix_cell(), degree=1, shape=(self.topology_dim,))
        self.scalar_point_element = element("CG", self.domain.basix_cell(), degree=1)

        self.vector_point_functionSpace = dolfinx.fem.functionspace(self.domain, self.vector_point_element)
        self.scalar_point_functionSpace = dolfinx.fem.functionspace(self.domain, self.scalar_point_element)
        self.n_vec = MeshUtils.define_facet_norm(self.domain)
        self.ds = MeshUtils.define_ds(self.domain, self.facet_tags)
        self.dx = MeshUtils.define_dx(self.domain)

        self.re_mesh_obj: ReMeshObj = None

    def set_dofs(
            self,
            bc_fix_dofs: np.ndarray, bc_fix_marker: List[int],
            bc_free_dofs: np.ndarray, bc_free_marker: List[int]
    ):
        self.bc_free_marker = bc_free_marker
        self.bc_free_dofs = np.array(list(set(bc_free_dofs) - set(bc_fix_dofs)))

        self.bc_fix_marker = bc_fix_marker
        self.bc_fix_dofs = bc_fix_dofs

        self.internal_dofs = np.array(list(
            set(np.arange(0, self.num_points, 1)) - set(self.bc_fix_dofs) - set(self.bc_free_dofs)
        ))
        self.free_dofs = np.array(list(
            set(np.arange(0, self.num_points, 1)) - set(self.bc_fix_dofs)
        ))

        self.num_fix_points = self.bc_fix_dofs.shape[0]
        self.num_free_points = self.free_dofs.shape[0]

        self.re_mesh_obj = ReMeshObj(
            grid=self.grid, dim=self.topology_dim,
            bc_dofs=np.concatenate([self.bc_free_dofs, self.bc_fix_dofs], axis=-1),
            internal_dofs=self.internal_dofs
        )

    @staticmethod
    def epsilon(u):
        # return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    @staticmethod
    def sigma(u, lambda_v, mu):
        return lambda_v * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * SpringFeaMeshObj.epsilon(u)

    def re_mesh(self, max_iteration=1000, tol=1e-2, with_debug=False):
        res_dict = self.re_mesh_obj.laplacian_balance_smoothing(
            max_iteration=max_iteration, tol=tol, with_debug=with_debug
        )
        if res_dict['is_valid']:
            res_pcd = res_dict['res_pcd']
            if self.topology_dim == 2:
                res_pcd = np.concatenate([res_pcd, np.zeros(shape=(res_pcd.shape[0], 1))], axis=1)
            self.re_init_points(res_pcd)
        return res_dict['is_valid']

    def fit_disp(
            self,
            method: Literal['linear_elastic', 'poisson_eq', 'linear_elastic_obs'],
            kwargs: Dict,
            with_debug=False,
    ):
        u = ufl.TrialFunction(self.vector_point_functionSpace)
        v = ufl.TestFunction(self.vector_point_functionSpace)
        uh = dolfinx.fem.Function(self.vector_point_functionSpace)

        if method == 'linear_elastic':
            # https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html
            # linear elastic equation is navier equation
            f = dolfinx.fem.Function(self.vector_point_functionSpace)
            f.x.array[:] = kwargs['force'].reshape(-1).astype(float)

            bc_fix = dolfinx.fem.dirichletbc(
                np.zeros(shape=(self.topology_dim,)), self.bc_fix_dofs, self.vector_point_functionSpace
            )
            bcs = [bc_fix]
            T = dolfinx.fem.Constant(self.domain, tuple([0.] * self.topology_dim))

            a_ufl = ufl.inner(
                SpringFeaMeshObj.sigma(u, lambda_v=kwargs['lambda'], mu=kwargs['mu']),
                SpringFeaMeshObj.epsilon(v)
            ) * self.dx
            l_ufl = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * self.ds

        elif method == 'poisson_eq':
            # poisson equation, when f is zero is Laplace equation
            """
            依然是失败的，因为这里边界被定义死了，实际上边界应该灵活的，受力牵引但不一定向力牵引的方向移动，
            流体力学梯度会集中在扭曲区域而不是延展非扭曲区域，所以这种方法依然失败
            """
            bc_func = dolfinx.fem.Function(self.vector_point_functionSpace)
            bc_func.x.array[:] = kwargs['displacement'].reshape(-1)
            bc_free = dolfinx.fem.dirichletbc(bc_func, self.bc_free_dofs)
            bc_fix = dolfinx.fem.dirichletbc(
                np.zeros(shape=(self.topology_dim,)), self.bc_fix_dofs, self.vector_point_functionSpace
            )
            bcs = [bc_fix, bc_free]

            f = dolfinx.fem.Function(self.vector_point_functionSpace)
            a_ufl = ufl.inner(ufl.grad(u), ufl.grad(v)) * self.dx
            l_ufl = ufl.inner(f, v) * self.dx

        elif method == 'linear_elastic_obs':
            """这是个成功的方法，但问题是刚度系数调整必须靠试验"""
            f = dolfinx.fem.Function(self.vector_point_functionSpace)
            f.x.array[:] = kwargs['obs_force'].reshape(-1).astype(float)

            bc_fix = dolfinx.fem.dirichletbc(
                np.zeros(shape=(self.topology_dim,)), self.bc_fix_dofs, self.vector_point_functionSpace
            )
            bcs = [bc_fix]
            T = dolfinx.fem.Constant(self.domain, tuple([0.] * self.topology_dim))

            a_ufl = ufl.inner(
                SpringFeaMeshObj.sigma(u, lambda_v=kwargs['lambda'], mu=kwargs['mu']),
                SpringFeaMeshObj.epsilon(v)
            ) * self.dx
            l_ufl = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * self.ds

        else:
            raise ValueError(f'[Error]: Non-valid method {method}.')

        res = LinearProblemSolver.solve_fea_petsc(
            comm=self.domain.comm,
            uh=uh,
            a_form=dolfinx.fem.form(a_ufl),
            l_form=dolfinx.fem.form(l_ufl),
            bcs=bcs,
            ksp_option={
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1
            },
            with_debug=with_debug
        )
        if with_debug:
            print(f'[Info]: linear elastic equation max_error:{res["max_error"]}')
        if not res['is_solved']:
            return False, None

        disp = uh.x.array.reshape((-1, self.topology_dim)).copy()
        return True, disp


def debug_torch():
    from sklearn.neighbors import KDTree

    mesh = SpringTorchMeshObj(
        grad_norm=1e-4, stiff_cof=1.0,
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example05/model.msh',
        dim=2
    )

    bc_free_marker, bc_free_dofs = [], []
    bc_fix_marker, bc_fix_dofs = [], []
    bry_cfg = {
        'bry_fix': {'marker': 6, 'is_free': False},
        'bry_free': {'marker': 7, 'is_free': True},
    }
    vector_point_func_space = dolfinx.fem.VectorFunctionSpace(mesh.domain, ("CG", 1))

    for name, cfg in bry_cfg.items():
        if cfg['is_free']:
            bc_free_marker.append(cfg['marker'])
            bc_dofs = MeshUtils.extract_entity_dofs(
                vector_point_func_space, mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc_free_dofs.append(bc_dofs)
        else:
            bc_dofs = MeshUtils.extract_entity_dofs(
                vector_point_func_space, mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc_fix_marker.append(cfg['marker'])
            bc_fix_dofs.append(bc_dofs)

    mesh.set_dofs(
        bc_fix_dofs=np.concatenate(bc_fix_dofs, axis=-1), bc_fix_marker=bc_fix_marker,
        bc_free_dofs=np.concatenate(bc_free_dofs, axis=-1), bc_free_marker=bc_free_marker,
    )
    mesh.extract_spring_structure()
    # mesh.draw_mesh()

    force_outer = np.array([
        [0., -0.5]
    ])
    force_pos = np.array([
        [1., 0.25],
    ])

    tree = KDTree(mesh.points[:, :mesh.topology_dim])
    force_indices = tree.query(force_pos, k=1, return_distance=False)
    force_indices = np.concatenate(force_indices, axis=-1)
    strain_indices = np.array(list(set(np.arange(0, mesh.num_points, 1)) - set(force_indices)))

    force_agg = np.zeros(shape=(mesh.num_points, mesh.topology_dim))
    force_agg[force_indices, :] = force_outer

    # # ------
    force_agg_vis = force_agg
    if mesh.topology_dim == 2:
        force_agg_vis = np.concatenate([force_agg_vis, np.zeros((force_agg_vis.shape[0], 1))], axis=1)
    # vis = VisUtils()
    # vis.plot(mesh.grid, style='wireframe')
    # vis.plot_grad(mesh.grid.points, force_agg_vis, mag=1.0, color=(0., 1., 0.))
    # vis.show()
    # # ------

    # disp_init = np.random.normal(loc=0.0, scale=1e-4, size=(mesh.num_points, mesh.topology_dim))
    disp_init = np.zeros(shape=(mesh.num_points, mesh.topology_dim))
    mesh.assemble_problem(force=force_agg, disp_init=disp_init)

    loss_weight = {
        'loss_fix': 1000.0,  # loss_fix=1.0，不动点移动距离为1e-4
        'loss_force_bc': 100.0,  # loss_force=1.0，力的移动距离均衡残差1e-4
        'loss_force_internal': 100.0,  # loss_force=1.0，力的移动距离均衡残差1e-4,
        'loss_reg': 0.0
    }
    # loss = mesh.compute_loss(loss_weight)
    disp = mesh.fit_disp(min_lr_ratio=0.05, max_iter=2000, loss_weight=loss_weight)
    if mesh.topology_dim == 2:
        disp = np.concatenate([disp, np.zeros((disp.shape[0], 1))], axis=1)

    mesh.deform(disp)

    res_force = mesh.cal_internal_force(with_internal=True, with_bc=True)
    print(res_force[force_indices, :] + force_agg[force_indices, :])
    print(np.linalg.norm(res_force[strain_indices, :], axis=1, ord=2).max())
    # if mesh.topology_dim == 2:
    #     res_force = np.concatenate([res_force, np.zeros((res_force.shape[0], 1))], axis=1)

    mesh.draw_force_graph(force_outer=force_agg_vis)

    # vis = VisUtils()
    # vis.plot(mesh.grid, style='wireframe', color=(0., 1., 0.))
    # vis.plot(grid_orig, style='wireframe')
    # vis.plot_grad(grid_orig.points, disp, mag=1.0, color=(1., 0., 0.), opacity=1.0)
    # # vis.plot_grad(mesh.points, force_agg_vis, mag=1.0, color=(0., 1., 0.))
    # # vis.plot_grad(mesh.points, res_force, mag=1.0, color=(0., 0., 1.))
    # vis.show()


def debug_fea():
    mesh = SpringFeaMeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example05/model.msh', dim=2
    )

    bc_free_marker, bc_free_dofs = [], []
    bc_fix_marker, bc_fix_dofs = [], []
    bry_cfg = {
        'bry_fix': {'marker': 6, 'is_free': False},
        'bry_free': {'marker': 7, 'is_free': True},
    }
    vector_point_func_space = dolfinx.fem.VectorFunctionSpace(mesh.domain, ("CG", 1))

    for name, cfg in bry_cfg.items():
        if cfg['is_free']:
            bc_free_marker.append(cfg['marker'])
            bc_dofs = MeshUtils.extract_entity_dofs(
                vector_point_func_space, mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc_free_dofs.append(bc_dofs)
        else:
            bc_dofs = MeshUtils.extract_entity_dofs(
                vector_point_func_space, mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc_fix_marker.append(cfg['marker'])
            bc_fix_dofs.append(bc_dofs)
    mesh.set_dofs(
        bc_fix_dofs=np.concatenate(bc_fix_dofs, axis=-1), bc_fix_marker=bc_fix_marker,
        bc_free_dofs=np.concatenate(bc_free_dofs, axis=-1), bc_free_marker=bc_free_marker,
    )

    # ------ debug vis re-mesh
    grid_orig = mesh.grid.copy()
    mesh.re_mesh(max_iteration=1000)
    vis = VisUtils()
    vis.plot(grid_orig, style='wireframe')
    vis.plot(mesh.grid, style='wireframe', color=(0., 1., 0.))
    vis.show()
    # ------

    move_disp = np.random.uniform(-0.05, 0.05, size=(mesh.num_points, mesh.topology_dim))
    move_disp[mesh.internal_dofs, :] = 0.0
    move_disp[mesh.bc_fix_dofs, :] = 0.0

    move_disp_vis = move_disp
    if mesh.topology_dim == 2:
        move_disp_vis = np.concatenate([move_disp_vis, np.zeros((move_disp_vis.shape[0], 1))], axis=1)

    # ------
    vis = VisUtils()
    vis.plot(mesh.grid, style='wireframe')
    vis.plot_grad(mesh.grid.points, move_disp_vis, mag=1.0, color=(0., 1., 0.))
    vis.show()
    # ------

    is_valid, disp = mesh.fit_disp(
        method='poisson_eq',
        kwargs={'displacement': move_disp},
        with_debug=True
    )
    if mesh.topology_dim == 2:
        disp = np.concatenate([disp, np.zeros((disp.shape[0], 1))], axis=1)

    grid_orig = mesh.grid.copy()
    mesh.deform(disp)

    vis = VisUtils()
    vis.plot(mesh.grid, style='wireframe')
    vis.plot(grid_orig, style='wireframe', color=(0., 1., 0.))
    vis.plot_grad(grid_orig.points, disp, mag=1.0, color=(1., 0., 0.))
    vis.plot_grad(grid_orig.points, move_disp_vis, mag=1.0, color=(0., 0., 1.))
    vis.show()


if __name__ == '__main__':
    debug_fea()
