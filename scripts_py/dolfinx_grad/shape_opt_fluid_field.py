import numpy as np
import dolfinx
from typing import Dict, Union, List, Literal, Tuple
import pyvista
import ufl
from ufl import grad, inner, div, dot, elem_pow
from sklearn.neighbors import KDTree

from scripts_py.version_9.dolfinx_grad.shape_opt_fluid_scalar import ShapeFluidOptimizer
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils
from scripts_py.version_9.dolfinx_grad.gard_fuse_utils import GradRawFusion
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional


class CenterRepelFunctional(IntegralFunctional):
    def __init__(self, name: str, mesh: MeshObj, weight: float = 1.0):
        self.name = name
        self.mesh = mesh
        self.weight = dolfinx.fem.Constant(self.mesh.domain, weight)
        self.coord = MeshUtils.define_coordinate(self.mesh.domain)
        self.dx = MeshUtils.define_dx(self.mesh.domain)

        self.form: ufl.Form = self.weight * self.dx
        self.center_pcd: np.ndarray = np.array([])
        self.require_dist: np.ndarray = np.array([])

    def evaluate(self):
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.form))

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument]
    ):
        return ufl.derivative(self.form, argument, direction)

    def coefficients(self):
        coffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coffs

    def update_center_repel(self, center_pcd=np.array([]), require_dist=np.array([])):
        self.center_pcd = center_pcd
        self.require_dist = require_dist

    def update(self, step_num=100):
        if self.center_pcd.shape[0] == 0:
            self.form = 0.0 * self.dx
        else:
            # todo ufl不支持超过1000的加操作
            assert self.center_pcd.shape[0] < 1e5
            form = 0.0

            form_tmp = 0.0
            for i, (center, threshold) in enumerate(zip(self.center_pcd, self.require_dist)):
                if self.mesh.topology_dim == 2:
                    dist_square = (
                            elem_pow(self.coord[0] - float(center[0]), 2.0)
                            + elem_pow(self.coord[1] - float(center[1]), 2.0)
                    )
                else:
                    dist_square = (
                            elem_pow(self.coord[0] - float(center[0]), 2)
                            + elem_pow(self.coord[1] - float(center[1]), 2)
                            + elem_pow(self.coord[2] - float(center[2]), 2)
                    )
                dist = ufl.sqrt(dist_square)
                form_tmp += ufl.max_value(-1.0 * (dist - threshold), 0.0)

                if (i % step_num == 0) or (i == self.center_pcd.shape[0] - 1):
                    form += form_tmp
                    form_tmp = 0.0

            self.form = self.weight * form * self.dx

    def update_scale(self, weight: float):
        self.weight.value = weight


class ShapeFieldFluidOptimizer(ShapeFluidOptimizer):
    def __init__(
            self,
            mesh_file: str, dim: int, result_dir,
            field_pcd: np.ndarray = None, field_radius=-1.0, field_stiff=1.0
    ):
        super().__init__(mesh_file=mesh_file, dim=dim, result_dir=result_dir)

        self.with_field = (field_pcd is not None) and (field_radius > 0.)
        if self.with_field:
            self.field = dolfinx.fem.Function(self.scalar_point_functionSpace, name='field')
            self.field_pcd: np.ndarray = field_pcd
            self.field_radius: float = field_radius
            self.field_stiff: float = field_stiff
            self.field_cfg: Dict = {'in_field': False}
            self.field_tree = KDTree(self.field_pcd)
            self.field_cost_fun: Dict[str, CenterRepelFunctional] = {}

            # self.field_loss = dolfinx.fem.form(self.field * self.dx)
            # self.field_grad_form = -1.0 * grad(self.field)
            # self.field_grad_scalar_form = inner(grad(self.field), grad(self.field))
            # self.field_grad: dolfinx.fem.Function = dolfinx.fem.Function(self.vector_point_functionSpace)
            # self.coord = MeshUtils.define_coordinate(self.mesh.domain)

    def update_field_cfg(self):
        field_cfg = {'in_field': False, 'field_loss': 0.0}

        if self.with_field:
            idxs_list = self.field_tree.query_radius(self.mesh.points, r=self.field_radius, return_distance=False)
            rel_field_idxs, rel_mesh_idxs = [], []
            for mesh_idx, field_idxs in enumerate(idxs_list):
                if len(field_idxs) == 0:
                    continue
                rel_field_idxs.append(np.array(field_idxs))
                rel_mesh_idxs.append(np.full(shape=(len(field_idxs)), fill_value=mesh_idx, dtype=int))

            if len(rel_field_idxs) > 0:
                rel_field_idxs = np.concatenate(rel_field_idxs, axis=-1)
                rel_mesh_idxs = np.concatenate(rel_mesh_idxs, axis=-1)

                # vec_raw = self.mesh.points[rel_mesh_idxs, :] - self.field_pcd[rel_field_idxs, :]
                # vec_raw = vec_raw[:, :self.mesh.topology_dim]
                # vec_len = np.linalg.norm(vec_raw, ord=2, axis=1)
                # field_scale = (self.field_radius - vec_len) * self.field_stiff
                # field = np.zeros(shape=(self.mesh.num_points,))
                # for mesh_idx in np.unique(rel_mesh_idxs):
                #     field[mesh_idx] = np.max(field_scale[rel_mesh_idxs == mesh_idx])

                field_cfg.update({
                    'in_field': True,
                    'rel_mesh_idxs': rel_mesh_idxs,
                    'rel_field_idxs': rel_field_idxs,
                    # 'field': field
                })

        if field_cfg['in_field']:
            # self.field.x.array[:] = field_cfg['field'].reshape(-1)
            self.field_cfg = {
                # 'field_loss': AssembleUtils.assemble_scalar(self.field_loss),
                'rel_mesh_idxs': field_cfg['rel_mesh_idxs'],
                'rel_field_idxs': field_cfg['rel_field_idxs'],
                # 'field_grad': AssembleUtils.assemble_form_vec(
                #     self.field_grad_form, function_space=self.vector_point_functionSpace
                # ).array.copy().reshape((self.mesh.num_points, -1)),
                # 'field_grad_scalar': AssembleUtils.assemble_form_vec(
                #     self.field_grad_scalar_form, function_space=self.scalar_point_functionSpace
                # )
            }

        else:
            self.field.x.array[:] = 0.0
            self.field_cfg = {'field_loss': 0.0}
        self.field_cfg.update({'in_field': field_cfg['in_field']})

    def add_problem(
            self,
            problem_name: str,
            inlet_cfg: Dict[str, Dict[str, Union[int, np.ndarray, List]]],
            outlet_cfg: Dict[str, Dict[str, Union[int, float]]],
            wall_cfg: Dict[str, Dict],
            constant_nu: float,
            fluid_type: Literal['stoke', 'rans'],
            loss_cfgs: List[Dict[str, Union[str, int, float]]],
            state_ksp_option: Dict = None,
            adjoint_ksp_option: Dict = None,
            gradient_ksp_option: Dict = None,
            su2_cfg: Dict = None,
            scalar_product_method='default'
    ):
        super().add_problem(
            problem_name=problem_name,
            inlet_cfg=inlet_cfg,
            outlet_cfg=outlet_cfg,
            wall_cfg=wall_cfg,
            constant_nu=constant_nu,
            fluid_type=fluid_type,
            loss_cfgs=loss_cfgs,
            state_ksp_option=state_ksp_option,
            adjoint_ksp_option=adjoint_ksp_option,
            gradient_ksp_option=gradient_ksp_option,
            su2_cfg=su2_cfg,
            scalar_product_method=scalar_product_method
        )

        if self.with_field:
            opt_problem: OptimalShapeProblem = self.problem_cfgs[problem_name]['opt_problem']

            # ------ 失败方法，只留作日后参考
            # cost_form = inner(ufl.grad(self.field), self.coord) * self.dx
            # # cost_form = inner(self.field_grad, self.coord) * self.dx
            # cost_fun = IntegralFunctional(
            #     name=f'{problem_name}_loss_field',
            #     domain=self.mesh.domain, form=cost_form, weight=self.field_loss_weight
            # )
            # opt_problem.cost_functional_list.append(cost_fun)

            cost_fun = CenterRepelFunctional(name='center_repel', mesh=self.mesh, weight=self.field_stiff)
            self.field_cost_fun[problem_name] = cost_fun
            opt_problem.cost_functional_list.append(cost_fun)

            if scalar_product_method == 'default':
                opt_problem_new = OptimalShapeProblem(
                    state_system=opt_problem.state_system,
                    shape_problem=opt_problem.shape_problem,
                    cost_functional_list=opt_problem.cost_functional_list,
                    shape_regularization_list=opt_problem.reg_pipe,
                    scalar_product=None,
                    scalar_product_method={'method': "default"}
                )
            elif scalar_product_method == 'Poincare-Steklov operator':
                opt_problem_new = OptimalShapeProblem(
                    state_system=opt_problem.state_system,
                    shape_problem=opt_problem.shape_problem,
                    cost_functional_list=opt_problem.cost_functional_list,
                    shape_regularization_list=opt_problem.reg_pipe,
                    scalar_product=None,
                    scalar_product_method={
                        'method': 'Poincare-Steklov operator',
                        'cell_tags': self.mesh.cell_tags,
                        'facet_tags': self.mesh.facet_tags,
                        'bry_free_markers': self.mesh.bc_free_marker,
                        'bry_fixed_markers': self.mesh.bc_fix_marker,
                        "lambda_lame": 1.0,
                        "damping_factor": 0.2,
                        "mu_fix": 1.0,
                        "mu_free": 1.0,
                        "use_inhomogeneous": False,
                        "inhomogeneous_exponent": 1.0,
                        "update_inhomogeneous": False
                    },
                )
            else:
                raise ValueError(f'[Error]: Non-valid scalar product method:{scalar_product_method}.')

            self.problem_cfgs[problem_name].update({'opt_problem': opt_problem_new})

    def update_field(self):
        if self.with_field:
            self.update_field_cfg()
            for name, cost_fun in self.field_cost_fun.items():
                if self.field_cfg['in_field']:
                    cost_fun.update_center_repel(
                        center_pcd=self.field_pcd[self.field_cfg['rel_field_idxs'], :self.mesh.topology_dim],
                        require_dist=np.full(
                            shape=(self.field_cfg['rel_field_idxs'].shape[0]), fill_value=float(self.field_radius)
                        )
                    )
                else:
                    cost_fun.update_center_repel()

                opt_problem: OptimalShapeProblem = self.problem_cfgs[name]['opt_problem']
                opt_problem.gradient_system.update_gradient_equations()

    def log(self, step):
        for name, cfg in self.problem_cfgs.items():
            if cfg['run_type'] in ['rans', 'stoke']:
                uh: dolfinx.fem.Function = cfg['uh']

                vel_ref = dolfinx.fem.Function(self.vector_point_functionSpace, name='vel')
                vel_ref.interpolate(uh.sub(0).collapse())
                vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.topology_dim))
                if vel_ref_np.shape[1] == 2:
                    vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
                self.recorder.add_point_scalar(f'{name}_velocity', vel_ref_np)

                p_ref = uh.sub(1).collapse()
                self.recorder.add_point_scalar(f'{name}_pressure', p_ref.x.array)

        # if self.with_field:
        #     # self.recorder.add_point_scalar('field_real', self.field.x.array[:].reshape(-1))
        #     u: dolfinx.fem.Function = self.problem_cfgs['aux']['u']
        #     self.recorder.add_point_scalar('field_simulate', u.x.array[:].reshape(-1))

        self.recorder.save(step)

    def draw_problem(self, with_field=True, with_field_detail=True, vis: VisUtils = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        vis.plot(self.mesh.grid, style='wireframe', color=(1., 0., 0.))
        if with_field:
            vis.plot(pyvista.PointSet(self.field_pcd))
            if self.field_cfg['in_field']:
                vis.plot(pyvista.PointSet(self.field_pcd[self.field_cfg['rel_field_idxs'], :]), color=(0., 1., 1.))
                vis.plot(pyvista.PointSet(self.mesh.points[self.field_cfg['rel_mesh_idxs'], :]), color=(0., 1., 1.))
                if with_field_detail:
                    field = self.field_cfg['field']
                    if field.ndim == 2:
                        if field.shape[1] == 2:
                            field = np.concatenate([field, np.zeros(shape=(field.shape[0], 1))], axis=1)
                        vis.plot_grad(self.mesh.grid.points, field, mag=0.1, color=(0., 1., 1.), opacity=0.25)
                    else:
                        self.mesh.grid = VisUtils.set_point_data(self.mesh.grid, field, name='field')
                        vis.plot(self.mesh.grid, scalars='field', show_scalar_bar=True, style='wireframe')
        if show_vis:
            vis.show()


def run_optimize(
        optim: ShapeFieldFluidOptimizer,
        max_iteration=200,
        max_deformation=0.05,
        min_deformation=0.01,
        state_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        adjoint_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        gradient_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        with_debug=False,
        with_re_mesh=False
):
    fuser = GradRawFusion(mesh=optim.mesh)
    step, min_loss = 0, np.inf
    while True:
        optim.update_field()
        is_solved, grad_fluid_dict = optim.compute_fluid_grad(
            state_mat_assemble_method=state_mat_assemble_method,
            adjoint_mat_assemble_method=adjoint_mat_assemble_method,
            gradient_mat_assemble_method=gradient_mat_assemble_method,
            with_debug=with_debug
        )
        if not is_solved:
            print('[Info]: solving linear elastic equation fail.')
            return False

        for i, (name, pcd_grad) in enumerate(grad_fluid_dict.items()):
            pcd_grad = pcd_grad.reshape((-1, optim.mesh.topology_dim)) * -1.0
            fuser.add_grad(name, pcd_grad)

        disp = fuser.solve_disp(max_step=max_deformation)
        if optim.mesh.topology_dim == 2:
            disp = np.concatenate([disp, np.zeros((disp.shape[0], 1))], axis=1)

        # # ------
        # vis = VisUtils()
        # optim.draw_problem(with_field=True, with_field_detail=True, vis=vis)
        # vis.plot_grad(optim.mesh.grid.points, disp, color=(0., 0., 1.), mag=1.0, opacity=0.5)
        # vis.show()
        # # ------

        loss_dict = optim.get_loss_dict(
            update_state=False, mat_assemble_method=state_mat_assemble_method, with_debug=with_debug
        )
        # loss_dict.update({'field_loss': optim.field_cfg['field_loss']})
        loss_sum = np.sum(list(loss_dict.values()))

        # ------
        desc = f'[iter:{step}] '
        for loss_n, loss_f in loss_dict.items():
            desc += f'{loss_n}:{loss_f:.3f} '
        desc += f'loss:{loss_sum:.3f}'
        print(desc)
        optim.log(step)

        is_valid = optim.update_geometry(
            displacement=disp,
            deform_method='line_search',
            min_deform=min_deformation,
            max_deform=max_deformation,
            line_search_step=5,
            with_vol_check=False,
            with_angle_check=False
        )
        if not is_valid:
            print('[Info]: update geometry fail.')
            return False

        if with_re_mesh:
            optim.mesh.re_mesh()

        step += 1
        if step > max_iteration:
            print('[Info]: reach max iteration.')
            break

        # min_loss = np.minimum(min_loss, loss_sum)
        # if loss_sum > min_loss * 1.1:
        #     print('[Info]: loss overfit.')
        #     break

    # ------ last log
    loss_dict = optim.get_loss_dict(
        update_state=False, mat_assemble_method=state_mat_assemble_method, with_debug=with_debug
    )
    loss_sum = np.sum(list(loss_dict.values()))
    print(f"[Debug]: Step_{step} loss_fluid:{loss_sum:.3f}")
    desc = f'[iter:{step}] '
    for loss_n, loss_f in loss_dict.items():
        desc += f'{loss_n}:{loss_f:.3f} '
    desc += f'loss:{loss_sum:.3f}'
    print(desc)
    optim.log(step)

    return True


def debug():
    field_mesh = pyvista.read('/home/admin123456/Desktop/work/topopt_exps/new_test/example04/obs.stl')

    optim = ShapeFieldFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/cache',
        field_pcd=np.asarray(field_mesh.points),
        field_radius=0.2,
        field_stiff=1e3,
    )

    optim.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=0.6,
        weight=100.0
    )
    optim.define_control_wall({
        'inlet': {'marker': 35, 'is_free': False},
        'outlet': {'marker': 36, 'is_free': False},
        'wall': {'marker': 37, 'is_free': True},
    })

    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 35,
                'value': [1., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 36,
                'value': 0.0
            }
        },
        wall_cfg={
            'wall': {
                'marker': 37
            }
        },
        constant_nu=0.01,
        fluid_type='stoke',
        loss_cfgs=[
            {
                'style': 'minimum_dissipation',
                'weight': 1.0
            }
        ],
        su2_cfg={
            'run_iter': 1000,
            'constant_viscosity': 0.01,
            'refit_by_fenics': False
        },
        # scalar_product_method='Poincare-Steklov operator',
    )

    run_optimize(
        optim=optim,
        max_iteration=100,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',
        max_deformation=0.03,
        min_deformation=0.01,
        with_debug=False,
        with_re_mesh=True
    )


if __name__ == '__main__':
    debug()
