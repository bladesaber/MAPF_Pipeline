import numpy as np
import dolfinx
from typing import Dict, Union, List, Literal, Tuple
import pyvista

from scripts_py.version_9.dolfinx_grad.shape_opt_fluid_scalar import ShapeFluidOptimizer
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils
from scripts_py.version_9.dolfinx_grad.gard_fuse_utils import GradRawFusion


class ShapeFieldFluidOptimizer(ShapeFluidOptimizer):
    """
    todo: 好似由于保守场的问题，力场不能够回避障碍物，
              但这也表明我不需要body_force的参与，我只需要一个表述势能的函数f（甚至不是一个全局势能函数，而是一个局部势能函数）
              然后对这个势能函数在mesh上积分作为惩罚变量就足够了，这样甚至不需要设置body_force从而避免仿真器的修改
        然后修正表面曲率与re-mesh问题就完美了
    """
    def __init__(
            self,
            mesh_file: str, dim: int, result_dir, constant_field: np.ndarray = None,
            field_pcd: np.ndarray = None, field_radius: float = -1.0, field_stiff: float = 1.0
    ):
        super().__init__(mesh_file=mesh_file, dim=dim, result_dir=result_dir)

        self.constant_field = constant_field if constant_field is not None else np.array([0.] * self.mesh.topology_dim)
        self.field_pcd = field_pcd
        self.field_radius = field_radius
        self.field_stiff = field_stiff
        self.field_cfg: Dict = {'in_field': False}
        self.with_field = (self.field_pcd is not None) and (field_radius > 0.)

    def cal_field_force(self):
        # if self.with_field:
        #     rel_mesh_idxs, rel_field_idxs = self.mesh.detect_obstacle(
        #         obs_pcd=self.field_pcd, obs_radius=self.field_radius, unit_lr=0.0
        #     )
        #     if rel_mesh_idxs.shape[0] > 0:
        #         vec_raw = self.mesh.points[rel_mesh_idxs, :self.mesh.topology_dim] - \
        #                   self.field_pcd[rel_field_idxs, :self.mesh.topology_dim]
        #         vec_len = np.linalg.norm(vec_raw, ord=2, axis=1)
        #         vec_norm = vec_raw / vec_len[:, np.newaxis]
        #         force_scale = (self.field_radius - vec_len) * self.field_stiff
        #         force_pcd = vec_norm * force_scale.reshape((-1, 1))
        #
        #         force_field = np.zeros(shape=(self.mesh.num_points, self.mesh.topology_dim))
        #         for mesh_idx in np.unique(rel_mesh_idxs):
        #             force_agg = np.mean(force_pcd[rel_mesh_idxs == mesh_idx], axis=0)
        #             force_field[mesh_idx, :] = force_agg
        #
        #         self.field_cfg.update({
        #             'in_field': True,
        #             'rel_mesh_idxs': rel_mesh_idxs,
        #             'rel_field_idxs': rel_field_idxs,
        #             'force_field': force_field
        #         })
        #         return
        # self.field_cfg = {'in_field': False}

        force_field = np.zeros(shape=(self.mesh.num_points, self.mesh.topology_dim))
        select_idxs = self.mesh.points[:, 0] < 3.0
        force_field[select_idxs, 1] = -200.0  # todo too sensitivity
        self.field_cfg.update({
            'in_field': True,
            'rel_mesh_idxs': None,
            'rel_field_idxs': None,
            'force_field': force_field
        })

    def update_force_body(self):
        force_body = np.zeros(shape=(self.mesh.num_points, self.mesh.topology_dim))
        force_body += self.constant_field

        self.cal_field_force()
        if self.field_cfg['in_field']:
            force_body += self.field_cfg['force_field']

        for name, cfg in self.problem_cfgs.items():
            cfg['force_body'].x.array[:] = force_body.reshape(-1)

        # if self.field_cfg['in_field']:
        #     self.draw_problem()

    def log(self, step):
        for name, cfg in self.problem_cfgs.items():
            uh: dolfinx.fem.Function = cfg['uh']

            vel_ref = dolfinx.fem.Function(self.vector_point_functionSpace, name='vel')
            vel_ref.interpolate(uh.sub(0).collapse())
            vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.topology_dim))
            if vel_ref_np.shape[1] == 2:
                vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
            self.recorder.add_point_scalar(f'{name}_velocity', vel_ref_np)

            p_ref = uh.sub(1).collapse()
            self.recorder.add_point_scalar(f'{name}_pressure', p_ref.x.array)

            force_body_np = cfg['force_body'].x.array.reshape((-1, self.mesh.topology_dim))
            if force_body_np.shape[1] == 2:
                force_body_np = np.concatenate([force_body_np, np.zeros((force_body_np.shape[0], 1))], axis=1)
            self.recorder.add_point_scalar(f'{name}_body_force', force_body_np)

        self.recorder.save(step)

    def draw_problem(self, with_field=True, with_field_force=True, vis: VisUtils = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        vis.plot(self.mesh.grid, style='wireframe', color=(1., 0., 0.))
        if with_field:
            vis.plot(pyvista.PointSet(self.field_pcd))
            if self.field_cfg['in_field']:
                # vis.plot(pyvista.PointSet(self.field_pcd[self.field_cfg['rel_field_idxs'], :]), color=(0., 1., 1.))
                # vis.plot(pyvista.PointSet(self.mesh.points[self.field_cfg['rel_mesh_idxs'], :]), color=(0., 1., 1.))
                if with_field_force:
                    force_body = self.field_cfg['force_field']
                    if force_body.shape[1] == 2:
                        force_body = np.concatenate([force_body, np.zeros(shape=(force_body.shape[0], 1))], axis=1)
                    vis.plot_grad(self.mesh.grid.points, force_body, mag=0.1, color=(0., 1., 1.), opacity=0.25)

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
        optim.update_force_body()

        is_solved, grad_fluid_dict = optim.compute_fluid_grad(
            state_mat_assemble_method=state_mat_assemble_method,
            adjoint_mat_assemble_method=adjoint_mat_assemble_method,
            gradient_mat_assemble_method=gradient_mat_assemble_method,
            with_debug=with_debug
        )
        if not is_solved:
            print('[Info]: solving linear elastic equation fail.')
            return False

        for name, pcd_grad in grad_fluid_dict.items():
            pcd_grad = pcd_grad.reshape((-1, optim.mesh.topology_dim)) * -1.0
            fuser.add_grad(name, pcd_grad)
        disp = fuser.solve_disp(max_step=0.05)
        if optim.mesh.topology_dim == 2:
            disp = np.concatenate([disp, np.zeros((disp.shape[0], 1))], axis=1)

        # # ------
        # vis = VisUtils()
        # optim.draw_problem(with_field=True, with_field_force=False, vis=vis)
        # vis.plot_grad(optim.mesh.grid.points, disp, color=(0., 0., 1.), mag=5.0, opacity=0.5)
        # vis.show()
        # # ------

        loss_dict = optim.get_loss_dict(
            update_state=False, mat_assemble_method=state_mat_assemble_method, with_debug=with_debug
        )
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
            optim.mesh.re_mesh()  # todo need more adjust

        step += 1
        if step > max_iteration:
            print('[Info]: reach max iteration.')
            break

        # min_loss = np.minimum(min_loss, loss_sum)  # todo too sensitivity
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
    field_mesh = pyvista.read('/home/admin123456/Desktop/work/topopt_exps/new_test/example05/obs.stl')

    optim = ShapeFieldFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example05/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example05/cache',
        constant_field=np.array([0.0, 0.0]),
        field_pcd=np.asarray(field_mesh.points),
        field_radius=1.5,
        field_stiff=300.0  # todo too sensitivity
    )

    optim.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=0.5,  # todo too sensitivity
        weight=200.0  # todo too sensitivity
    )
    optim.define_control_wall({
        'inlet': {'marker': 6, 'is_free': False},
        'outlet': {'marker': 7, 'is_free': False},
        'wall': {'marker': 8, 'is_free': True},
    })

    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 6,
                'value': [1., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 7,
                'value': 0.0
            }
        },
        wall_cfg={
            'wall': {
                'marker': 8
            }
        },
        constant_nu=1.0,   # todo too sensitivity
        fluid_type='stoke',
        loss_cfgs=[
            # {
            #     'style': 'minimum_dissipation',
            #     'weight': 1.0
            # }
            {
                'style': 'minimum_stress',  # todo too sensitivity
                'marker': 6,
                'weight': 1.0
            }
        ],
        su2_cfg={
            'run_iter': 1000,
            'constant_viscosity': 0.01,
            'refit_by_fenics': False
        },
        # scalar_product_method='Poincare-Steklov operator'
    )

    # # ------ constant field debug
    # optim.update_force_body()
    # optim.simulate(mat_assemble_method='lift', with_debug=True)
    # optim.log(0)

    run_optimize(
        optim=optim,
        max_iteration=80,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',
        max_deformation=0.02,
        min_deformation=0.01,
        with_debug=False,
        with_re_mesh=False
    )


if __name__ == '__main__':
    debug()
