import os
import shutil
import numpy as np
import dolfinx
import pandas as pd
import ufl
from typing import Dict, Union, List, Literal, Tuple
import pyvista

from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
from scripts_py.version_9.dolfinx_grad.dynamic_recorder import VtuRecorder
from scripts_py.version_9.simulator.su2_simulator import Su2Simulator
from scripts_py.version_9.dolfinx_grad.deform_mesh_utils import DeformMeshObj

from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import VolumeRegularization, \
    ShapeRegularizationPipe
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional, \
    ScalarTrackingFunctional

from scripts_py.version_9.dolfinx_grad.shape_opt_fluid import ShapeFluidOptimizer, Su2SimulatorPacker


class ShapeFfdFluidOptimizer(ShapeFluidOptimizer):
    def __init__(
            self,
            mesh_file: str, dim: int, result_dir,
            ffd_num: List[int] = None, ffd_degree: List[int] = None
    ):
        self.result_dir = result_dir
        assert mesh_file.endswith('.msh')
        self.mesh_file = mesh_file
        self.su2_file = os.path.join(result_dir, 'mesh.su2')
        Su2Simulator.msh_to_su2(self.mesh_file, su2_file=self.su2_file)

        self.mesh = DeformMeshObj(
            ffd_num=ffd_num, ffd_degree=ffd_degree,
            mesh_file=mesh_file, dim=dim
        )
        self.W = dolfinx.fem.FunctionSpace(
            self.mesh.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.mesh.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.mesh.domain.ufl_cell(), 1)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()

        self.vector_cell_functionSpace = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("DG", 0))
        self.scalar_cell_functionSpace = dolfinx.fem.FunctionSpace(self.mesh.domain, ("DG", 0))
        self.vector_point_functionSpace = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("CG", 1))
        self.scalar_point_functionSpace = dolfinx.fem.FunctionSpace(self.mesh.domain, ("CG", 1))

        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)

        self.surface_area = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.ds
        ))
        self.volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.dx
        ))

        self.bcs_control = []
        self.regularization_list: List[Dict] = []
        self.problem_cfgs = {}

        self.recorder = VtuRecorder(self.mesh, save_dir=os.path.join(result_dir, 'result_vtu'), is_binary=True)
        self.var_grad: np.ndarray = None

    def define_control_wall(self, bry_cfg: Dict):
        super().define_control_wall(bry_cfg=bry_cfg)
        fix_dofs = []
        for name, cfg in bry_cfg.items():
            bc_dofs = MeshUtils.extract_entity_dofs(
                self.scalar_point_functionSpace, self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                )
            )
            fix_dofs.append(bc_dofs)
        fix_dofs = np.concatenate(fix_dofs, axis=-1)
        self.mesh.set_fix_boundary(fix_dofs)

    def fuse_grad(self, grad_dict: Dict[str, np.ndarray]):
        grad_merge = np.zeros((self.mesh.points.shape[0], self.mesh.topology_dim), dtype=float)
        for name, pcd_grad in grad_dict.items():
            grad_merge += pcd_grad.reshape((-1, self.mesh.topology_dim))

        if self.mesh.topology_dim == 2:
            grad_merge = np.concatenate([grad_merge, np.zeros((grad_merge.shape[0], 1))], axis=1)

        grad_merge = grad_merge * -1.0
        var_grad = self.mesh.compute_cof_grad(grad_merge)
        self.var_grad = var_grad / np.max(np.abs(var_grad)) * 0.1

    def update_control(self, min_deform=0.01, max_deform=0.1, line_search_step=5):
        # is_valid = self.mesh.deform_safe(displacement_np=self.var_grad)

        scales = (np.linspace(0.1, 1.0, line_search_step)[::-1]).tolist()
        is_valid = self.mesh.deform_line_search(
            self.var_grad, scales=scales,
            with_vol_check=True, vol_change=0.2,
            with_angle_check=True, angle_tol_max=160.0, angle_tol_min=20.0
        )

        # ------ only for debug
        # is_valid = True
        # self.mesh.deform(ffd_cof_dif=para_dif)

        if not is_valid:
            return False

        # ----------------------------------------------------------
        for name, cfg in self.problem_cfgs.items():
            if cfg['fluid_type'] == 'rans':
                simulator: Su2SimulatorPacker = cfg['simulator']
                simulator.update_geometry()
        return True


def debug():
    optim = ShapeFfdFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/cache',
        ffd_num=[30, 30], ffd_degree=[3, 3]
    )
    optim.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=1.0,
        weight=2000.0
    )
    optim.define_control_wall({
        'inlet': {'marker': 13},
        'outlet': {'marker': 14},
        'wallFixed': {'marker': 15},
    })
    # optim.mesh.draw_mesh()

    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 13,
                'value': [3., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 14,
                'value': 0.0
            }
        },
        wall_cfg={
            'wallFixed': {
                'marker': 15
            },
            'wallFree': {
                'marker': 16
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
        }
    )

    # # ------ test simulation
    # optim.simulate()
    # optim.log(0)

    # ------ test optimize
    optim.run(
        max_iteration=1000,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',  # 必须使用Identity_row否则边界条件不正确
        with_debug=False
    )


if __name__ == '__main__':
    debug()
