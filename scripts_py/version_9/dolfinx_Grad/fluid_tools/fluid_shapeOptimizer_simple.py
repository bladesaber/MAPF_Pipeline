import numpy as np
import dolfinx
import os
import shutil
import ufl
from ufl import sym, grad, nabla_grad, dot, inner, div, Identity
from typing import List, Union, Dict, Tuple
import ctypes
import pickle

from ..recorder_utils import VTKRecorder, TensorBoardRecorder
from ..dolfinx_utils import MeshUtils, AssembleUtils, BoundaryUtils
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver
from ..lagrange_method.type_database import create_shape_problem, create_state_problem, ShapeDataBase
from ..lagrange_method.problem_state import StateProblem
from ..lagrange_method.cost_functions import CostFunctional_types
from ..lagrange_method.solver_optimize import OptimalShapeProblem
from ..lagrange_method.shape_regularization import ShapeRegularization
from ..remesh_helper import MeshDeformationRunner
from ..optimizer_utils import CostConvergeHandler, CostWeightHandler
from ..vis_mesh_utils import VisUtils


class FluidShapeOptSimple(object):
    """
    1. mesh不可以有90度角，否则Assemble后的矩阵有奇异
    2. msh文件中，每一个entity(curve, surface等)只能在一个physical group，否则xdmf会有歧义
    """

    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            Re: float,
            deformation_cfg: Dict,
            isStokeEqu=False,
    ):
        """
        Re: Reynolds number
        """

        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.W = dolfinx.fem.FunctionSpace(
            domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))
        self.n_vec = MeshUtils.define_facet_norm(domain)
        self.ds = MeshUtils.define_ds(domain, facet_tags)

        self.up = dolfinx.fem.Function(self.W, name='state')
        self.u, self.p = ufl.split(self.up)
        self.vq = dolfinx.fem.Function(self.W, name='adjoint')
        v, q = ufl.split(self.vq)
        f = dolfinx.fem.Constant(domain, np.zeros(self.tdim))
        self.nu = dolfinx.fem.Constant(domain, 1. / Re)

        if isStokeEqu:
            self.F_form = (
                    self.nu * inner(grad(self.u), grad(v)) * ufl.dx
                    - self.p * div(v) * ufl.dx
                    - q * div(self.u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
            self.is_linear_state = True
        else:
            self.F_form = (
                    self.nu * inner(grad(self.u), grad(v)) * ufl.dx
                    + inner(grad(self.u) * self.u, v) * ufl.dx
                    - inner(self.p, div(v)) * ufl.dx
                    + inner(div(self.u), q) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
            self.is_linear_state = False

        self.V_S = dolfinx.fem.FunctionSpace(
            self.domain, self.domain.ufl_domain().ufl_coordinate_element()
        )

        self.energy_form = dolfinx.fem.form(ufl.inner(self.u, self.u) * ufl.dx)
        self.energy_loss_form = dolfinx.fem.form(ufl.inner(grad(self.u), grad(self.u)) * ufl.dx)

        self.bcs_info_state = []
        self.bcs_info_control = []
        self.state_system: StateProblem = None
        self.control_problem: ShapeDataBase = None
        self.cost_weight: dict = {}
        self.opt_problem: OptimalShapeProblem = None
        self.deformation_handler = MeshDeformationRunner(
            self.domain,
            **deformation_cfg
        )

    def add_state_boundary(self, value: dolfinx.fem.Function, marker: int, is_velocity: bool):
        if is_velocity:
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(value, bc_dofs, self.W0)
            self.bcs_info_state.append((bc, self.W0, bc_dofs, value))

        else:
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(value, bc_dofs, self.W1)
            self.bcs_info_state.append((bc, self.W1, bc_dofs, value))

    def add_control_boundary(self, value: dolfinx.fem.Function, marker: int):
        bc_dofs = MeshUtils.extract_entity_dofs(
            self.V_S, self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
        )
        bc = dolfinx.fem.dirichletbc(value, bc_dofs, None)
        self.bcs_info_control.append((bc, self.V_S, bc_dofs, value))

    def state_initiation(
            self,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    ):
        self.state_system = StateProblem([
            create_state_problem(
                name='state_1',
                F_form=self.F_form,
                state=self.up,
                adjoint=self.vq,
                is_linear=self.is_linear_state,
                bcs_info=self.bcs_info_state,
                state_ksp_option=state_ksp_option,
                adjoint_ksp_option=adjoint_ksp_option
            )
        ])

        self.control_problem = create_shape_problem(
            domain=self.domain,
            bcs_info=self.bcs_info_control,
            gradient_ksp_option=gradient_ksp_option
        )

    def optimization_initiation(
            self,
            cost_functional_list: List[CostFunctional_types],
            cost_weight: Dict,
            shapeRegularization: ShapeRegularization,
            scalar_product_method,
    ):
        self.cost_weight = cost_weight
        self.opt_problem = OptimalShapeProblem(
            state_system=self.state_system,
            shape_problem=self.control_problem,
            shape_regulariztions=shapeRegularization,
            cost_functional_list=cost_functional_list,
            scalar_product=None,
            scalar_product_method=scalar_product_method
        )

    def log_step(self, logger_dicts: dict, log_recorder, u_recorder, step):
        for tag_name in logger_dicts.keys():
            log_dict = logger_dicts[tag_name]
            if len(log_dict) == 1:
                marker_name = list(log_dict.keys())[0]
                value = AssembleUtils.assemble_scalar(log_dict[marker_name])
                log_recorder.write_scalar(marker_name, value, step)
            else:
                data_cell = {}
                for marker_name in log_dict.keys():
                    data_cell[marker_name] = AssembleUtils.assemble_scalar(log_dict[marker_name])
                log_recorder.write_scalars(tag_name, data_cell, step)

        log_recorder.write_scalar('energy', AssembleUtils.assemble_scalar(self.energy_form), step)
        log_recorder.write_scalar('energy_loss', AssembleUtils.assemble_scalar(self.energy_loss_form), step)
        u_recorder.write_function(self.up.sub(0).collapse(), step)

    def solve(self, record_dir, logger_dicts={}, max_iter=200, **kwargs):
        # ------ Step 1: Create Record Directory
        simulate_dir = os.path.join(record_dir, 'simulate')
        if os.path.exists(simulate_dir):
            shutil.rmtree(simulate_dir)
        os.mkdir(simulate_dir)
        u_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

        tensorBoard_dir = os.path.join(record_dir, 'log')
        if os.path.exists(tensorBoard_dir):
            shutil.rmtree(tensorBoard_dir)
        os.mkdir(tensorBoard_dir)
        log_recorder = TensorBoardRecorder(tensorBoard_dir)

        # ------ Step 2: Init Calculation and log first step
        self.opt_problem.state_system.solve(self.domain.comm, with_debug=kwargs.get('with_debug', False))
        if self.cost_weight is not None:
            weight_handler = CostWeightHandler()
            for cost_func in self.opt_problem.cost_functional_list:
                cost = cost_func.evaluate()
                weight_handler.add_cost(cost_func.name, cost)
            weight_handler.compute_weight(self.cost_weight)
            for cost_func in self.opt_problem.cost_functional_list:
                cost_func.update_scale(weight_handler.get_weight(cost_func.name))
                print(f"[Info] Cost Weight: {cost_func.name}: {weight_handler.get_weight(cost_func.name)}")

        init_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
        loss_storge_ctype = ctypes.c_double(init_loss)
        cost_converger = CostConvergeHandler(stat_num=25, warm_up_num=25, tol=5e-3, scale=1.0 / init_loss)

        self.log_step(logger_dicts, log_recorder, u_recorder, step=0)
        tol_rho = kwargs.get('opt_tol_rho', 0.05)

        def detect_cost_valid_func():
            loss = self.opt_problem.evaluate_cost_functional(
                self.domain.comm, update_state=True, with_debug=kwargs.get('with_debug', False)
            )
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
            return is_valid

        step = 0
        while True:
            step += 1

            shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(
                self.domain.comm,
                state_kwargs={'with_debug': kwargs.get('with_debug', False)},
                adjoint_kwargs={'with_debug': kwargs.get('with_debug', False)},
                gradient_kwargs={
                    'with_debug': kwargs.get('with_debug', False),
                    'A_assemble_method': 'Identity_row'
                },
            )

            shape_grad_np = shape_grad.x.array
            shape_grad_np = shape_grad_np * -1.0
            displacement_np = np.zeros(self.domain.geometry.x.shape)
            displacement_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

            # # ------ Just For Debug
            # grid = VisUtils.convert_to_grid(self.domain)
            # grid['grad_test'] = displacement_np
            # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=1.0).show()
            # raise ValueError
            # # ------

            success_flag, stepSize = self.deformation_handler.move_mesh_by_line_search(
                displacement_np, max_iter=10, init_stepSize=1.0, stepSize_lower=1e-3,
                detect_cost_valid_func=detect_cost_valid_func,
                max_step_limit=kwargs.get('max_step_limit', 0.1)
            )

            if success_flag:
                loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
                loss_storge_ctype.value = loss

                is_converge = cost_converger.is_converge(loss)

                # ------ record
                scale_loss = cost_converger.compute_scale_loss(loss)
                log_recorder.write_scalar('scale_loss', scale_loss, step=step)
                log_recorder.write_scalar('scale_loss_var', cost_converger.scale_cost_variation, step=step)
                self.log_step(logger_dicts, log_recorder, u_recorder, step=step)

                # ------ debug output
                print(f"[###Info {step}] loss:{loss:.8f}, stepSize:{stepSize}")
                # ------------------

                if is_converge:
                    break

                if step > max_iter:
                    break

            else:
                break

    def load_initiation_pickle(self, u_pickle_file: str, p_pickle_file: str):
        with open(u_pickle_file, 'rb') as f:
            u_array = pickle.load(f)
        with open(p_pickle_file, 'rb') as f:
            p_array = pickle.load(f)

        self.up.x.array[self.V_to_W_dofs] = u_array
        self.up.x.array[self.Q_to_W_dofs] = p_array
