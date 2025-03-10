import numpy as np
import ufl
import dolfinx
from typing import List, Union, Dict, Tuple

from scripts_py.version_9.dolfinx_grad.dolfinx_utils import BoundaryUtils

KspOption = Dict[str, Union[int, float, str, None]]


class GovDataBase(object):
    default_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }

    def __init__(
            self,
            name: str,
            F_form: ufl.Form,
            state: Union[dolfinx.fem.Function, ufl.TrialFunction],
            adjoint: Union[dolfinx.fem.Function, ufl.TestFunction],
            is_linear: bool,
            state_ksp_option: KspOption = None,
            adjoint_ksp_option: KspOption = None,
            snes_option: KspOption = None,
            snes_criterion: KspOption = None
    ):
        self.name = name

        self.F_form = F_form
        self.is_linear = is_linear
        self.bcs: List[dolfinx.fem.DirichletBC] = []
        self.bcs_infos: List[Tuple[
            dolfinx.fem.FunctionSpace, np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]] = []
        self.homogenize_bcs: List[dolfinx.fem.DirichletBC] = []

        self.state = state
        self.adjoint = adjoint

        self.state_ksp_option = state_ksp_option if (state_ksp_option is not None) else self.default_ksp_option
        self.adjoint_ksp_option = adjoint_ksp_option if (adjoint_ksp_option is not None) else self.default_ksp_option

        if not self.is_linear:
            self.snes_option = snes_option if (snes_option is not None) else {}
            self.snes_criterion = snes_criterion if (snes_criterion is not None) else {}

        # ------ state problem
        self.state_eq_form: ufl.Form = None
        self.state_lhs_form: ufl.Form = None
        self.state_rhs_form: ufl.Form = None
        self.state_lhs_dolfin: dolfinx.fem.Form = None
        self.state_rhs_dolfin: dolfinx.fem.Form = None

        # ------ adjoint problem
        self.adjoint_eq_form: ufl.Form = None
        self.adjoint_lhs_form: ufl.Form = None
        self.adjoint_rhs_form: ufl.Form = None
        self.adjoint_lhs_dolfin: dolfinx.fem.Form = None
        self.adjoint_rhs_dolfin: dolfinx.fem.Form = None

    def set_state_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: Union[ufl.Form, float]):
        self.state_eq_form = eqs_form
        self.state_lhs_form = lhs
        self.state_rhs_form = rhs
        self.state_lhs_dolfin = dolfinx.fem.form(lhs)
        if self.is_linear:
            self.state_rhs_dolfin = dolfinx.fem.form(rhs)

    def set_adjoint_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: ufl.Form):
        self.adjoint_eq_form = eqs_form
        self.adjoint_lhs_form = lhs
        self.adjoint_rhs_form = rhs
        self.adjoint_lhs_dolfin = dolfinx.fem.form(lhs)
        self.adjoint_rhs_dolfin = dolfinx.fem.form(rhs)

    def add_bc(
            self,
            bc: dolfinx.fem.DirichletBC, bc_V: dolfinx.fem.FunctionSpace,
            bc_dofs: np.ndarray, value_type: Union[float, np.ndarray, dolfinx.fem.Function]
    ):
        self.bcs.append(bc)
        self.bcs_infos.append((bc_V, bc_dofs, value_type))

    def compute_adjoint_bc(self):
        self.homogenize_bcs.clear()
        for bc_V, bc_dofs, value_type in self.bcs_infos:
            homo_bc = BoundaryUtils.create_homogenize_bc(bc_V, bc_dofs, value_type)
            self.homogenize_bcs.append(homo_bc)


class ControlDataBase(object):
    default_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }

    def __init__(
            self,
            controls: List[dolfinx.fem.Function],
            gradient_ksp_options: Dict[str, KspOption] = None,
    ):
        self.controls = controls
        self.num_controls = len(controls)

        for control in self.controls:
            control_name = control.name
            ksp_option = gradient_ksp_options.get(control_name, None)
            ksp_option = ksp_option if (ksp_option is not None) else self.default_ksp_option
            gradient_ksp_options[control_name] = ksp_option
        self.gradient_ksp_options = gradient_ksp_options

        self.bcs: Dict[str, List[dolfinx.fem.DirichletBC]] = {}
        self.bcs_infos: Dict[str, List[Tuple[
            dolfinx.fem.FunctionSpace, np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]]] = {}

        self.control_eq_forms = {}

        self.control_grads = {}
        for control in self.controls:
            self.control_grads[control.name] = dolfinx.fem.Function(
                control.function_space, name=f"{control.name}_grad"
            )

    def set_gradient_eq_form(self, name, lhs: ufl.Form, rhs: ufl.Form):
        eqs = lhs - rhs
        self.control_eq_forms[name] = {
            'gradient_eq_form': eqs,
            'gradient_lhs_form': lhs,
            'gradient_rhs_form': rhs,
            'gradient_lhs_dolfin': dolfinx.fem.form(lhs),
            'gradient_rhs_dolfin': dolfinx.fem.form(rhs),
        }

    def add_bc(
            self, name,
            bc: dolfinx.fem.DirichletBC,
            bc_V: dolfinx.fem.FunctionSpace,
            bc_dofs: np.ndarray,
            value_type: Union[float, np.ndarray, dolfinx.fem.Function]
    ):
        if name not in self.bcs.keys():
            self.bcs[name] = []
        self.bcs[name].append(bc)

        if name not in self.bcs_infos.keys():
            self.bcs_infos[name] = []
        self.bcs_infos[name].append((bc_V, bc_dofs, value_type))


class ShapeDataBase(object):
    default_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }

    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            gradient_ksp_option: Dict = None,
    ):
        self.domain = domain
        self.gradient_ksp_option = gradient_ksp_option if (gradient_ksp_option is not None) else self.default_ksp_option

        coordinate_space = domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.functionspace(self.domain, coordinate_space)

        self.bcs: List[dolfinx.fem.DirichletBC] = []
        self.bcs_infos: List[Tuple[
            dolfinx.fem.FunctionSpace, np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]] = []

        self.shape_grad = dolfinx.fem.Function(self.deformation_space, name='shape_grad')

        self.gradient_eq_form: ufl.Form = None
        self.gradient_lhs_form: ufl.Form = None
        self.gradient_rhs_form: ufl.Form = None
        self.gradient_lhs_dolfin: dolfinx.fem.Form = None
        self.gradient_rhs_dolfin: dolfinx.fem.Form = None

    def add_bc(
            self, bc: dolfinx.fem.DirichletBC, bc_V: dolfinx.fem.FunctionSpace,
            bc_dofs: np.ndarray, value_type: Union[float, np.ndarray, dolfinx.fem.Function]
    ):
        self.bcs.append(bc)
        self.bcs_infos.append((bc_V, bc_dofs, value_type))

    @staticmethod
    def define_fixed_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpace):
        """
        bc_V is always larger than 1, at least 2 dimension
        """
        field_size = bc_V.ufl_element().value_size()
        bc_value = np.array([0] * field_size)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_V)
        return bc, bc_V, bc_dofs, bc_value

    @staticmethod
    def define_fixed_x_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpace):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(0)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    @staticmethod
    def define_fixed_y_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpace):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(1)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    @staticmethod
    def define_fixed_z_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpace):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(2)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    def set_gradient_eq_form(self, lhs: ufl.Form, rhs: ufl.Form):
        self.gradient_eq_form = lhs - rhs
        self.gradient_lhs_form = lhs
        self.gradient_rhs_form = rhs
        self.gradient_lhs_dolfin = dolfinx.fem.form(lhs)
        self.gradient_rhs_dolfin = dolfinx.fem.form(rhs)


def create_state_problem(
        name: str,
        F_form: ufl.Form,
        state: Union[dolfinx.fem.Function, ufl.TrialFunction],
        adjoint: Union[dolfinx.fem.Function, ufl.TestFunction],
        is_linear: bool,
        bcs_info: List[Tuple[
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpace,
            np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]],
        state_ksp_option: KspOption = None,
        adjoint_ksp_option: KspOption = None,
        snes_option: KspOption = None,
        snes_criterion: KspOption = None
):
    problem = GovDataBase(
        name, F_form, state, adjoint,
        is_linear=is_linear,
        state_ksp_option=state_ksp_option, adjoint_ksp_option=adjoint_ksp_option,
        snes_option=snes_option, snes_criterion=snes_criterion
    )
    for bc, bc_V, bc_dofs, value_type in bcs_info:
        problem.add_bc(bc, bc_V, bc_dofs, value_type)
    return problem


def create_control_problem(
        controls: List[dolfinx.fem.Function],
        bcs_info: Dict[str, List[Tuple[
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpace,
            np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]]] = None,
        gradient_ksp_options: Dict[str, KspOption] = None,
):
    problem = ControlDataBase(controls, gradient_ksp_options)
    # for control_name in bcs_info.keys():
    #     for bc, bc_V, bc_dofs, value_type in bcs_info[control_name]:
    #         problem.add_bc(control_name, bc, bc_V, bc_dofs, value_type)
    return problem


def create_shape_problem(
        domain: dolfinx.mesh.Mesh,
        bcs_info: List[Tuple[
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpace,
            np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]],
        gradient_ksp_option: Dict = None,
):
    problem = ShapeDataBase(domain, gradient_ksp_option)
    for bc, bc_V, bc_dofs, value_type in bcs_info:
        problem.add_bc(bc, bc_V, bc_dofs, value_type)
    return problem
