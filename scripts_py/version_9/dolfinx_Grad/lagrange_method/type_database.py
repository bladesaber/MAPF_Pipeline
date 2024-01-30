import numpy as np
import ufl
import dolfinx
from typing import List, Union, Dict, Tuple

from ..dolfinx_utils import BoundaryUtils

KspOption = Dict[str, Union[int, float, str, None]]


class GovDataBase(object):
    def __init__(
            self,
            name: str,
            F_form: ufl.Form,
            state: Union[dolfinx.fem.Function, ufl.TrialFunction],
            adjoint: Union[dolfinx.fem.Function, ufl.TestFunction],
            is_linear: bool,
            state_ksp_option: KspOption = None,
            adjoint_ksp_option: KspOption = None,
    ):
        self.name = name

        self.F_form = F_form
        self.is_linear = is_linear
        self.bcs: List[dolfinx.fem.DirichletBC] = []
        self.bcs_infos: List = []
        self.homogenize_bcs: List[dolfinx.fem.DirichletBC] = []

        self.state = state
        self.adjoint = adjoint

        self.state_ksp_option = self.parse_ksp_option(state_ksp_option)
        self.adjoint_ksp_option = self.parse_ksp_option(adjoint_ksp_option)

    def set_state_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: ufl.Form):
        self.state_eq_form = eqs_form
        self.state_eq_form_lhs = lhs
        self.state_eq_form_rhs = rhs

        self.state_eq_dolfinx_form_lhs = dolfinx.fem.form(lhs)
        if self.is_linear:
            self.state_eq_dolfinx_form_rhs = dolfinx.fem.form(rhs)

    def set_adjoint_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: ufl.Form):
        self.adjoint_eq_form = eqs_form
        self.adjoint_eq_form_lhs = lhs
        self.adjoint_eq_form_rhs = rhs

        self.adjoint_eq_dolfinx_form_lhs = dolfinx.fem.form(lhs)
        self.adjoint_eq_dolfinx_form_rhs = dolfinx.fem.form(rhs)

    def add_bc(
            self, bc: dolfinx.fem.DirichletBC, bc_V: dolfinx.fem.FunctionSpaceBase,
            bc_dofs: np.ndarray, value_type: Union[float, np.ndarray, dolfinx.fem.Function]
    ):
        self.bcs.append(bc)
        self.bcs_infos.append((bc_V, bc_dofs, value_type))

    def compute_adjoint_bc(self):
        self.homogenize_bcs.clear()
        for bc_V, bc_dofs, value_type in self.bcs_infos:
            homo_bc = BoundaryUtils.create_homogenize_bc(bc_V, bc_dofs, value_type)
            self.homogenize_bcs.append(homo_bc)

    @staticmethod
    def parse_ksp_option(ksp_option):
        if ksp_option is None:
            return {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
            }
        return ksp_option


class ControlDataBase(object):
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
            ksp_option = GovDataBase.parse_ksp_option(ksp_option)
            gradient_ksp_options[control_name] = ksp_option
        self.gradient_ksp_options = gradient_ksp_options

        self.bcs: Dict[str, List[dolfinx.fem.DirichletBC]] = {}
        self.bcs_infos: Dict[str, List] = {}

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
            'gradient_eq_form_lhs': lhs,
            'gradient_eq_form_rhs': rhs,
            'gradient_eq_dolfinx_form_lhs': dolfinx.fem.form(lhs),
            'gradient_eq_dolfinx_form_rhs': dolfinx.fem.form(rhs),
        }

    def add_bc(
            self,
            name,
            bc: dolfinx.fem.DirichletBC,
            bc_V: dolfinx.fem.FunctionSpaceBase,
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
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            lambda_lame: float = 0.0,
            damping_factor: float = 0.0,
            gradient_ksp_option: Dict = None,
    ):
        self.domain = domain
        self.gradient_ksp_option = GovDataBase.parse_ksp_option(gradient_ksp_option)

        coordinate_space = domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.FunctionSpace(self.domain, coordinate_space)
        # self.deformation_space = dolfinx.fem.VectorFunctionSpace(self.domain, ("CG", 1))  # replace

        self.bcs: List[dolfinx.fem.DirichletBC] = []
        self.bcs_infos: List = []

        self.lambda_lame = lambda_lame
        self.damping_factor = damping_factor

        self.shape_grad = dolfinx.fem.Function(self.deformation_space, name='shape_grad')

    def add_bc(
            self, bc: dolfinx.fem.DirichletBC, bc_V: dolfinx.fem.FunctionSpaceBase,
            bc_dofs: np.ndarray, value_type: Union[float, np.ndarray, dolfinx.fem.Function]
    ):
        self.bcs.append(bc)
        self.bcs_infos.append((bc_V, bc_dofs, value_type))

    @staticmethod
    def define_fixed_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpaceBase):
        """
        bc_V is always larger than 1, at least 2 dimension
        """
        field_size = bc_V.ufl_element().value_size()
        bc_value = np.array([0] * field_size)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_V)
        return bc, bc_V, bc_dofs, bc_value

    @staticmethod
    def define_fixed_x_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpaceBase):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(0)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    @staticmethod
    def define_fixed_y_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpaceBase):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(1)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    @staticmethod
    def define_fixed_z_boundary(bc_dofs: np.ndarray, bc_V: dolfinx.fem.FunctionSpaceBase):
        bc_value = 0.0
        bc_sub_space = bc_V.sub(2)
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, bc_sub_space)
        return bc, bc_sub_space, bc_dofs, bc_value

    def set_gradient_eq_form(self, lhs: ufl.Form, rhs: ufl.Form):
        self.gradient_eq_form = lhs - rhs
        self.gradient_eq_form_lhs = lhs
        self.gradient_eq_form_rhs = rhs

        self.gradient_eq_dolfinx_form_lhs = dolfinx.fem.form(lhs)
        self.gradient_eq_dolfinx_form_rhs = dolfinx.fem.form(rhs)


def create_state_problem(
        name: str,
        F_form: ufl.Form,
        state: Union[dolfinx.fem.Function, ufl.TrialFunction],
        adjoint: Union[dolfinx.fem.Function, ufl.TestFunction],
        is_linear: bool,
        bcs_info: List[Tuple[
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpaceBase,
            np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]],
        state_ksp_option: KspOption = None,
        adjoint_ksp_option: KspOption = None
):
    problem = GovDataBase(name, F_form, state, adjoint, is_linear, state_ksp_option, adjoint_ksp_option)
    for bc, bc_V, bc_dofs, value_type in bcs_info:
        problem.add_bc(bc, bc_V, bc_dofs, value_type)

    return problem


def create_control_problem(
        controls: List[dolfinx.fem.Function],
        bcs_info: Dict[str, List[Tuple[
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpaceBase,
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
            dolfinx.fem.DirichletBC, dolfinx.fem.FunctionSpaceBase,
            np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        ]],
        lambda_lame: float = 0.0,
        damping_factor: float = 0.0,
        gradient_ksp_option: Dict = None,
):
    problem = ShapeDataBase(domain, lambda_lame, damping_factor, gradient_ksp_option)
    for bc, bc_V, bc_dofs, value_type in bcs_info:
        problem.add_bc(bc, bc_V, bc_dofs, value_type)
    return problem
