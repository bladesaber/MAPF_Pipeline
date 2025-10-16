from typing import Union, Callable

import ufl
import dolfinx

import numpy as np

from scripts_py.dolfinx_utils import BoundaryUtils


class BoundaryCell(object):
    def __init__(
        self,
        name: str,
        bc: dolfinx.fem.DirichletBC,
        bc_func_space: dolfinx.fem.FunctionSpace,
        value_func_space: dolfinx.fem.FunctionSpace,
        bc_dofs: np.ndarray,
        value_type: Union[float, np.ndarray, dolfinx.fem.Function, dolfinx.fem.Constant, Callable],
        is_mixed_element: bool
    ):
        self.name = name
        self.bc = bc
        self.bc_func_space = bc_func_space
        self.value_func_space = value_func_space
        self.bc_dofs = bc_dofs
        self.value_type = value_type
        self.is_mixed_element = is_mixed_element
        self.homogenize_bc: dolfinx.fem.DirichletBC

    @staticmethod
    def define_bry_cell(
        name: str,
        bc_func_space: dolfinx.fem.FunctionSpace,
        value_func_space: dolfinx.fem.FunctionSpace,
        dofs: np.ndarray,
        bc_value: Union[Callable, float, np.ndarray, dolfinx.fem.Function, dolfinx.fem.Constant],
        is_mixed_element: bool
    ):
        bc = BoundaryUtils.define_dirichlet_bc(
            bc_func_space=bc_func_space, 
            value_func_space=value_func_space, 
            dofs=dofs, 
            bc_value=bc_value, 
            is_mixed_element=is_mixed_element
        )
        return BoundaryCell(
            name=name, 
            bc=bc, 
            bc_func_space=bc_func_space, 
            value_func_space=value_func_space, 
            bc_dofs=dofs, 
            value_type=bc_value, 
            is_mixed_element=is_mixed_element
        )
    

class StateProblemDataBase(object):
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
        state: Union[dolfinx.fem.Function, ufl.TrialFunction],  # type: ignore
        adjoint: Union[dolfinx.fem.Function, ufl.TestFunction],  # type: ignore
        is_linear: bool,
        state_ksp_option: dict | None = None,
        adjoint_ksp_option: dict | None = None,
        snes_option: dict | None = None,
        snes_criterion: dict | None = None,
    ):
        self.name = name
        self.bc_cells: dict[str, BoundaryCell] = {}

        self.F_form = F_form
        self.is_linear = is_linear

        self.state = state
        self.adjoint = adjoint
        self.state_ksp_option = state_ksp_option if (state_ksp_option is not None) else self.default_ksp_option
        self.adjoint_ksp_option = adjoint_ksp_option if (adjoint_ksp_option is not None) else self.default_ksp_option

        if not self.is_linear:
            self.snes_option = snes_option if (snes_option is not None) else {}
            self.snes_criterion = snes_criterion if (snes_criterion is not None) else {}

        # ------ state problem
        self.state_eq_form: ufl.Form
        self.state_lhs_form: ufl.Form
        self.state_rhs_form: ufl.Form | float
        self.state_lhs_dolfin: dolfinx.fem.Form
        self.state_rhs_dolfin: dolfinx.fem.Form

        # ------ adjoint problem
        self.adjoint_eq_form: ufl.Form
        self.adjoint_lhs_form: ufl.Form
        self.adjoint_rhs_form: ufl.Form | float
        self.adjoint_lhs_dolfin: dolfinx.fem.Form
        self.adjoint_rhs_dolfin: dolfinx.fem.Form

    def set_state_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: Union[ufl.Form, float]):
        self.state_eq_form = eqs_form
        self.state_lhs_form = lhs
        self.state_rhs_form = rhs
        self.state_lhs_dolfin = dolfinx.fem.form(lhs)  # type: ignore
        if self.is_linear:
            self.state_rhs_dolfin = dolfinx.fem.form(rhs)  # type: ignore

    def set_adjoint_eq_form(self, eqs_form: ufl.Form, lhs: ufl.Form, rhs: ufl.Form):
        self.adjoint_eq_form = eqs_form
        self.adjoint_lhs_form = lhs
        self.adjoint_rhs_form = rhs
        self.adjoint_lhs_dolfin = dolfinx.fem.form(lhs)  # type: ignore
        self.adjoint_rhs_dolfin = dolfinx.fem.form(rhs)  # type: ignore

    def add_bc(self, bc_cell: BoundaryCell):
        self.bc_cells[bc_cell.name] = bc_cell

    def compute_adjoint_bc(self):
        for bc_cell in self.bc_cells.values():
            bc_cell.homogenize_bc = BoundaryUtils.create_homogenize_bc(
                bc_func_space=bc_cell.bc_func_space, 
                value_func_space=bc_cell.value_func_space, 
                dofs=bc_cell.bc_dofs, 
                is_mixed_element=bc_cell.is_mixed_element,
                value_type=bc_cell.value_type
            )


class ControlProblemDataBase(object):
    default_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }
    
    class ControlPar(object):
        def __init__(self, name: str, control: dolfinx.fem.Function, ksp_option: dict):
            self.name = name
            self.bc_cells: dict[str, BoundaryCell] = {}
            self.control = control
            self.ksp_option = ksp_option
            self.grad: dolfinx.fem.Function = dolfinx.fem.Function(self.control.function_space, name=f"{self.name}_grad")  # type: ignore
            
            self.gradient_eq_form: ufl.Form
            self.gradient_lhs_form: ufl.Form
            self.gradient_rhs_form: ufl.Form
            self.gradient_lhs_dolfin: dolfinx.fem.Form
            self.gradient_rhs_dolfin: dolfinx.fem.Form
            
    def __init__(self, controls: list[dolfinx.fem.Function], gradient_ksp_options: dict = {}):
        self.controls = controls
        self.num_controls = len(controls)
        self.control_pars: dict[str, ControlProblemDataBase.ControlPar] = {}

        for control in self.controls:
            self.control_pars[control.name] = ControlProblemDataBase.ControlPar(
                name=control.name, 
                control=control, 
                ksp_option=gradient_ksp_options.get(control.name, self.default_ksp_option)
            )

    def set_gradient_eq_form(self, name: str|dolfinx.fem.Function, lhs: ufl.Form, rhs: ufl.Form):
        if isinstance(name, dolfinx.fem.Function):
            name = name.name
        if name not in self.control_pars.keys():
            raise ValueError(f"Control {name} not found")
        self.control_pars[name].gradient_eq_form = lhs - rhs
        self.control_pars[name].gradient_lhs_form = lhs
        self.control_pars[name].gradient_rhs_form = rhs
        self.control_pars[name].gradient_lhs_dolfin = dolfinx.fem.form(lhs)  # type: ignore
        self.control_pars[name].gradient_rhs_dolfin = dolfinx.fem.form(rhs)  # type: ignore

    def add_bc(self, name: str|dolfinx.fem.Function, bc_cell: BoundaryCell):
        if isinstance(name, dolfinx.fem.Function):
            name = name.name
        if name not in self.control_pars.keys():
            raise ValueError(f"Control {name} not found")
        self.control_pars[name].bc_cells[bc_cell.name] = bc_cell


class ShapeProblemDataBase(object):
    default_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }

    def __init__(self, domain: dolfinx.mesh.Mesh, gradient_ksp_option: dict | None = None):
        self.domain = domain
        self.gradient_ksp_option = gradient_ksp_option if (gradient_ksp_option is not None) else self.default_ksp_option
        self.bc_cells: dict[str, BoundaryCell] = {}

        coordinate_space = domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.functionspace(self.domain, coordinate_space)
        self.shape_grad: dolfinx.fem.Function = dolfinx.fem.Function(self.deformation_space, name='shape_grad')  # type: ignore

        # self.bcs: List[dolfinx.fem.DirichletBC] = []
        # self.bcs_infos: List[Tuple[
        #     dolfinx.fem.FunctionSpace, np.ndarray, Union[float, np.ndarray, dolfinx.fem.Function]
        # ]] = []

        self.gradient_eq_form: ufl.Form
        self.gradient_lhs_form: ufl.Form
        self.gradient_rhs_form: ufl.Form
        self.gradient_lhs_dolfin: dolfinx.fem.Form
        self.gradient_rhs_dolfin: dolfinx.fem.Form

    def add_bc(self, bc_cell: BoundaryCell):
        self.bc_cells[bc_cell.name] = bc_cell

    @staticmethod
    def define_fixed_boundary(bc_dofs: np.ndarray, bc_func_space: dolfinx.fem.FunctionSpace) -> BoundaryCell:
        field_size = bc_func_space.ufl_element().value_size()  # 2 for 2D, 3 for 3D
        bc_value = np.array([0] * field_size)
        func_space = bc_func_space
        return BoundaryCell(
            name='fixed_xyz',
            bc=dolfinx.fem.dirichletbc(bc_value, bc_dofs, func_space),  # type: ignore
            bc_func_space=func_space,
            bc_dofs=bc_dofs,
            value_type=bc_value
        )

    @staticmethod
    def define_fixed_x_boundary(bc_dofs: np.ndarray, bc_func_space: dolfinx.fem.FunctionSpace) -> BoundaryCell:
        func_space = bc_func_space.sub(0)
        return BoundaryCell(
            name='fixed_x',
            bc=dolfinx.fem.dirichletbc(0.0, bc_dofs, func_space),  # type: ignore
            bc_func_space=func_space,
            bc_dofs=bc_dofs,
            value_type=0.0
        )

    @staticmethod
    def define_fixed_y_boundary(bc_dofs: np.ndarray, bc_func_space: dolfinx.fem.FunctionSpace) -> BoundaryCell:
        func_space = bc_func_space.sub(1)        
        return BoundaryCell(
            name='fixed_y',
            bc=dolfinx.fem.dirichletbc(0.0, bc_dofs, func_space),  # type: ignore
            bc_func_space=func_space,
            bc_dofs=bc_dofs,
            value_type=0.0
        )
        
    @staticmethod
    def define_fixed_z_boundary(bc_dofs: np.ndarray, bc_func_space: dolfinx.fem.FunctionSpace) -> BoundaryCell:
        func_space = bc_func_space.sub(2)        
        return BoundaryCell(
            name='fixed_z',
            bc=dolfinx.fem.dirichletbc(0.0, bc_dofs, func_space),  # type: ignore
            bc_func_space=func_space,
            bc_dofs=bc_dofs,
            value_type=0.0
        )

    def set_gradient_eq_form(self, lhs: ufl.Form, rhs: ufl.Form):
        self.gradient_eq_form = lhs - rhs
        self.gradient_lhs_form = lhs
        self.gradient_rhs_form = rhs
        self.gradient_lhs_dolfin = dolfinx.fem.form(lhs)  # type: ignore
        self.gradient_rhs_dolfin = dolfinx.fem.form(rhs)  # type: ignore

