import numpy as np
import dolfinx
import ufl
from typing import Union
from petsc4py import PETSc

from Thirdparty.pyadjoint import pyadjoint
from Thirdparty.pyadjoint.pyadjoint import annotate_tape, get_working_tape, stop_annotating

from .backend_dolfinx import SolverUtils, BoundaryUtils
from .type_Function import Function
from .type_Mesh import Mesh
from .type_DirichletBC import DirichletBC
from .petsc_utils import PETScUtils


def solve(*args, **kwargs):
    """
    Be Careful:
    When solve linear problem, you need to provide a new Function as uh
    When solve nonlinear problem, uh you provided must be in the form(expression)
    """
    annotate = annotate_tape(kwargs)
    sb_kwargs = SolveBlock.pop_kwargs(kwargs)
    sb_kwargs.update(kwargs)

    if annotate:
        tape = get_working_tape()
        block = SolveBlock(*args, **sb_kwargs)
        tape.add_block(block)

    with stop_annotating():
        if sb_kwargs['is_linear']:
            output = SolverUtils.solve_linear_variational_problem(
                *args, petsc_options=sb_kwargs.pop('petsc_options', {}),
                **kwargs
            )
        else:
            is_valid = False
            exp_form: ufl.form.Form = args[1]
            for cof in exp_form.coefficients():
                if cof == args[0]:
                    is_valid = True
            assert is_valid

            run_times, is_converged, output = SolverUtils.solve_nonlinear_variational_problem(
                *args, rtol=sb_kwargs.pop('petsc_options', 1e-6),
                petsc_options=sb_kwargs.pop('petsc_options', {})
            )
            assert is_converged

    if annotate:
        block_variable = args[0].create_block_variable()
        block.add_output(block_variable)

    return output


class SolveBlock(pyadjoint.Block):
    """
    每个Block代表一个Form，overType代表Form的符号表述，BlockVariable储存符号表述的输入输出变化
    """
    pop_kwargs_keys = ["domain", "is_linear", "rtol", "petsc_options"]

    def __init__(self, *args, **kwargs):
        super(SolveBlock, self).__init__()

        self.is_linear = kwargs['is_linear']
        self.domain: Mesh = kwargs['domain']
        self.add_dependency(self.domain)

        self._init_dependencies(*args, **kwargs)

        self.linear_system_solver_setting = kwargs.pop("linear_system_solver_setting", {})

    def __str__(self):
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def _init_dependencies(self, *args, **kwargs):
        self.func: dolfinx.fem.Function = args[0]
        if self.is_linear:
            self.lhs: ufl.form.Form = args[1]
            self.rhs: ufl.form.Form = args[2]
            for c in self.rhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
            self.bcs = args[3]

        else:
            self.lhs: ufl.form.Form = args[1]
            self.rhs: ufl.form.Form = 0.0
            self.bcs = args[2]

        for bc in self.bcs:
            self.add_dependency(bc, no_duplicates=True)

        for c in self.lhs.coefficients():
            self.add_dependency(c, no_duplicates=True)

    def create_F_form(self):
        """
        在 is_linear 下，lhs 包含 TrialFunction 以及 TestFunction，action会替换 TrialFunction
        """
        if self.is_linear:
            tmp_u = Function(self.func.function_space)
            F_form = ufl.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replace_map = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in F_form.coefficients():
                replace_map[coeff] = block_variable.saved_output

        replace_map[tmp_u] = self.get_outputs()[0].saved_output  # important here
        return ufl.replace(F_form, replace_map)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        F_form = self.create_F_form()

        # 使用Argument(TrialFunction或TestFunction)作为扰动方向，即扰动方向就为未知值
        # 获得 dFdu 的变分的 Form
        output_var = self.get_outputs()[0]
        u: Function = output_var.output
        dFdu = ufl.derivative(
            F_form, output_var.saved_output, ufl.TrialFunction(u.function_space)
        )
        r = {'form': F_form, 'dFdu': dFdu}
        return r

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared: dict = None):
        F_form: ufl.form.Form = prepared["form"]

        bcs = []
        dFdm: ufl.form.Form = 0.0
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, DirichletBC):
                if tlm_value is None:
                    # TODO why need homogenize
                    new_bc = BoundaryUtils.create_bc(
                        c.block.original_value,
                        c.block.dofs, c.block.function_space, homogenize=True
                    )
                    bcs.append(new_bc)
                else:
                    bcs.append(tlm_value)
                    continue

            elif isinstance(c, Mesh):
                X = ufl.SpatialCoordinate(c)
                c_rep = X

            if tlm_value is None:
                continue

            if c == self.func:
                continue

            if isinstance(c, Mesh):
                raise NotImplementedError

            elif isinstance(c, Function):
                dFdm += ufl.derivative(-F_form, c_rep, tlm_value)

        dFdu: ufl.form.Form = prepared["dFdu"]
        A_mat = SolverUtils.assemble_mat(dolfinx.fem.form(dFdu), bcs)

        dFdm_vec = SolverUtils.assemble_vec(dolfinx.fem.form(dFdm))
        SolverUtils.apply_boundary_to_vec(dFdm_vec, bcs, dolfinx.fem.form(dFdu), clean_vec=False)

        V = self.get_outputs()[idx].output.function_space
        du = dolfinx.fem.Function(V)

        # # ------ Debug
        # size = PETScUtils.get_Size(A_mat)[0]
        # A_mat_np = PETScUtils.getArray_from_Mat(range(size), range(size), A_mat)
        # print(f"Is A_mat Inf: {np.any(np.isinf(A_mat_np))} or Nan: {np.any(np.isnan(A_mat_np))}")
        # b_vec_np = PETScUtils.getArray_from_Vec(dFdm_vec)
        # print(f"Is b_vec Inf: {np.any(np.isinf(b_vec_np))} or Nan: {np.any(np.isnan(b_vec_np))}")
        # PETScUtils.save_data(A_mat, '/home/admin123456/Desktop/work/test_code/A_mat.dat')
        # PETScUtils.save_data(dFdm_vec, '/home/admin123456/Desktop/work/test_code/b_vec.dat')
        # # -----------------------

        SolverUtils.solve_linear_system_problem(
            du, dFdm_vec, A_mat, self.domain.comm,
            solver=None,
            solver_setting=self.linear_system_solver_setting
        )

        return du

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        # Step 1. Homogenize Boundary
        bcs = []
        for bc in self.bcs:
            # TODO why need homogenize
            BoundaryUtils.create_bc(
                bc.block.original_value, bc.block.dofs, bc.block.function_space, homogenize=True
            )
            bcs.append(bc)

        # Step 2. Compute the form of variation dFdu and assemble lhs matrix
        output_var = self.get_outputs()[0]
        u = output_var.output

        # # ------ Debug
        # print(f"Is A_mat Inf: {np.any(np.isinf(u.x.array))} or Nan: {np.any(np.isnan(u.x.array))}")
        # # ------

        F_form = self.create_F_form()
        dFdu = ufl.derivative(F_form, output_var.saved_output, ufl.TrialFunction(u.function_space))
        dFdu_adjoint = ufl.adjoint(dFdu)
        A_mat: PETSc.Mat = SolverUtils.assemble_mat(dolfinx.fem.form(dFdu_adjoint), bcs)

        # Step 3. Apply boundary to adjoint value
        dJdu_vec: PETSc.Vec = adj_inputs[0]
        dJdu_copy_vec = dJdu_vec.copy()
        SolverUtils.apply_boundary_to_vec(dJdu_copy_vec, bcs, dolfinx.fem.form(dFdu), clean_vec=False)

        # Step Solve the linear system
        if isinstance(self.func, dolfinx.fem.Function):
            V = self.func.function_space
        else:
            V = self.func.ufl_function_space
        lmd = dolfinx.fem.Function(V)

        # # ------ Debug
        # size = PETScUtils.get_Size(A_mat)[0]
        # A_mat_np = PETScUtils.getArray_from_Mat(range(size), range(size), A_mat)
        # print(f"Is A_mat Inf: {np.any(np.isinf(A_mat_np))} or Nan: {np.any(np.isnan(A_mat_np))}")
        # b_vec_np = PETScUtils.getArray_from_Vec(dJdu_copy_vec)
        # print(f"Is b_vec Inf: {np.any(np.isinf(b_vec_np))} or Nan: {np.any(np.isnan(b_vec_np))}")
        # PETScUtils.save_data(A_mat, '/home/admin123456/Desktop/work/test_code/A_mat.dat')
        # PETScUtils.save_data(dJdu_copy_vec, '/home/admin123456/Desktop/work/test_code/b_vec.dat')
        # # -----------------------

        SolverUtils.solve_linear_system_problem(
            lmd, dJdu_copy_vec, A_mat, self.domain.comm,
            solver=None, solver_setting=self.linear_system_solver_setting
        )

        r = {'form': F_form, 'adj_sol': lmd}
        return r

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        F_form: ufl.form.Form = prepared["form"]
        lmd: Function = prepared["adj_sol"]

        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, Function):
            trial_function = ufl.TrialFunction(c.function_space)
            dFdm = -ufl.derivative(F_form, c_rep, trial_function)
            dFdm_adj = ufl.adjoint(dFdm)
            dFdm_adj = dFdm_adj * lmd
            dFdm_vec = SolverUtils.assemble_vec(dolfinx.fem.form(dFdm_adj))
        else:
            raise NotImplementedError

        return dFdm_vec

    def prepare_recompute_component(self, inputs, relevant_outputs):
        guess_func: Function = Function(self.func.function_space)

        bcs = []
        replace_map = {}
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output
            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)

            if c in self.lhs.coefficients():
                c_rep = block_variable.saved_output
                replace_map[c] = c_rep

                # important here, we need to compute a new solution
                if c == self.func:
                    # guess_func.assign(c_rep)
                    replace_map[c] = guess_func

        lhs = ufl.replace(self.lhs, replace_map)
        if self.is_linear:
            rhs = ufl.replace(self.rhs, replace_map)
        else:
            rhs = 0.0

        ret = {
            'lhs': lhs, 'rhs': rhs, 'bcs': bcs, 'func': guess_func,
        }
        return ret

    def recompute_component(self, inputs, block_variable, idx, prepared: dict):
        """
        所有的recompute都使用checkpoint，因为pyadjoint.Block.recompute中会将最新的值传递到output的checkpoint
        """
        lhs = prepared['lhs']
        rhs = prepared['rhs']
        bcs = prepared['bcs']
        func = prepared['func']

        if self.is_linear:
            SolverUtils.solve_linear_variational_problem(func, lhs, rhs, bcs)
        else:
            _, is_converged, _ = SolverUtils.solve_nonlinear_variational_problem(func, lhs, bcs)
            assert is_converged

        return func