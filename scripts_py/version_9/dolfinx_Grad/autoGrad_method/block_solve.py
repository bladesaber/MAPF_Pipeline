import dolfinx
import ufl
from petsc4py import PETSc

from Thirdparty.pyadjoint import pyadjoint
from Thirdparty.pyadjoint.pyadjoint import annotate_tape, get_working_tape, stop_annotating

from .type_Function import Function
from .type_Mesh import Mesh
from .type_DirichletBC import DirichletBC
from ..dolfinx_utils import BoundaryUtils, MeshUtils, AssembleUtils
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver


def solve(*args, **kwargs):
    """
    Be Careful:
    When solve linear problem, you need to provide a new Function as uh
    When solve nonlinear problem, uh you provided must be in the form(expression)

    Linear Problem sequence:
        0. func: type_Function.Function
        1. lhs: ufl.Form
        2. rhs: ufl.Form
        3. bcs: List[type_DirichletBC.DirichletBC]
        4. domain: type_Mesh.Mesh
        5. is_linear: True

    NonLinear Problem sequence:
        func: type_Function.Function
        lhs: ufl.Form
        bcs: List[type_DirichletBC.DirichletBC]
        domain: type_Mesh.Mesh
        is_linear: False
    """
    annotate = annotate_tape(kwargs)

    is_linear = kwargs.pop('is_linear', None)
    domain: Mesh = kwargs.pop('domain', None)
    assert (is_linear is not None) and (domain is not None)

    forward_ksp_option = kwargs['forward_ksp_option']

    if annotate:
        tape = get_working_tape()
        block = SolveBlock(
            *args, domain=domain, is_linear=is_linear, **kwargs
        )
        tape.add_block(block)

    with stop_annotating():
        if is_linear:
            res_dict = LinearProblemSolver.solve_by_petsc_form(
                comm=domain.comm, uh=args[0],
                a_form=args[1], L_form=args[2], bcs=args[3],
                ksp_option=forward_ksp_option, **kwargs
            )
            output = res_dict['res']

            if kwargs.get('with_debug', False):
                print(f"[Forward Inference] max_error:{res_dict['max_error']:.5f} cost_time:{res_dict['cost_time']:.5f}")

        else:
            is_valid = False
            exp_form: ufl.form.Form = args[1]
            for cof in exp_form.coefficients():
                if cof == args[0]:
                    is_valid = True
            assert is_valid

            res_dict = NonLinearProblemSolver.solve_by_dolfinx(
                args[1], args[0], args[2], domain.comm, forward_ksp_option, **kwargs
            )
            assert res_dict['is_converge']
            output = res_dict['res']

    if annotate:
        block_variable = args[0].create_block_variable()
        block.add_output(block_variable)

    return output


class SolveBlock(pyadjoint.Block):
    """
    每个Block代表一个Form，overType代表Form的符号表述，BlockVariable储存符号表述的输入输出变化
    """

    def __init__(
            self, *args,
            domain, is_linear,
            tlm_ksp_option,
            adj_ksp_option,
            forward_ksp_option,
            **kwargs
    ):
        super(SolveBlock, self).__init__()

        self.is_linear = is_linear
        self.domain: Mesh = domain
        self.add_dependency(self.domain)

        self._init_dependencies(*args)

        self.linear_system_solver_setting = kwargs.get("linear_system_solver_setting", {})
        self.linear_variational_solver_setting = kwargs.get('petsc_options', {})
        self.nonlinear_variational_solver_setting = kwargs.get('nonlinear_variational_solver_setting', {})

        self.tlm_ksp_option = tlm_ksp_option
        self.adj_ksp_option = adj_ksp_option
        self.forward_ksp_option = forward_ksp_option

        self.tlm_with_debug = kwargs.get('tlm_with_debug', False)
        self.adj_with_debug = kwargs.get('adj_with_debug', False)
        self.recompute_with_debug = kwargs.get('recompute_with_debug', False)

    def __str__(self):
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def _init_dependencies(self, *args):
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
        if self.is_linear:
            # lhs包含TrialFunction，action会使用Function替换TrialFunction
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
        u: dolfinx.fem.Function = output_var.output
        dFdu = ufl.derivative(
            F_form, output_var.saved_output, ufl.TrialFunction(u.function_space)  # why use TrialFunction here
            # F_form, output_var.saved_output, ufl.TestFunction(u.function_space)
        )
        r = {'form': F_form, 'dFdu': dFdu}
        return r

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared: dict = None):
        """
        F(u, v)=0 --> dot(grad(F, u), variation_u) + dot(grad(F, v), variation_v) = 0
                  --> dot(grad(F, u), variation_u) = dot(grad(-F, v), variation_v)
        """
        F_form: ufl.form.Form = prepared["form"]

        bcs = []
        dFdm: ufl.form.Form = 0.0
        dFdm_shape = 0.

        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, DirichletBC):
                if tlm_value is None:
                    new_bc = BoundaryUtils.create_homogenize_bc(
                        c.block.function_space, c.block.dofs, self.domain.topology.dim
                    )
                    bcs.append(new_bc)
                else:
                    raise NotImplementedError("No Method")

            if tlm_value is None:
                continue

            if c == self.func:
                continue

            if isinstance(c, dolfinx.mesh.Mesh):
                x = MeshUtils.define_coordinate(c)
                shape_form = ufl.derivative(-F_form, x, tlm_value)
                dFdm_shape += shape_form

            elif isinstance(c, dolfinx.fem.Function):
                dFdm += ufl.derivative(-F_form, c_rep, tlm_value)

        dFdu: ufl.form.Form = prepared["dFdu"]
        dFdu = dolfinx.fem.form(dFdu)
        a_mat = AssembleUtils.assemble_mat(dFdu, bcs)

        if not isinstance(dFdm_shape, float):
            dFdm_shape = dolfinx.fem.form(dFdm_shape)
            b_vec_shape = AssembleUtils.assemble_vec(dFdm_shape)
        else:
            b_vec_shape = dFdm_shape

        dFdm = dolfinx.fem.form(dFdm)
        b_vec = AssembleUtils.assemble_vec(dFdm)
        b_vec += b_vec_shape
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, dFdu, clean_vec=False)

        solver = LinearProblemSolver.create_petsc_solver(self.domain.comm, self.tlm_ksp_option, a_mat)
        res_dict = LinearProblemSolver.solve_by_petsc(b_vec, solver, a_mat, with_debug=self.tlm_with_debug)
        if self.tlm_with_debug:
            print(f"[Alm Inference] max_error:{res_dict['max_error']:.5f} cost_time:{res_dict['cost_time']:.5f}")

        solver.destroy()
        a_mat.destroy()
        b_vec.destroy()

        V = self.get_outputs()[idx].output.function_space
        du = dolfinx.fem.Function(V)
        du.vector.aypx(0.0, res_dict['res'])

        return du

    def evaluate_tlm(self, markings=False):
        super().evaluate_tlm(markings)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        # Step 1. Homogenize Boundary
        bcs = []
        for c in self.bcs:
            new_bc = BoundaryUtils.create_homogenize_bc(
                c.block.function_space, c.block.dofs,
                value_type=c.block.original_value
            )
            bcs.append(new_bc)

        # Step 2. Compute the form of variation dFdu and assemble lhs matrix
        output_var = self.get_outputs()[0]
        u = output_var.output

        F_form = self.create_F_form()
        dFdu = ufl.derivative(F_form, output_var.saved_output, ufl.TrialFunction(u.function_space))
        dFdu_adjoint = ufl.adjoint(dFdu)
        a_mat: PETSc.Mat = AssembleUtils.assemble_mat(dolfinx.fem.form(dFdu_adjoint), bcs)

        # Step 3. Apply boundary to adjoint value
        dJdu_vec: PETSc.Vec = adj_inputs[0]
        dJdu_copy_vec = dJdu_vec.copy()
        BoundaryUtils.apply_boundary_to_vec(dJdu_copy_vec, bcs, dolfinx.fem.form(dFdu), clean_vec=False)

        # Step Solve the linear system
        solver = LinearProblemSolver.create_petsc_solver(self.domain.comm, self.adj_ksp_option, a_mat)
        res_dict = LinearProblemSolver.solve_by_petsc(dJdu_copy_vec, solver, a_mat, with_debug=self.adj_with_debug)
        if self.adj_with_debug:
            print(f"[Adj Inference] max_error:{res_dict['max_error']:.5f} cost_time:{res_dict['cost_time']:.5f}")

        V = self.func.function_space
        lmd = dolfinx.fem.Function(V)
        lmd.vector.aypx(0.0, res_dict['res'])

        r = {'form': F_form, 'adj_sol': lmd}
        return r

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        F_form: ufl.form.Form = prepared["form"]
        lmd: dolfinx.fem.Function = prepared["adj_sol"]

        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, dolfinx.fem.Function):
            trial_function = ufl.TrialFunction(c.function_space)  # why TrialFunction
            dFdm = -ufl.derivative(F_form, c_rep, trial_function)
            dFdm_adj = ufl.adjoint(dFdm)
            dFdm_adj = dFdm_adj * lmd
            dFdm_vec = AssembleUtils.assemble_vec(dolfinx.fem.form(dFdm_adj))

        elif isinstance(c, dolfinx.mesh.Mesh):
            X = MeshUtils.define_coordinate(c_rep)
            coordinate_space = X.ufl_domain().ufl_coordinate_element()
            function_space = dolfinx.fem.FunctionSpace(c, coordinate_space)
            du = ufl.TestFunction(function_space)

            # -----------------------------
            # TODO why we need action operator first, may be method1 and method2 are the same
            # ------ method 1
            F_form_tmp = ufl.action(F_form, lmd)
            dFdm = ufl.derivative(-F_form_tmp, X, du)
            dFdm_vec = AssembleUtils.assemble_vec(dolfinx.fem.form(dFdm))

            # ------ method 2
            # dFdm = -ufl.derivative(F_form, X, du)
            # dFdm_adj = ufl.adjoint(dFdm)
            # dFdm_adj = dFdm_adj * lmd
            # dFdm_vec = SolverUtils.assemble_vec(dolfinx.fem.form(dFdm_adj))
            # -----------------------------

        else:
            raise NotImplementedError

        return dFdm_vec

    def evaluate_adj(self, markings=False):
        super().evaluate_adj(markings)

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

        ret = {'lhs': lhs, 'rhs': rhs, 'bcs': bcs, 'func': guess_func}
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
            res_dict = LinearProblemSolver.solve_by_petsc_form(
                self.domain.comm, func, lhs, rhs, bcs, self.forward_ksp_option,
                with_debug=self.recompute_with_debug
            )
            func = res_dict['res']

            if self.recompute_with_debug:
                print(f"[Forward Inference] max_error:{res_dict['max_error']:.5f} cost_time:{res_dict['cost_time']:.5f}")

        else:
            res_dict = NonLinearProblemSolver.solve_by_dolfinx(
                lhs, func, bcs, comm=self.domain.comm, ksp_option=self.forward_ksp_option,
                with_debug=self.recompute_with_debug
            )
            assert res_dict['is_converge']
            func = res_dict['res']
        return func
