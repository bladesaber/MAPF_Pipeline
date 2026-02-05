from petsc4py import PETSc
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import petsc as dolfinx_petsc
import ufl
from typing import Literal
from functools import partial
import pandas as pd
import pyvista

from line_profiler import profile

from scripts_py.sparse_utils import PetscUtils, ScipySparseUtils
from scripts_py.dolfinx_utils import AssembleUtils, BoundaryUtils, DolfinUtils

"""
KSP Convergence Reasons:
Reference Link: https://petsc.org/release/manualpages/KSP/KSPConvergedReason/
  /* converged */
  KSP_CONVERGED_RTOL_NORMAL               = 1,
  KSP_CONVERGED_ATOL_NORMAL               = 9,
  KSP_CONVERGED_RTOL                      = 2,
  KSP_CONVERGED_ATOL                      = 3,
  KSP_CONVERGED_ITS                       = 4,
  KSP_CONVERGED_NEG_CURVE                 = 5,
  KSP_CONVERGED_CG_NEG_CURVE_DEPRECATED   = 5,
  KSP_CONVERGED_CG_CONSTRAINED_DEPRECATED = 6,
  KSP_CONVERGED_STEP_LENGTH               = 6,
  KSP_CONVERGED_HAPPY_BREAKDOWN           = 7,
  /* diverged */
  KSP_DIVERGED_NULL                      = -2,
  KSP_DIVERGED_ITS                       = -3,
  KSP_DIVERGED_DTOL                      = -4,
  KSP_DIVERGED_BREAKDOWN                 = -5,
  KSP_DIVERGED_BREAKDOWN_BICG            = -6,
  KSP_DIVERGED_NONSYMMETRIC              = -7,
  KSP_DIVERGED_INDEFINITE_PC             = -8,
  KSP_DIVERGED_NANORINF                  = -9,
  KSP_DIVERGED_INDEFINITE_MAT            = -10,
  KSP_DIVERGED_PC_FAILED                 = -11,
  KSP_DIVERGED_PCSETUP_FAILED_DEPRECATED = -11,

  KSP_CONVERGED_ITERATING = 0

SNES Convergence Reasons:
Ref: https://petsc.org/release/manualpages/SNES/SNESConvergedReason/
  SNES_CONVERGED_FNORM_ABS           = 2, /* ||F|| < atol */
  SNES_CONVERGED_FNORM_RELATIVE      = 3, /* ||F|| < rtol*||F_initial|| */
  SNES_CONVERGED_SNORM_RELATIVE      = 4, /* Newton computed step size small; || delta x || < stol || x ||*/
  SNES_CONVERGED_ITS                 = 5, /* maximum iterations reached */
  SNES_BREAKOUT_INNER_ITER           = 6, /* Flag to break out of inner loop after checking custom convergence. */
                                          /* it is used in multi-phase flow when state changes */
  /* diverged */
  SNES_DIVERGED_FUNCTION_DOMAIN      = -1, /* the new x location passed the function is not in the domain of F */
  SNES_DIVERGED_FUNCTION_COUNT       = -2,
  SNES_DIVERGED_LINEAR_SOLVE         = -3, /* the linear solve failed */
  SNES_DIVERGED_FNORM_NAN            = -4,
  SNES_DIVERGED_MAX_IT               = -5,
  SNES_DIVERGED_LINE_SEARCH          = -6,  /* the line search failed */
  SNES_DIVERGED_INNER                = -7,  /* inner solve failed */
  SNES_DIVERGED_LOCAL_MIN            = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
  SNES_DIVERGED_DTOL                 = -9,  /* || F || > divtol*||F_initial|| */
  SNES_DIVERGED_JACOBIAN_DOMAIN      = -10, /* Jacobian calculation does not make sense */
  SNES_DIVERGED_TR_DELTA             = -11,
  SNES_CONVERGED_TR_DELTA_DEPRECATED = -11,

  SNES_CONVERGED_ITERATING = 0
"""


class LinearProblemSolver(object):
    @staticmethod
    def create_petsc_solver(
        comm=MPI.COMM_WORLD, 
        solver_setting: dict|None = None, 
        mat: PETSc.Mat|None = None,
        setup_pc: bool = False
    ):
        """find more info from: https://petsc4py.readthedocs.io/en/stable/manual/ksp/"""
        solver: PETSc.KSP = PETSc.KSP().create(comm)
        if solver_setting is None:
            return solver

        # ------- easy use
        # solver.setType(solver_setting.get("ksp_type", "preonly"))
        # solver.getPC().setType(solver_setting.get("pc_type", "lu"))
        # if "pc_factor_mat_solver_type" in solver_setting.keys():
        #     solver.getPC().setFactorSolverType(solver_setting["pc_factor_mat_solver_type"])
        # if "pc_hypre_mat_solver_type" in solver_setting.keys():
        #     solver.getPC().setHYPREType(solver_setting["pc_hypre_mat_solver_type"])
        # solver.setFromOptions()
        # ----------------------------

        opts = PETSc.Options()
        for key, value in solver_setting.items():
            opts.setValue(key, value)
        solver.setFromOptions()
        
        if mat is not None:
            solver.setOperators(mat)
        
        # TODO: I think we need setup, it is very important for static matrix
        if setup_pc:
            solver.getPC().setUp()
        
        return solver

    @staticmethod
    def log_petsc_solver(solver: PETSc.KSP):
        solver.view()

    @staticmethod
    def solve_by_np(a_mat: np.ndarray, b_vec: np.ndarray, with_residual=False) -> dict:
        res_vec = np.dot(np.linalg.inv(a_mat), b_vec)
        res_dict = {'res': res_vec, 'is_solved': True}
        if with_residual:
            res_dict.update({'max_residual': np.max(np.abs(np.dot(a_mat, res_vec) - b_vec))})
        return res_dict

    @staticmethod
    def solve_fea_scipy(
        uh: dolfinx.fem.Function,
        a_form: dolfinx.fem.Form | ufl.Form, l_form: dolfinx.fem.Form | ufl.Form,
        bcs: list[dolfinx.fem.DirichletBC], mat_assemble_method: Literal['lift', 'identity_row'] = 'lift',
        sparse_method: Literal['bicg', 'gmres', 'lsqr'] = 'bicg',
        with_residual=False
    ) -> dict:
        if isinstance(a_form, ufl.Form):
            a_form = DolfinUtils.ufl_to_dolfin(a_form)
        if isinstance(l_form, ufl.Form):
            l_form = DolfinUtils.ufl_to_dolfin(l_form)

        a_mat = AssembleUtils.assemble_mat(a_form, bcs, method=mat_assemble_method)
        a_mat = PetscUtils.mat_to_scipy(a_mat)

        b_vec = AssembleUtils.assemble_vec(l_form)
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, a_form, clean_vec=False, method=mat_assemble_method)
        b_vec: np.ndarray = PetscUtils.vec_to_np(b_vec)

        x, is_solved = ScipySparseUtils.linear_solve(a_mat, b_vec, method=sparse_method)
        res_dict = {'is_solved': is_solved}
        if res_dict['is_solved']:
            uh.x.array[:] = x
            res_dict.update({'res': uh})

        if with_residual:
            errors = (a_mat.dot(x) - b_vec)
            res_dict.update({'max_residual': np.max(np.abs(errors))})

        return res_dict

    @staticmethod
    def solve_fea_dolfin(
        uh: dolfinx.fem.Function, a_form: ufl.Form, L_form: ufl.Form,
        bcs: list[dolfinx.fem.DirichletBC], ksp_option: dict|None = None
    ):
        if ksp_option is None:
            ksp_option = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
            }
        problem = LinearProblem(a_form, L_form, bcs, u=uh, petsc_options=ksp_option)
        uh = problem.solve()
        res_dict = {'res': uh}
        return res_dict

    @staticmethod
    def solve_by_petsc(
        b_vec: PETSc.Vec, a_mat: PETSc.Mat, solver: PETSc.KSP,
        setOperators=False, with_residual=False
    ) -> dict:
        """
        solverType0: method to solve the problem
        solverType1: method to preprocessing the data
        """
        if setOperators:
            solver.setOperators(a_mat)
        res_vec = PetscUtils.create_vec(PetscUtils.get_size(b_vec))

        solver.solve(b_vec, res_vec)
        res_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        res_dict = {'res': res_vec, 'converged_reason': solver.getConvergedReason()}
        res_dict.update({'is_solved': LinearProblemSolver.is_ksp_converged(res_dict['converged_reason'])})

        if with_residual:
            errors = (a_mat * res_vec).array - b_vec.array
            res_dict.update({'max_residual': np.max(np.abs(errors))})
        return res_dict

    @staticmethod
    def solve_fea_petsc(
            comm,
            uh: dolfinx.fem.Function,
            a_form: dolfinx.fem.Form | ufl.Form, l_form: dolfinx.fem.Form | ufl.Form,
            bcs: list[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['lift', 'identity_row'] = 'lift',
            ksp_option=None,
            with_residual=False
    ) -> dict:
        if isinstance(a_form, ufl.Form):
            a_form = DolfinUtils.ufl_to_dolfin(a_form)
        if isinstance(l_form, ufl.Form):
            l_form = DolfinUtils.ufl_to_dolfin(l_form)

        # import time
        # start_time = time.time()
        a_mat = AssembleUtils.assemble_mat(a_form, bcs, method=mat_assemble_method)
        # print(f"Assemble matrix time: {time.time() - start_time}")

        # start_time = time.time()
        b_vec = AssembleUtils.assemble_vec(l_form)
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, a_form, clean_vec=False, method=mat_assemble_method)
        # print(f"Assemble vector time: {time.time() - start_time}")
        
        # PetscUtils.save_data(a_mat, '/home/admin123456/Desktop/work/gmsh_test/t23/a_mat.dat')
        # PetscUtils.save_data(b_vec, '/home/admin123456/Desktop/work/gmsh_test/t23/b_vec.dat')
        
        solver = LinearProblemSolver.create_petsc_solver(comm, ksp_option, a_mat)
        # solver.setMonitor(lambda _, it, residual: print(f"Iter:{it} residual:{residual}"))
        res_dict = LinearProblemSolver.solve_by_petsc(b_vec, a_mat, solver, setOperators=False, with_residual=with_residual)

        solver.destroy()
        a_mat.destroy()
        b_vec.destroy()
        PETSc.garbage_cleanup(comm=comm)

        if res_dict['is_solved']:
            '''
            vec.axpy(x, y) = vec + x * y
            vec.aypx(x, y) = vec * x + y
            '''
            uh.x.petsc_vec.aypx(0.0, res_dict['res'])
            res_dict.update({'res': uh})

        return res_dict

    @staticmethod
    def is_ksp_converged(KSP_code: int):
        return KSP_code in [1, 9, 2, 3, 7, 4]


class PetscNonlinearPdeHelper:
    @staticmethod
    def residual(
        snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec,
        solution: dolfinx.fem.Function, F_form: dolfinx.fem.Form,
        jacobi_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        '''
        more reference about apply lifting:
            https://jsdokken.com/FEniCS23-tutorial/src/lifting.html
        '''
        assert mat_assemble_method in ['identity_row', 'lift']

        # ------ update the solution in the F_form
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(solution.x.petsc_vec)  # copy x to solution
        solution.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # ------ clear rhs(right hand side, residual)
        with F_vec.localForm() as f_local:
            f_local.set(0.0)

        # ------ recompute residual since solution change
        dolfinx_petsc.assemble_vector(F_vec, F_form)

        # ------ apply boundary function to residual
        if mat_assemble_method == 'lift':
            # apply_lifting = scale * (bc.value[dof] - x0[dof])
            dolfinx_petsc.apply_lifting(F_vec, [jacobi_form], bcs=[bcs], x0=[x], alpha=-1.0)
            F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # TODO why need x0=x
        dolfinx_petsc.set_bc(F_vec, bcs, x0=x, alpha=-1.0)  # x0[dof] - bc.value[dof] 促使x等于Dirichlet边界的值
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    @staticmethod
    def jacobi(
        snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P: PETSc.Mat,
        jacobi_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        # petsc.assemble_matrix(J_mat, jacobi_form, bcs=bcs, diagonal=1.0)  # recompute the jacobi matrix
        AssembleUtils.assemble_mat(a_form=jacobi_form, bcs=bcs, a_mat=J_mat, diagonal=1.0, method=mat_assemble_method)
        J_mat.assemble()

    @staticmethod
    def obj_func(snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec):
        return F_vec.norm()


class NonLinearProblemSolver(object):
    @staticmethod
    def create_petsc_solver(comm, snes_setting: dict|None = None, ksp_setting: dict|None = None, criterion={}):
        """
        Ref: https://petsc4py.readthedocs.io/en/stable/manual/snes/

        snes_setting = {
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_linesearch_alpha': 1e-1,      # slope descent parameter
            'snes_linesearch_damping': 1e-2,    # initial step length
            'snes_linesearch_order': 3,         # order of approximation for trial result
            'snes_linesearch_maxstep': 1.0,     # max step length
            'snes_linesearch_minlambda': 1e-6,  # minimum step length
            'snes_linesearch_max_it': 40,       # maximum number of shrinking step
            'snes_linesearch_keeplambda': 1,    # keep previous step as init

            'snes_type': 'newtonls',
            'snes_linesearch_type': 'basic',    # not a line search at all, only full step scale by damping parameter
            'snes_linesearch_maxstep': 1.0,     # max step length
            'snes_linesearch_minlambda': 1e-6,  # minimum step length
            # 'snes_linesearch_norms': 1,       # bool for basic method
            'snes_linesearch_damping': 1.0,     # search vector is scaled by this amount

            'snes_type': 'newtonls',
            'snes_linesearch_type': 'cp',
            'snes_linesearch_damping': 1.0,     # initial step length scale by this amount
            'snes_linesearch_maxstep': 1.0,     # max step length
            'snes_linesearch_minlambda': 1e-6,  # minimum step length
            'snes_linesearch_max_it': 40,       # maximum number of shrinking step

            'snes_type': 'ncg',
            'snes_ncg_type': 'prp',
            'snes_linesearch_type': 'cp'

            'snes_type': 'qn',
            'snes_qn_type': 'lbfgs',
            'snes_linesearch_type': 'cp'
        }
        """

        solver = PETSc.SNES().create(comm)
        solver.setTolerances(
            rtol=criterion.get('rtol', 1e-6),
            atol=criterion.get('atol', 1e-6),
            max_it=criterion.get('max_it', 1000),
        )

        # ------ snes setting
        # solver.setType(snes_setting.get('type', 'newtonls'))
        if snes_setting is not None:
            opts = PETSc.Options()
            for key, value in snes_setting.items():
                opts.setValue(key, value)
            solver.setFromOptions()

        # ------ ksp setting
        if ksp_setting is not None:
            ksp = solver.getKSP()
            opts = PETSc.Options()
            for key, value in ksp_setting.items():
                opts.setValue(key, value)
            ksp.setFromOptions()

        return solver

    @staticmethod
    def log_petsc_solver(solver: PETSc.SNES):
        solver.view()

    @staticmethod
    def solve_by_dolfin(
        comm, F_form: ufl.Form, uh: dolfinx.fem.Function, bcs: list[dolfinx.fem.DirichletBC], with_residual=False,
    ) -> dict:
        problem = dolfinx_petsc.NonlinearProblem(F_form, uh, bcs)
        solver = NewtonSolver(comm, problem)

        solver.convergence_criterion = 'incremental'
        solver.rtol = 1e-6
        solver.atol = 1e-6
        # solver.max_it = kwargs.pop('max_it', 1000)
        solver.report = False

        ksp: PETSc.KSP = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        # for key, value in ksp_option.items():
        #     opts[f"{option_prefix}{key}"] = value
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
        opts[f"{option_prefix}pc_type"] = "hypre"
        opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
        opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
        opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
        ksp.setFromOptions()

        _, is_converged = solver.solve(uh)
        res = {'res': uh, 'is_solved': is_converged}

        if with_residual:
            residual_vec = AssembleUtils.assemble_vec(F_form)
            res.update({'max_residual': np.max(np.abs(residual_vec.array))})

        return res

    @staticmethod
    def solve_fea_petsc(
        comm,
        f_form: dolfinx.fem.Form|ufl.Form, uh: dolfinx.fem.Function,
        jacobi_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        snes_setting: dict|None = None, ksp_setting: dict|None = None, criterion={},
        with_residual=False, with_monitor=False
    ) -> dict:
        """Please have the correct init solution of uh"""
        # dFdu = ufl.derivative(F_form, u, ufl.TrialFunction(V))  # jacobi_form
        if isinstance(f_form, ufl.Form):
            f_form = DolfinUtils.ufl_to_dolfin(f_form)
        if isinstance(jacobi_form, ufl.Form):
            jacobi_form = DolfinUtils.ufl_to_dolfin(jacobi_form)

        jacobi_mat: PETSc.Mat = dolfinx_petsc.create_matrix(jacobi_form)
        residual_vec: PETSc.Vec = dolfinx_petsc.create_vector(f_form)

        residual_func = partial(
            PetscNonlinearPdeHelper.residual,
            solution=uh,
            F_form=f_form,
            jacobi_form=jacobi_form,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        jacobi_func = partial(
            PetscNonlinearPdeHelper.jacobi,
            jacobi_form=jacobi_form,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        obj_func = partial(
            PetscNonlinearPdeHelper.obj_func,
            F_vec=residual_vec
        )

        solver = NonLinearProblemSolver.create_petsc_solver(
            comm=comm, snes_setting=snes_setting, ksp_setting=ksp_setting, criterion=criterion
        )
        solver.setObjective(obj_func)
        solver.setFunction(residual_func, residual_vec)
        solver.setJacobian(jacobi_func, jacobi_mat, P=None)

        # ------ Just For Debug
        if with_monitor:
            solver.setMonitor(lambda _, it, residual: print(f"Iter:{it} residual:{residual}"))
            NonLinearProblemSolver.log_petsc_solver(solver)

        # ------ newton setting
        solver.setConvergenceHistory()
        solver.solve(None, uh.x.petsc_vec)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        res_dict = {'res': uh, 'converged_reason': solver.getConvergedReason(),}
        res_dict.update({'is_solved': NonLinearProblemSolver.is_sens_converged(res_dict['converged_reason'])})

        if with_residual:
            convergence_history = solver.getConvergenceHistory()[0]
            res_dict.update({
                'max_residual': np.max(np.abs(residual_vec.array)),
                'object_function': residual_vec.norm(),
                'last_residual': convergence_history[-1] if len(convergence_history) else 0.0,
            })

        solver.destroy()
        residual_vec.destroy()
        jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm)

        return res_dict

    @staticmethod
    def is_sens_converged(SENS_code: int):
        return SENS_code in [2, 3, 6]


if __name__ == '__main__':
    pass
