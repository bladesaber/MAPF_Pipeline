import scipy.sparse
from petsc4py import PETSc
import numpy as np
import dolfinx
from mpi4py import MPI
import time
from dolfinx.fem.petsc import LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import petsc
import ufl
from typing import List, Union
from functools import partial
import os
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .sparse_utils import PETScUtils
from .dolfinx_utils import AssembleUtils, BoundaryUtils

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
            comm=MPI.COMM_WORLD, solver_setting: dict = None, A_mat: PETSc.Mat = None
    ):
        """
        find more info from: https://petsc4py.readthedocs.io/en/stable/manual/ksp/
        """
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

        # ------ detail use
        opts = PETSc.Options()
        # option_prefix = solver.getOptionsPrefix()
        for key, value in solver_setting.items():
            opts.setValue(key, value)
        solver.setFromOptions()

        if A_mat is not None:
            solver.setOperators(A_mat)

        return solver

    @staticmethod
    def view_solver(solver: PETSc.KSP):
        solver.view()

    @staticmethod
    def solve_by_np(A_mat: np.ndarray, b_vec: np.ndarray, with_debug=False):
        if with_debug:
            tick0 = time.time()

        res = np.dot(np.linalg.inv(A_mat), b_vec)
        res_dict = {'res': res}

        if with_debug:
            cost_time = time.time() - tick0
            errors = np.dot(A_mat, res) - b_vec
            res_dict.update({
                'cost_time': cost_time,
                'mean_error': np.mean(np.abs(errors)),
                'max_error': np.max(np.abs(errors))
            })
        return res_dict

    @staticmethod
    def solve_by_petsc(
            b_vec: PETSc.Vec, solver: PETSc.KSP, A_mat: PETSc.Mat, setOperators=False, with_debug=False
    ):
        """
        solverType0: method to solve the problem
        solverType1: method to preprocessing the data
        """
        if setOperators:
            solver.setOperators(A_mat)

        res_vec = PETScUtils.create_vec(PETScUtils.get_size(b_vec))

        if with_debug:
            tick0 = time.time()

        solver.solve(b_vec, res_vec)
        res_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        res_dict = {'res': res_vec}

        if with_debug:
            cost_time = time.time() - tick0

            errors = (A_mat * res_vec).array - b_vec.array
            mean_error = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))

            convergence_reason = solver.getConvergedReason()
            iter_num = solver.getIterationNumber()

            res_dict.update({
                'mean_error': mean_error,
                'max_error': max_error,
                'cost_time': cost_time,
                'convergence_reason': convergence_reason,
                'iter_num': iter_num,
                'converged_reason': solver.getConvergedReason()
            })
            # if res_dict['converged_reason'] < 0:
            #     raise ValueError(f"[ERROR]: KSP Converge Fail Code {res_dict['converged_reason']}")

        return res_dict

    @staticmethod
    def solve_by_dolfinx(
            uh: dolfinx.fem.function.Function,
            a_form: ufl.Form, L_form: ufl.Form, bcs: List[dolfinx.fem.DirichletBC],
            ksp_option=None
    ):
        if ksp_option is None:
            ksp_option = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
            }

        problem = LinearProblem(a_form, L_form, bcs, u=uh, petsc_options=ksp_option)

        tick0 = time.time()
        uh = problem.solve()
        cost_time = time.time() - tick0

        res_dict = {
            'res': uh,
            'cost_time': cost_time,
        }
        return res_dict

    @staticmethod
    def solve_by_petsc_form(
            comm,
            uh: dolfinx.fem.function.Function,
            a_form: Union[ufl.Form, dolfinx.fem.Form],
            L_form: Union[ufl.Form, dolfinx.fem.Form],
            bcs: List[dolfinx.fem.DirichletBC],
            ksp_option=None,
            **kwargs
    ):
        if isinstance(a_form, ufl.Form):
            a_compile_form = dolfinx.fem.form(a_form)
        else:
            a_compile_form = a_form

        if isinstance(L_form, ufl.Form):
            b_compile_form = dolfinx.fem.form(L_form)
        else:
            b_compile_form = L_form

        tick0 = time.time()
        a_mat = AssembleUtils.assemble_mat(a_compile_form, bcs, method=kwargs.get('A_assemble_method', 'lift'))
        time_a_mat = time.time() - tick0

        tick0 = time.time()
        b_vec = AssembleUtils.assemble_vec(b_compile_form)
        BoundaryUtils.apply_boundary_to_vec(
            b_vec, bcs, a_compile_form, clean_vec=False, method=kwargs.get('A_assemble_method', 'lift')
        )
        time_b_vec = time.time() - tick0

        if kwargs.get('record_mat_dir', False):
            PETScUtils.save_data(a_mat, os.path.join(kwargs['record_mat_dir'], 'A_mat.dat'))
            PETScUtils.save_data(b_vec, os.path.join(kwargs['record_mat_dir'], 'b_vec.dat'))

        solver = LinearProblemSolver.create_petsc_solver(comm, ksp_option, a_mat)
        res_dict = LinearProblemSolver.solve_by_petsc(
            b_vec, solver, a_mat, setOperators=False, with_debug=kwargs.get('with_debug', False)
        )

        solver.destroy()
        a_mat.destroy()
        b_vec.destroy()
        PETSc.garbage_cleanup(comm=comm)
        PETSc.garbage_cleanup()

        uh.vector.aypx(0.0, res_dict['res'])
        res_dict.update({
            'res': uh,
            'Amat_time': time_a_mat,
            'bvec_time': time_b_vec
        })

        return res_dict

    @staticmethod
    def solve_by_scipy_form(
            uh: dolfinx.fem.function.Function,
            a_form: Union[ufl.Form, dolfinx.fem.Form],
            L_form: Union[ufl.Form, dolfinx.fem.Form],
            bcs: List[dolfinx.fem.DirichletBC],
            ksp_option={'method': 'spsolve', 'params': {}},
            **kwargs
    ):
        if isinstance(a_form, ufl.Form):
            a_compile_form = dolfinx.fem.form(a_form)
        else:
            a_compile_form = a_form

        if isinstance(L_form, ufl.Form):
            b_compile_form = dolfinx.fem.form(L_form)
        else:
            b_compile_form = L_form

        tick0 = time.time()
        a_mat = AssembleUtils.assemble_mat(a_compile_form, bcs, method=kwargs.get('A_assemble_method', 'lift'))
        a_mat: sparse.csr_matrix = PETScUtils.convert_mat_to_scipy(a_mat)
        time_a_mat = time.time() - tick0

        tick0 = time.time()
        b_vec = AssembleUtils.assemble_vec(b_compile_form)
        BoundaryUtils.apply_boundary_to_vec(
            b_vec, bcs, a_compile_form, clean_vec=False, method=kwargs.get('A_assemble_method', 'lift')
        )
        b_vec: np.ndarray = b_vec.array
        time_b_vec = time.time() - tick0

        if kwargs.get('record_mat_dir', False):
            PETScUtils.save_data(a_mat, os.path.join(kwargs['record_mat_dir'], 'A_mat.dat'))
            PETScUtils.save_data(b_vec, os.path.join(kwargs['record_mat_dir'], 'b_vec.dat'))

        with_debug = kwargs.get('with_debug', False)
        if with_debug:
            tick0 = time.time()

        exit_code = 0
        if ksp_option['method'] == 'spsolve':
            x = sparse_linalg.spsolve(a_mat, b_vec)

        elif ksp_option['method'] == 'spsolve_triangular':
            x = sparse_linalg.spsolve_triangular(a_mat, b_vec)

        elif ksp_option['method'] == 'bicg':
            x, exit_code = sparse_linalg.bicg(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'bicgstab':
            x, exit_code = sparse_linalg.bicgstab(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'cg':
            x, exit_code = sparse_linalg.cg(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'cgs':
            x, exit_code = sparse_linalg.cgs(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'gmres':
            x, exit_code = sparse_linalg.gmres(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'lgmres':
            x, exit_code = sparse_linalg.lgmres(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'minres':
            x, exit_code = sparse_linalg.minres(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'qmr':
            x, exit_code = sparse_linalg.qmr(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'gcrotmk':
            x, exit_code = sparse_linalg.gcrotmk(a_mat, b_vec, **ksp_option['params'])

        elif ksp_option['method'] == 'tfqmr':
            x, exit_code = sparse_linalg.tfqmr(a_mat, b_vec, **ksp_option['params'])

        # elif ksp_option['method'] == 'lsqr':
        #     x, istop, itn, residual = sparse_linalg.lsqr(a_mat, b_vec, **ksp_option['params'])[:4]
        #
        # elif ksp_option['method'] == 'lsmr':
        #     x, istop, itn, residual = sparse_linalg.lsmr(a_mat, b_vec, **ksp_option['params'])[:4]

        else:
            raise ValueError('[ERROR]: Non-Valid Method')

        assert exit_code == 0

        res_dict = {}

        if with_debug:
            cost_time = time.time() - tick0

            errors = a_mat.dot(x) - b_vec
            mean_error = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))

            res_dict.update({
                'mean_error': mean_error,
                'max_error': max_error,
                'cost_time': cost_time,
            })

        uh.vector.x[:] = x
        res_dict.update({
            'res': uh,
            'Amat_time': time_a_mat,
            'bvec_time': time_b_vec
        })

        return res_dict

    @staticmethod
    def equation_investigation(
            a_form: Union[ufl.Form, dolfinx.fem.Form],
            L_form: Union[ufl.Form, dolfinx.fem.Form],
            bcs: List[dolfinx.fem.DirichletBC],
            uh: dolfinx.fem.function.Function = None,
            **kwargs
    ):
        if isinstance(a_form, ufl.Form):
            a_compile_form = dolfinx.fem.form(a_form)
        else:
            a_compile_form = a_form

        if isinstance(L_form, ufl.Form):
            b_compile_form = dolfinx.fem.form(L_form)
        else:
            b_compile_form = L_form

        a_mat = AssembleUtils.assemble_mat(a_compile_form, bcs, method=kwargs.get('A_assemble_method', 'lift'))
        b_vec = AssembleUtils.assemble_vec(b_compile_form)
        BoundaryUtils.apply_boundary_to_vec(
            b_vec, bcs, a_compile_form, clean_vec=False, method=kwargs.get('A_assemble_method', 'lift')
        )

        if kwargs.get('record_mat_dir', False):
            PETScUtils.save_data(a_mat, os.path.join(kwargs['record_mat_dir'], 'A_mat.dat'))
            PETScUtils.save_data(b_vec, os.path.join(kwargs['record_mat_dir'], 'b_vec.dat'))

        if uh is not None:
            errors = (a_mat * uh.vector).array - b_vec.array
            mean_error = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))

            res_dict = {
                'mean_error': mean_error,
                'max_error': max_error
            }
            return res_dict
        else:
            return {}

    @staticmethod
    def is_converged(KSP_code: int):
        return KSP_code in [1, 9, 2, 3, 7]


class NonlinearPdeHelper:
    @staticmethod
    def residual(
            snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec,
            solution: dolfinx.fem.Function, F_form: dolfinx.fem.Form,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            method: str
    ):
        assert method in ['Identity_row', 'lift']

        # ------ update the solution in the F_form
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(solution.vector)
        solution.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # ------ clear rhs(right hand side, residual)
        with F_vec.localForm() as f_local:
            f_local.set(0.0)

        # ------ recompute residual since solution change
        petsc.assemble_vector(F_vec, F_form)

        # ------ apply boundary function to residual
        if method == 'lift':
            # apply_lifting = scale * (bc.value[dof] - x0[dof])
            petsc.apply_lifting(F_vec, [jacobi_form], bcs=[bcs], x0=[x], scale=-1.0)
            F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # set_bc = scale * (bc.value[dof] - x0[dof])
        petsc.set_bc(F_vec, bcs, x0=x, scale=-1.0)  # x0[dof] - bc.value[dof]
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    @staticmethod
    def jacobi(
            snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P: PETSc.Mat,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            method='lift'
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        # petsc.assemble_matrix(J_mat, jacobi_form, bcs=bcs, diagonal=1.0)  # recompute the jacobi matrix
        AssembleUtils.assemble_mat(
            a_form=jacobi_form, bcs=bcs, A_mat=J_mat,
            diagonal=1.0,
            method=method
        )
        J_mat.assemble()

    @staticmethod
    def obj_func(snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec):
        return F_vec.norm()


class NonLinearProblemSolver(object):
    @staticmethod
    def create_petsc_solver(
            comm=MPI.COMM_WORLD,
            snes_setting: dict = {},
            ksp_setting: dict = {},
            **kwargs
    ):
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
            rtol=kwargs.pop('rtol', 1e-6), atol=kwargs.pop('atol', 1e-6), max_it=kwargs.pop('max_it', 1000),
        )

        # ------ snes setting
        # solver.setType(snes_setting.get('type', 'newtonls'))

        if len(snes_setting) > 0:
            opts = PETSc.Options()
            for key, value in snes_setting.items():
                opts.setValue(key, value)
            solver.setFromOptions()

        # ------ ksp setting
        ksp = solver.getKSP()

        # ksp.setType(solver_setting.get("ksp_type", "preonly"))
        # ksp.getPC().setType(solver_setting.get("pc_type", "lu"))
        # if "pc_factor_mat_solver_type" in solver_setting.keys():
        #     ksp.getPC().setFactorSolverType(solver_setting["pc_factor_mat_solver_type"])
        # if "pc_hypre_mat_solver_type" in solver_setting.keys():
        #     ksp.getPC().setHYPREType(solver_setting["pc_hypre_mat_solver_type"])

        opts = PETSc.Options()
        for key, value in ksp_setting.items():
            opts.setValue(key, value)
        ksp.setFromOptions()

        return solver

    @staticmethod
    def view_solver(solver: PETSc.KSP):
        solver.view()

    @staticmethod
    def solve_by_dolfinx(
            F_form: ufl.Form, uh: dolfinx.fem.Function, bcs: List[dolfinx.fem.DirichletBC],
            comm=MPI.COMM_WORLD, ksp_option={},
            with_debug=False, **kwargs,
    ):
        problem = dolfinx.fem.petsc.NonlinearProblem(F_form, uh, bcs)
        solver = NewtonSolver(comm, problem)

        # ------ ksp setting
        ksp = solver.krylov_solver

        ksp.setType(ksp_option.get("ksp_type", "preonly"))
        ksp.getPC().setType(ksp_option.get("pc_type", "lu"))

        if "pc_factor_mat_solver_type" in ksp_option.keys():
            ksp.getPC().setFactorSolverType(ksp_option["pc_factor_mat_solver_type"])

        if "pc_hypre_mat_solver_type" in ksp_option.keys():
            ksp.getPC().setHYPREType(ksp_option["pc_hypre_mat_solver_type"])

        # ------ newton setting
        solver.convergence_criterion = 'incremental'
        solver.rtol = kwargs.pop('rtol', 1e-10)
        solver.atol = kwargs.pop('atol', 1e-10)
        solver.max_it = kwargs.pop('max_it', 1000)

        if with_debug:
            tick0 = time.time()

        n_times, is_converg = solver.solve(uh)

        res = {
            'res': uh,
            'run_times': n_times,
            'is_converge': is_converg,
        }

        if with_debug:
            res.update({
                'cost_time': time.time() - tick0
            })

        return res

    @staticmethod
    def solve_by_petsc(
            F_form: ufl.Form, uh: dolfinx.fem.Function,
            jacobi_form: ufl.Form, bcs: List[dolfinx.fem.DirichletBC],
            comm=MPI.COMM_WORLD,
            snes_setting: dict = {},
            ksp_option={},
            with_debug=False, **kwargs
    ):
        """
        please have the correct init solution of u
        """
        # dFdu = ufl.derivative(F_form, u, ufl.TrialFunction(V))  # jacobi_form

        F_dolfinx = dolfinx.fem.form(F_form)
        dFdu_dolfinx = dolfinx.fem.form(jacobi_form)

        jacobi_mat: PETSc.Mat = petsc.create_matrix(dFdu_dolfinx)
        residual_vec: PETSc.Vec = petsc.create_vector(F_dolfinx)

        residual_func = partial(
            NonlinearPdeHelper.residual,
            solution=uh,
            F_form=F_dolfinx,
            jacobi_form=dFdu_dolfinx,
            bcs=bcs,
            method=kwargs.get('A_assemble_method', 'lift')
        )
        jacobi_func = partial(
            NonlinearPdeHelper.jacobi,
            jacobi_form=dFdu_dolfinx,
            bcs=bcs,
            method=kwargs.get('A_assemble_method', 'lift')
        )
        obj_func = partial(
            NonlinearPdeHelper.obj_func,
            F_vec=residual_vec
        )

        solver = NonLinearProblemSolver.create_petsc_solver(
            comm=comm, snes_setting=snes_setting, ksp_setting=ksp_option, **kwargs
        )
        solver.setObjective(obj_func)
        solver.setFunction(residual_func, residual_vec)
        solver.setJacobian(jacobi_func, jacobi_mat, P=None)

        # ------ Just For Debug
        # solver.setMonitor(lambda _, it, residual: print(f"Iter:{it} residual:{residual}"))
        # NonLinearProblemSolver.view_solver(solver)

        # ------ newton setting
        if with_debug:
            tick0 = time.time()
            solver.setConvergenceHistory()

        solver.solve(None, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        res_dict = {'res': uh}

        if with_debug:
            convergence_history = solver.getConvergenceHistory()[0]

            res_dict.update({
                'mean_error': np.mean(np.abs(residual_vec)),
                'max_error': np.max(np.abs(residual_vec)),
                'norm_error': residual_vec.norm(),
                'cost_time': time.time() - tick0,
                'converged_history': convergence_history,
                'last_error': convergence_history[-1] if len(convergence_history) else 0.0,
                'converged_reason': solver.getConvergedReason(),
                'iteration_number': solver.getIterationNumber(),
            })
            # if res_dict['converged_reason'] < 0:
            #     raise ValueError(f"[ERROR]: SNES Converge Fail Code {res_dict['converged_reason']}")

        solver.destroy()
        residual_vec.destroy()
        jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm)
        PETSc.garbage_cleanup()

        return res_dict

    @staticmethod
    def equation_investigation(
            lhs_form: ufl.Form,
            bcs: List[dolfinx.fem.DirichletBC],
            uh: dolfinx.fem.function.Function = None,
            **kwargs
    ):
        if kwargs.get('record_mat_dir', False):
            # todo 不确认非线性等式转为线性表述是否合理，应该不合理
            lhs_mat_form = ufl.replace(lhs_form, {uh, ufl.TrialFunction(uh.function_space)})
            lhs_mat_form_dolfin = dolfinx.fem.form(lhs_mat_form)

            a_mat: PETSc.Mat = AssembleUtils.assemble_mat(
                lhs_mat_form_dolfin, bcs=bcs, method=kwargs.get('A_assemble_method', 'lift')
            )
            b_vec: PETSc.Vec = a_mat.getVecRight()
            BoundaryUtils.apply_boundary_to_vec(
                b_vec, bcs, lhs_mat_form_dolfin, clean_vec=False, method=kwargs.get('A_assemble_method', 'lift')
            )

            PETScUtils.save_data(a_mat, os.path.join(kwargs['record_mat_dir'], 'A_mat.dat'))
            PETScUtils.save_data(b_vec, os.path.join(kwargs['record_mat_dir'], 'b_vec.dat'))

        if uh is not None:
            lhs_form_dolfin = dolfinx.fem.form(lhs_form)
            lhs_vec: PETSc.Vec = AssembleUtils.assemble_vec(lhs_form_dolfin)

            petsc.set_bc(lhs_vec, bcs, x0=uh.vector, scale=-1.0)
            lhs_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            errors = lhs_vec.array
            max_error = np.max(np.abs(errors))

            res_dict = {
                'max_error': max_error
            }
            return res_dict
        else:
            return {}

    @staticmethod
    def is_converged(SNES_code: int):
        return SNES_code in [2, 3, 6]


def find_linear_ksp_option(
        record_file: str,
        A_mat_dat: Union[str, PETSc.Mat],
        b_vec_dat: Union[str, PETSc.Vec],
        ref_record_file: str = None,
        ksp_types: List[str] = None,
        pc_types: List[str] = None,
        other_ksp_option: dict = {}
):
    if isinstance(A_mat_dat, str):
        assert A_mat_dat.endswith('.dat')
    if isinstance(b_vec_dat, str):
        assert b_vec_dat.endswith('.dat')
    assert record_file.endswith('.csv')

    if ksp_types is None:
        ksp_types = [
            'richardson', 'preonly', 'cg', 'pipecg', 'groppcg', 'pipecgrr',
            'cgne', 'fcg', 'pipefcg', 'cgls', 'nash', 'stcg', 'gltr',
            'bicg', 'bcgs', 'ibcgs', 'qmrcgs', 'fbcgs', 'bcgsl', 'minres', 'gmres',
            'fgmres', 'dgmres', 'pgmres', 'pipefgmres', 'lgmres', 'cr', 'gcr',
            'pipecr', 'cgs', 'tfqmr', 'tcqmr', 'lsqr', 'symmlq',
            # 'chebyshev', 'qcg', 'tsirm', 'fetidp',
        ]
    if pc_types is None:
        pc_types = [
            'lu', 'ksp', 'none',
            'jacobi', 'bjacobi', 'sor', 'eisenstat', 'icc', 'ilu', 'asm',
            'gasm', 'gamg', 'cholesky',
            # 'bddc', 'composite'  # give up
        ]

    if isinstance(A_mat_dat, str):
        A_mat = PETScUtils.create_mat()
        PETScUtils.load_data(A_mat, A_mat_dat)
    else:
        A_mat = A_mat_dat

    if isinstance(b_vec_dat, str):
        b_vec = PETScUtils.create_vec()
        PETScUtils.load_data(b_vec, b_vec_dat)
    else:
        b_vec = b_vec_dat

    print(f"A_mat Size:{PETScUtils.get_size(A_mat)}, b_vec Size:{PETScUtils.get_size(b_vec)}")
    print(f"b_vec isNan:{np.any(np.isnan(b_vec))}, isInf:{np.any(np.isinf(b_vec))} abs_sum:{np.sum(np.abs(b_vec))}")

    if os.path.exists(record_file):
        record = pd.read_csv(record_file, index_col=0)
    else:
        record = pd.DataFrame(columns=['ksp', 'pc', 'costTime', 'meanError', 'maxError', 'log'])

    ref_record_enable = False
    if ref_record_file is not None:
        if os.path.exists(ref_record_file):
            ref_record = pd.read_csv(ref_record_file, index_col=0)
            ref_record_enable = True

    for ksp_type in ksp_types:
        for pc_type in pc_types:
            res = record[(record['ksp'] == ksp_type) & (record['pc'] == pc_type)]
            if res.shape[0] > 0:
                continue

            if ref_record_enable:
                ref_res = ref_record[(ref_record['ksp'] == ksp_type) & (ref_record['pc'] == pc_type)]
                if ref_res.shape[0] > 0:
                    if ref_res['log'] != '':
                        print(f"[###] Skip [{ksp_type}] [{pc_type}]")
                        continue

            idx = record.shape[0]
            try:
                trial_ksp_option = {'ksp_type': ksp_type, 'pc_type': pc_type}
                trial_ksp_option.update(other_ksp_option)

                solver = LinearProblemSolver.create_petsc_solver(
                    MPI.COMM_WORLD, A_mat=A_mat,
                    solver_setting=trial_ksp_option
                )
                res_dict = LinearProblemSolver.solve_by_petsc(
                    b_vec, solver, A_mat, setOperators=False, with_debug=True
                )

                record.loc[
                    idx, ['ksp', 'pc', 'costTime', 'meanError', 'maxError', 'log']
                ] = ksp_type, pc_type, res_dict['cost_time'], res_dict['mean_error'], res_dict['max_error'], ''
                print(
                    f"[{ksp_type}] [{pc_type}]: costTime:{res_dict['cost_time']:.3f}, "
                    f"meanError:{res_dict['mean_error']:.8f}, maxError:{res_dict['max_error']:.8f}"
                )

            except Exception as e:
                record.loc[idx, ['ksp', 'pc', 'log']] = ksp_type, pc_type, str(e)
                print(f"[{ksp_type}] [{pc_type}]: {str(e)}")

            record.to_csv(record_file)


def find_nonlinear_ksp_option(
        comm, F_form: ufl.Form, jacobi_form: ufl.Form, uh: dolfinx.fem.Function, bcs: List[dolfinx.fem.DirichletBC],
        record_file: str, ref_record_file: str = None
):
    ksp_types = [
        'richardson', 'preonly', 'cg', 'pipecg', 'groppcg', 'pipecgrr',
        'cgne', 'fcg', 'pipefcg', 'cgls', 'nash', 'stcg', 'gltr',
        'bicg', 'bcgs', 'ibcgs', 'qmrcgs', 'fbcgs', 'bcgsl', 'minres', 'gmres',
        'fgmres', 'dgmres', 'pgmres', 'pipefgmres', 'lgmres', 'cr', 'gcr',
        'pipecr', 'cgs', 'tfqmr', 'tcqmr', 'lsqr', 'symmlq',
        # 'chebyshev', 'qcg', 'tsirm', 'fetidp',
    ]
    pc_types = [
        'lu', 'ksp', 'none',
        'jacobi', 'bjacobi', 'sor', 'eisenstat', 'icc', 'ilu', 'asm',
        'gasm', 'gamg', 'cholesky',
        # 'bddc', 'composite'  # give up
    ]

    if os.path.exists(record_file):
        record = pd.read_csv(record_file, index_col=0)
    else:
        record = pd.DataFrame(columns=['ksp', 'pc', 'costTime', 'meanError', 'maxError', 'log'])

    ref_record_enable = False
    if ref_record_file is not None:
        if os.path.exists(ref_record_file):
            ref_record = pd.read_csv(ref_record_file, index_col=0)
            ref_record_enable = True

    uh_tmp_np = uh.x.array
    for ksp_type in ksp_types:
        for pc_type in pc_types:

            if ref_record_enable:
                ref_res = ref_record[(ref_record['ksp'] == ksp_type) & (ref_record['pc'] == pc_type)]
                if ref_res.shape[0] > 0:
                    if ref_res['log'] != '':
                        print(f"[###] Skip [{ksp_type}] [{pc_type}]")
                        continue

            up_tmp = dolfinx.fem.Function(uh.function_space)
            up_tmp.x.array[:] = uh_tmp_np

            F_form_tmp = ufl.replace(F_form, {uh: up_tmp})
            jacobi_form_tmp = ufl.replace(jacobi_form, {uh: up_tmp})

            idx = record.shape[0]
            try:
                res_dict = NonLinearProblemSolver.solve_by_petsc(
                    F_form=F_form_tmp,
                    uh=up_tmp,
                    jacobi_form=jacobi_form_tmp,
                    bcs=bcs,
                    comm=comm,
                    ksp_option={'ksp_type': ksp_type, 'pc_type': pc_type},
                    with_debug=True
                )

                record.loc[
                    idx, ['ksp', 'pc', 'costTime', 'meanError', 'maxError', 'log']
                ] = ksp_type, pc_type, res_dict['cost_time'], res_dict['mean_error'], res_dict['max_error'], ''
                print(
                    f"[{ksp_type}] [{pc_type}]: costTime:{res_dict['cost_time']:.3f}, "
                    f"meanError:{res_dict['mean_error']:.8f}, maxError:{res_dict['max_error']:.8f}"
                )

            except Exception as e:
                record.loc[idx, ['ksp', 'pc', 'log']] = ksp_type, pc_type, str(e)
                print(f"[{ksp_type}] [{pc_type}]: {str(e)}")
