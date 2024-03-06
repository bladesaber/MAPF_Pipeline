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

from .petsc_utils import PETScUtils
from .dolfinx_utils import AssembleUtils, BoundaryUtils

"""
Convergence Reasons:
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
"""


class LinearProblemSolver(object):
    @staticmethod
    def create_petsc_solver(comm=MPI.COMM_WORLD, solver_setting: dict = None, A_mat: PETSc.Mat = None):
        """
        find more info from: https://petsc4py.readthedocs.io/en/stable/manual/ksp/
        """
        solver: PETSc.KSP = PETSc.KSP().create(comm)
        if solver_setting is None:
            return solver

        # ------- easy use
        # solver.setType(solver_setting["ksp_type"])
        # solver.getPC().setType(solver_setting["pc_type"])

        # ------ detail use
        # solver.report = True
        # opts = PETSc.Options()
        # option_prefix = solver.getOptionsPrefix()

        # opts[f"{option_prefix}ksp_type"] = solver_setting.get("ksp_type", "preonly")
        solver.setType(solver_setting.get("ksp_type", "preonly"))

        # opts[f"{option_prefix}pc_type"] = solver_setting.get("pc_type", 'lu')
        solver.getPC().setType(solver_setting.get("pc_type", "lu"))

        if "pc_factor_mat_solver_type" in solver_setting.keys():
            # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            solver.getPC().setFactorSolverType(solver_setting["pc_factor_mat_solver_type"])

        if "pc_hypre_mat_solver_type" in solver_setting.keys():
            solver.getPC().setHYPREType(solver_setting["pc_hypre_mat_solver_type"])

        solver.setFromOptions()

        if A_mat is not None:
            solver.setOperators(A_mat)

        return solver

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
            # solver.setConvergenceHistory()
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
            # convergence_history = solver.getConvergenceHistory()

            res_dict.update({
                'mean_error': mean_error,
                'max_error': max_error,
                'cost_time': cost_time,
                'convergence_reason': convergence_reason,
                'iter_num': iter_num,
                # 'convergence_last_history': convergence_history[-1] if len(convergence_history) else 0
            })

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
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, a_compile_form, clean_vec=False)
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


class NonlinearPdeHelper:
    @staticmethod
    def residual(
            snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec,
            solution: dolfinx.fem.Function, F_form: dolfinx.fem.Form,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC]
    ):
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
        petsc.apply_lifting(F_vec, [jacobi_form], bcs=[bcs], x0=[x], scale=-1.0)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(F_vec, bcs, x, -1.0)

    @staticmethod
    def jacobi(
            snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P: PETSc.Mat,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            method='lift'
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        # petsc.assemble_matrix(J_mat, jacobi_form, bcs=bcs, diagonal=1.0)  # recompute the jacobi matrix
        AssembleUtils.assemble_mat(a_form=jacobi_form, bcs=bcs, A_mat=J_mat, diagonal=1.0, method=method)
        J_mat.assemble()

    @staticmethod
    def obj_func(snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec):
        return F_vec.norm()


class NonLinearProblemSolver(object):
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
            comm=MPI.COMM_WORLD, ksp_option={}, with_debug=False, **kwargs
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
            bcs=bcs
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

        solver = PETSc.SNES().create(comm)
        solver.setObjective(obj_func)
        solver.setFunction(residual_func, residual_vec)
        solver.setJacobian(jacobi_func, jacobi_mat, P=None)
        solver.setTolerances(
            rtol=kwargs.pop('rtol', 1e-10),
            atol=kwargs.pop('atol', 1e-10),
            max_it=kwargs.pop('max_it', 1000),
        )
        # solver.setMonitor(lambda _, it, residual: print(f"Iter:{it} residual:{residual}"))

        # ------ ksp setting
        ksp = solver.getKSP()

        ksp.setType(ksp_option.get("ksp_type", "preonly"))
        ksp.getPC().setType(ksp_option.get("pc_type", "lu"))

        if "pc_factor_mat_solver_type" in ksp_option.keys():
            ksp.getPC().setFactorSolverType(ksp_option["pc_factor_mat_solver_type"])

        if "pc_hypre_mat_solver_type" in ksp_option.keys():
            ksp.getPC().setHYPREType(ksp_option["pc_hypre_mat_solver_type"])

        # ------ newton setting
        if with_debug:
            tick0 = time.time()
            solver.setConvergenceHistory()

        solver.solve(None, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        res = {'res': uh}

        if with_debug:
            convergence_history = solver.getConvergenceHistory()[0]

            res.update({
                'mean_error': np.mean(np.abs(residual_vec)),
                'max_error': np.max(np.abs(residual_vec)),
                'norm_error': residual_vec.norm(),
                'cost_time': time.time() - tick0,
                'last_error': convergence_history[-1] if len(convergence_history) else 0
            })

        solver.destroy()
        residual_vec.destroy()
        jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm)
        PETSc.garbage_cleanup()

        return res


def find_linear_ksp_option(
        record_file: str,
        A_mat_dat: Union[str, PETSc.Mat],
        b_vec_dat: Union[str, PETSc.Vec],
        ref_record_file: str = None,
        ksp_types: List[str] = None,
        pc_types: List[str] = None
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
                solver = LinearProblemSolver.create_petsc_solver(
                    MPI.COMM_WORLD, A_mat=A_mat,
                    solver_setting={'ksp_type': ksp_type, 'pc_type': pc_type}
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
