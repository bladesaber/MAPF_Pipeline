from petsc4py import PETSc
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import petsc
import ufl
from typing import List, Union, Literal, Dict
from functools import partial
import pandas as pd
import pyvista

from scripts_py.version_9.dolfinx_grad.sparse_utils import PetscUtils, ScipySparseUtils
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils, BoundaryUtils

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
    def create_petsc_solver(comm=MPI.COMM_WORLD, solver_setting: dict = None, mat: PETSc.Mat = None):
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

        opts = PETSc.Options()
        for key, value in solver_setting.items():
            opts.setValue(key, value)
        solver.setFromOptions()

        if mat is not None:
            solver.setOperators(mat)
        return solver

    @staticmethod
    def log_petsc_solver(solver: PETSc.KSP):
        solver.view()

    @staticmethod
    def solve_by_np(a_mat: np.ndarray, b_vec: np.ndarray, with_debug=False):
        res_vec = np.dot(np.linalg.inv(a_mat), b_vec)
        res_dict = {'res': res_vec, 'is_solved': True}
        if with_debug:
            errors = np.dot(a_mat, res_vec) - b_vec
            res_dict.update({'max_error': np.max(np.abs(errors))})
        return res_dict

    @staticmethod
    def solve_by_petsc(b_vec: PETSc.Vec, solver: PETSc.KSP, a_mat: PETSc.Mat, setOperators=False, with_debug=False):
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
        res_dict.update({
            'is_solved': LinearProblemSolver.is_ksp_converged(res_dict['converged_reason'])
        })

        if with_debug:
            errors = (a_mat * res_vec).array - b_vec.array
            max_error = np.max(np.abs(errors))
            res_dict.update({'max_error': max_error})
        return res_dict

    @staticmethod
    def solve_fea_dolfin(
            uh: dolfinx.fem.function.Function, a_form: ufl.Form, L_form: ufl.Form,
            bcs: List[dolfinx.fem.DirichletBC], ksp_option=None
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
    def solve_fea_petsc(
            comm,
            uh: dolfinx.fem.function.Function,
            a_form: dolfinx.fem.Form, l_form: dolfinx.fem.Form,
            bcs: List[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['lift', 'Identity_row'] = 'lift',
            ksp_option=None,
            with_debug=False
    ):
        # 最耗时操作
        a_mat = AssembleUtils.assemble_mat(a_form, bcs, method=mat_assemble_method)

        b_vec = AssembleUtils.assemble_vec(l_form)
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, a_form, clean_vec=False, method=mat_assemble_method)

        solver = LinearProblemSolver.create_petsc_solver(comm, ksp_option, a_mat)
        res_dict = LinearProblemSolver.solve_by_petsc(b_vec, solver, a_mat, setOperators=False, with_debug=with_debug)

        solver.destroy()
        a_mat.destroy()
        b_vec.destroy()
        PETSc.garbage_cleanup(comm=comm)
        PETSc.garbage_cleanup()

        if res_dict['is_solved']:
            uh.x.petsc_vec.aypx(0.0, res_dict['res'])
            res_dict.update({'res': uh})

        return res_dict

    @staticmethod
    def solve_fea_scipy(
            uh: dolfinx.fem.function.Function,
            a_form: dolfinx.fem.Form, l_form: dolfinx.fem.Form,
            bcs: List[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['lift', 'Identity_row'] = 'lift',
            with_debug=False
    ):
        a_mat = AssembleUtils.assemble_mat(a_form, bcs, method=mat_assemble_method)
        a_mat = PetscUtils.convert_mat_to_scipy(a_mat)

        b_vec = AssembleUtils.assemble_vec(l_form)
        BoundaryUtils.apply_boundary_to_vec(b_vec, bcs, a_form, clean_vec=False, method=mat_assemble_method)
        b_vec: np.ndarray = b_vec.array

        x, is_solved = ScipySparseUtils.linear_solve(a_mat, b_vec, method='bicg')
        res_dict = {'is_solved': is_solved}
        if res_dict['is_solved']:
            uh.x.array[:] = x
            res_dict.update({'res': uh})

        if with_debug:
            errors = (a_mat.dot(x) - b_vec)
            max_error = np.max(np.abs(errors))
            res_dict.update({'max_error': max_error})

        return res_dict

    @staticmethod
    def is_ksp_converged(KSP_code: int):
        return KSP_code in [1, 9, 2, 3, 7, 4]


class PetscNonlinearPdeHelper:
    @staticmethod
    def residual(
            snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec,
            solution: dolfinx.fem.Function, F_form: dolfinx.fem.Form,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['Identity_row', 'lift']
    ):
        assert mat_assemble_method in ['Identity_row', 'lift']

        # ------ update the solution in the F_form
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(solution.x.petsc_vec)
        solution.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # ------ clear rhs(right hand side, residual)
        with F_vec.localForm() as f_local:
            f_local.set(0.0)

        # ------ recompute residual since solution change
        petsc.assemble_vector(F_vec, F_form)

        # ------ apply boundary function to residual
        if mat_assemble_method == 'lift':
            # apply_lifting = scale * (bc.value[dof] - x0[dof])
            petsc.apply_lifting(F_vec, [jacobi_form], bcs=[bcs], x0=[x], alpha=-1.0)
            F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        petsc.set_bc(F_vec, bcs, x0=x, alpha=-1.0)  # x0[dof] - bc.value[dof] 促使x等于Dirichlet边界的值
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    @staticmethod
    def jacobi(
            snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P: PETSc.Mat,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['Identity_row', 'lift']
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        # petsc.assemble_matrix(J_mat, jacobi_form, bcs=bcs, diagonal=1.0)  # recompute the jacobi matrix
        AssembleUtils.assemble_mat(
            a_form=jacobi_form, bcs=bcs, a_mat=J_mat,
            diagonal=1.0,
            method=mat_assemble_method
        )
        J_mat.assemble()

    @staticmethod
    def obj_func(snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec):
        return F_vec.norm()


class NonLinearProblemSolver(object):
    @staticmethod
    def create_petsc_solver(comm, snes_setting: Dict = None, ksp_setting: Dict = None, criterion={}):
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
            comm, F_form: ufl.Form, uh: dolfinx.fem.Function, bcs: List[dolfinx.fem.DirichletBC], with_debug=False,
    ):
        problem = dolfinx.fem.petsc.NonlinearProblem(F_form, uh, bcs)
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

        if with_debug:
            error = AssembleUtils.assemble_vec(F_form)
            res.update({'max_error': np.max(np.abs(error.array))})

        return res

    @staticmethod
    def solve_fea_petsc(
            comm,
            f_form: dolfinx.fem.Form, uh: dolfinx.fem.Function,
            jacobi_form: dolfinx.fem.Form, bcs: List[dolfinx.fem.DirichletBC],
            mat_assemble_method: Literal['Identity_row', 'lift'],
            snes_setting: Dict = None, ksp_setting: Dict = None, criterion={},
            with_debug=False, with_monitor=False
    ):
        """Please have the correct init solution of uh"""
        # dFdu = ufl.derivative(F_form, u, ufl.TrialFunction(V))  # jacobi_form

        jacobi_mat: PETSc.Mat = petsc.create_matrix(jacobi_form)  # todo
        residual_vec: PETSc.Vec = petsc.create_vector(f_form)

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

        res_dict = {
            'res': uh,
            'converged_reason': solver.getConvergedReason(),
        }
        res_dict.update({
            'is_solved': NonLinearProblemSolver.is_sens_converged(res_dict['converged_reason'])
        })

        if with_debug:
            convergence_history = solver.getConvergenceHistory()[0]

            res_dict.update({
                'max_error': np.max(np.abs(residual_vec.array)),
                'norm_error': residual_vec.norm(),
                'last_error': convergence_history[-1] if len(convergence_history) else 0.0,
            })

        solver.destroy()
        residual_vec.destroy()
        jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm)
        PETSc.garbage_cleanup()

        return res_dict

    @staticmethod
    def is_sens_converged(SENS_code: int):
        return SENS_code in [2, 3, 6]


def debug():
    from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
    from ufl import inner, grad, div

    mesh = MeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model.msh', dim=2
    )

    P2 = ufl.VectorElement("CG", mesh.domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("CG", mesh.domain.ufl_cell(), 1)
    TH = ufl.MixedElement([P2, P1])
    W = dolfinx.fem.FunctionSpace(mesh.domain, TH)
    W0, W1 = W.sub(0), W.sub(1)
    V, V_to_W = W0.collapse()
    Q, Q_to_W = W1.collapse()

    wall_func = dolfinx.fem.Function(V, name='wall')
    wall_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), mesh.facet_dim,
        MeshUtils.extract_entity_indices(
            mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=15
        )
    )
    wall_bc = dolfinx.fem.dirichletbc(wall_func, wall_dofs, W0)

    inlet_func = dolfinx.fem.Function(V, name='inlet')
    inlet_func.x.array[:] = 1.0
    inlet_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), mesh.facet_dim,
        MeshUtils.extract_entity_indices(
            mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=13
        )
    )
    inlet_bc = dolfinx.fem.dirichletbc(inlet_func, inlet_dofs, W0)

    outlet_func = dolfinx.fem.Function(Q, name='outlet')
    outlet_dofs = MeshUtils.extract_entity_dofs(
        (W1, Q), mesh.facet_dim,
        MeshUtils.extract_entity_indices(
            mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=14
        )
    )
    outlet_bc = dolfinx.fem.dirichletbc(outlet_func, outlet_dofs, W1)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = dolfinx.fem.Constant(mesh.domain, np.zeros(mesh.topology_dim))

    nu = 1. / 400.
    f_form = nu * inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx + div(u) * q * ufl.dx - inner(f, v) * ufl.dx
    a_form: ufl.form.Form = ufl.lhs(f_form)
    l_form: ufl.form.Form = ufl.rhs(f_form)

    uh = dolfinx.fem.Function(W, name='res')
    res_dict = LinearProblemSolver.solve_fea_dolfin(
        uh, a_form, l_form, bcs=[inlet_bc, outlet_bc, wall_bc],
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'}
    )
    res_dict = LinearProblemSolver.solve_fea_petsc(
        comm=MPI.COMM_WORLD, uh=uh,
        a_form=dolfinx.fem.form(a_form), l_form=dolfinx.fem.form(l_form),
        bcs=[inlet_bc, outlet_bc, wall_bc],
        ksp_option={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        with_debug=True
    )

    vel_ref = dolfinx.fem.Function(dolfinx.fem.VectorFunctionSpace(mesh.domain, ('CG', 1)), name='vel_ref')
    vel_ref.interpolate(uh.sub(0).collapse())
    p_ref = uh.sub(1).collapse()

    vel_ref_np = vel_ref.x.array.reshape((-1, mesh.topology_dim))
    if vel_ref_np.shape[1] == 2:
        vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
    mesh.grid.point_data['velocity'] = vel_ref_np
    mesh.grid.point_data['pressure'] = p_ref.x.array
    mesh.grid.save('/home/admin123456/Desktop/work/topopt_exps/new_test/example01/res.vtu', binary=False)

    # plotter = pyvista.Plotter()
    # plotter.view_xy()
    # plotter.add_arrows(mesh.grid.points, mesh.grid.point_data['velocity'], mag=0.1, show_scalar_bar=False)
    # plotter.show()


if __name__ == '__main__':
    debug()
