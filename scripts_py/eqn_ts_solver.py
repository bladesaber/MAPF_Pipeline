from typing import Literal, Callable

import dolfinx
import ufl
from dolfinx.fem import petsc as dolfinx_petsc

import numpy as np
import petsc4py.PETSc as PETSc
from functools import partial

from scripts_py.dolfinx_utils import AssembleUtils, DolfinUtils
from scripts_py.vtu_recorder import VtuRecorder
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver


class PetscTsPdeHelper(object):
    @staticmethod
    def preStepFunction(
        solver: PETSc.TS,  # type: ignore
        source_term: dolfinx.fem.Function, 
        interpolate_func: Callable
    ) -> None:
        t: float = solver.getTime()
        source_term.interpolate(partial(interpolate_func, t=t))  # type: ignore
    
    @staticmethod
    def postStepFunction(solver: PETSc.TS, record_func: Callable) -> None:  # type: ignore
        record_func(solver.getTime(), solver.getTimeStep())
    
    @staticmethod
    def evalIFunction(
        solver: PETSc.TS, t: float, u: PETSc.Vec, u_dot: PETSc.Vec, F_vec: PETSc.Vec,  # type: ignore
        u_solution: dolfinx.fem.Function, 
        u_dot_solution: dolfinx.fem.Function,
        F_form: dolfinx.fem.Form,
        jacobi_form: dolfinx.fem.Form, 
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
    ):
        assert mat_assemble_method in ['identity_row', 'lift']
        
        # ------ update the solution in the F_form
        u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        u.copy(u_solution.x.petsc_vec)  # copy u to u_solution
        u_solution.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    # type: ignore
        
        u_dot.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        u_dot.copy(u_dot_solution.x.petsc_vec)  # copy u_dot to u_dot_solution
        u_dot_solution.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    # type: ignore
        
        # ------ clear rhs(right hand side, residual)
        with F_vec.localForm() as f_local:  # type: ignore
            f_local.set(0.0)
        
        # ------ recompute residual since solution change
        dolfinx_petsc.assemble_vector(F_vec, F_form)  # type: ignore
        
        # ------ apply boundary function to residual
        if mat_assemble_method == 'lift':
            # apply_lifting = scale * (bc.value[dof] - x0[dof])
            dolfinx_petsc.apply_lifting(F_vec, [jacobi_form], bcs=[bcs], x0=[u], alpha=-1.0)  # type: ignore
            F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        
        dolfinx_petsc.set_bc(F_vec, bcs, x0=u, alpha=-1.0)  # x0[dof] - bc.value[dof] 促使x等于Dirichlet边界的值
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    @staticmethod
    def evalIJacobian(
        solver: PETSc.TS, t: float, u: PETSc.Vec, u_dot: PETSc.Vec, shift: float, J_mat: PETSc.Mat, P: PETSc.Mat,    # type: ignore
        jacobi_form: dolfinx.fem.Form, 
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        AssembleUtils.assemble_mat(a_form=jacobi_form, bcs=bcs, a_mat=J_mat, diagonal=1.0, method=mat_assemble_method)
        J_mat.assemble()
    
    @staticmethod
    def evalRHSFunction(
        solver: PETSc.TS, t: float, u: PETSc.Vec, G_vec: PETSc.Vec,  # type: ignore
        u_solution: dolfinx.fem.Function, 
        G_form: dolfinx.fem.Form,
        jacobi_form: dolfinx.fem.Form, 
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        # ------ update the solution in the F_form
        u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        u.copy(u_solution.x.petsc_vec)  # copy u to u_solution
        u_solution.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    # type: ignore
        
        # ------ clear rhs(right hand side, residual)
        with G_vec.localForm() as f_local:  # type: ignore
            f_local.set(0.0)
        
        # ------ recompute residual since solution change
        dolfinx_petsc.assemble_vector(G_vec, G_form)  # type: ignore
        
        # ------ apply boundary function to residual
        if mat_assemble_method == 'lift':
            # apply_lifting = scale * (bc.value[dof] - x0[dof])
            dolfinx_petsc.apply_lifting(G_vec, [jacobi_form], bcs=[bcs], x0=[u], alpha=-1.0)  # type: ignore
            G_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        
        dolfinx_petsc.set_bc(G_vec, bcs, x0=u, alpha=-1.0)  # x0[dof] - bc.value[dof] 促使x等于Dirichlet边界的值
        G_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    @staticmethod
    def evalRHSJacobian(
        solver: PETSc.TS, t: float, u: PETSc.Vec, J_mat: PETSc.Mat, P: PETSc.Mat,    # type: ignore
        jacobi_form: dolfinx.fem.Form, 
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        J_mat.zeroEntries()  # clear jacobi matrix
        AssembleUtils.assemble_mat(a_form=jacobi_form, bcs=bcs, a_mat=J_mat, diagonal=1.0, method=mat_assemble_method)
        J_mat.assemble()
    
    @staticmethod
    def obj_func(solver: PETSc.TS, x: PETSc.Vec, F_vec: PETSc.Vec):  # type: ignore
        return F_vec.norm()


class TSProblemSolver(object):
    '''
    TS Convergence Reasons:
    ref: https://petsc.org/main/manualpages/TS/TSConvergedReason/
    
    TS_CONVERGED_ITERATING          = 0,
    TS_CONVERGED_TIME               = 1,
    TS_CONVERGED_ITS                = 2,
    TS_CONVERGED_USER               = 3,
    TS_CONVERGED_EVENT              = 4,
    TS_CONVERGED_PSEUDO_FATOL       = 5,
    TS_CONVERGED_PSEUDO_FRTOL       = 6,
    TS_DIVERGED_NONLINEAR_SOLVE     = -1,
    TS_DIVERGED_STEP_REJECTED       = -2,
    TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
    TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
    '''
    
    ts_type = {
        'euler': PETSc.TS.Type.EULER, # type: ignore
        'backward_euler': PETSc.TS.Type.BEULER, # type: ignore
        'rosw': PETSc.TS.Type.ROSW, # type: ignore
        'runge-kutta': PETSc.TS.Type.RK, # type: ignore
        'crank-nicolson': PETSc.TS.Type.CN, # type: ignore
    }
    
    problem_type = {
        'nonlinear': PETSc.TS.ProblemType.NONLINEAR, # type: ignore
        'linear': PETSc.TS.ProblemType.LINEAR, # type: ignore
    }
    
    equation_type = {
        'unspecified': PETSc.TS.EquationType.UNSPECIFIED, # type: ignore
        'explicit': PETSc.TS.EquationType.EXPLICIT, # type: ignore
        'implicit': PETSc.TS.EquationType.IMPLICIT, # type: ignore
        'ode_explicit': PETSc.TS.EquationType.ODE_EXPLICIT, # type: ignore
        'ode_implicit': PETSc.TS.EquationType.ODE_IMPLICIT, # type: ignore
        'dae_semi_explicit_index1': PETSc.TS.EquationType.DAE_SEMI_EXPLICIT_INDEX1, # type: ignore
        'dae_semi_explicit_index2': PETSc.TS.EquationType.DAE_SEMI_EXPLICIT_INDEX2, # type: ignore
        'dae_semi_explicit_index3': PETSc.TS.EquationType.DAE_SEMI_EXPLICIT_INDEX3, # type: ignore
        'dae_semi_explicit_indexhi': PETSc.TS.EquationType.DAE_SEMI_EXPLICIT_INDEXHI, # type: ignore
        'dae_implicit_index1': PETSc.TS.EquationType.DAE_IMPLICIT_INDEX1, # type: ignore
        'dae_implicit_index2': PETSc.TS.EquationType.DAE_IMPLICIT_INDEX2, # type: ignore
        'dae_implicit_index3': PETSc.TS.EquationType.DAE_IMPLICIT_INDEX3, # type: ignore
        'dae_implicit_indexhi': PETSc.TS.EquationType.DAE_IMPLICIT_INDEXHI, # type: ignore
    }
    
    @staticmethod
    def create_petsc_solver(
        comm,
        ts_type: str,
        problem_type: str,
        initial_conditions: PETSc.Vec, # type: ignore
        t0: float,
        dt: float,
        max_time: float,
        max_time_step: int = 5000,
        equation_type: str|None = None,
        snes_setting: dict|None = None, 
        ksp_setting: dict|None = None,
        criterion: dict = {},
        with_snes_monitor=False
    ):
        assert ts_type in TSProblemSolver.ts_type.keys(), f"Invalid time-stepping type: {ts_type}"
        assert problem_type in TSProblemSolver.problem_type.keys(), f"Invalid problem type: {problem_type}"

        solver = PETSc.TS().create(comm) # type: ignore
        solver.setType(TSProblemSolver.ts_type[ts_type])
        solver.setProblemType(TSProblemSolver.problem_type[problem_type])
        # setEquationType usually depends on the ts_type
        if equation_type is not None:
            assert equation_type in TSProblemSolver.equation_type.keys(), f"Invalid equation type: {equation_type}"
            solver.setEquationType(TSProblemSolver.equation_type[equation_type])
        
        solver.setSolution(initial_conditions)
        solver.setTime(t0)  # Set the initial time
        solver.setTimeStep(dt)  # Set the time step        
        solver.setMaxTime(max_time)  # Set the maximum time
        solver.setMaxSteps(max_time_step)  # Set the maximum time step
        
        snes_solver = solver.getSNES()
        if snes_setting is not None:
            opts = PETSc.Options() # type: ignore
            for key, value in snes_setting.items():
                opts.setValue(key, value)
            snes_solver.setFromOptions()
        
        snes_solver.setTolerances(
            rtol=criterion.get('rtol', 1e-6),
            atol=criterion.get('atol', 1e-6),
            max_it=criterion.get('max_it', 1000),
        )
        
        if with_snes_monitor:
            snes_solver.setMonitor(lambda _, it, residual: print(f"SNESTep:{it} residual:{residual}"))
        
        if ksp_setting is not None:
            ksp = snes_solver.getKSP()
            opts = PETSc.Options() # type: ignore
            for key, value in ksp_setting.items():
                opts.setValue(key, value)
            ksp.setFromOptions()
                
        solver.setFromOptions()
        
        return solver

    @staticmethod
    def create_implicit_kwargs(
        f_form: dolfinx.fem.Form|ufl.Form,
        f_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        sigma: float|None,
        uh: dolfinx.fem.Function, 
        u_dot: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        dt: float
    ):
        if f_jacobi_form is not None:
            if isinstance(f_jacobi_form, ufl.Form):
                f_jacobi_form = DolfinUtils.ufl_to_dolfin(f_jacobi_form)
        else:
            if sigma is None:
                sigma = 1.0 / dt
            f_jacobi_form = (
                sigma * ufl.derivative(f_form, u_dot, ufl.TrialFunction(uh.function_space))  # type: ignore
                + ufl.derivative(f_form, uh, ufl.TrialFunction(uh.function_space))  # type: ignore
            )
            f_jacobi_form = DolfinUtils.ufl_to_dolfin(f_jacobi_form)  # type: ignore
        
        if isinstance(f_form, ufl.Form):
            f_form = DolfinUtils.ufl_to_dolfin(f_form)
        
        jacobi_mat: PETSc.Mat = dolfinx_petsc.create_matrix(f_jacobi_form)  # type: ignore
        residual_vec: PETSc.Vec = dolfinx_petsc.create_vector(f_form)  # type: ignore
        
        iFunction_func = partial(
            PetscTsPdeHelper.evalIFunction,
            u_solution=uh,
            u_dot_solution=u_dot,
            F_form=f_form,
            jacobi_form=f_jacobi_form, 
            bcs=bcs, 
            mat_assemble_method=mat_assemble_method
        )
        iJacobian_func = partial(
            PetscTsPdeHelper.evalIJacobian,
            jacobi_form=f_jacobi_form,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        return {
            'iFunction_func': iFunction_func,
            'iJacobian_func': iJacobian_func,
            'jacobi_mat': jacobi_mat,
            'residual_vec': residual_vec
        }

    @staticmethod
    def create_explicit_kwargs(
        g_form: dolfinx.fem.Form|ufl.Form,
        g_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        uh: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift']
    ):
        if g_jacobi_form is not None:
            if isinstance(g_jacobi_form, ufl.Form):
                g_jacobi_form = DolfinUtils.ufl_to_dolfin(g_jacobi_form)
        else:
            g_jacobi_form = ufl.derivative(g_form, uh, ufl.TrialFunction(uh.function_space))  # type: ignore
            g_jacobi_form = DolfinUtils.ufl_to_dolfin(g_jacobi_form)  # type: ignore
        
        if isinstance(g_form, ufl.Form):
            g_form = DolfinUtils.ufl_to_dolfin(g_form)
        
        g_vec: PETSc.Vec = dolfinx_petsc.create_vector(g_form)  # type: ignore
        g_jacobi_mat: PETSc.Mat = dolfinx_petsc.create_matrix(g_jacobi_form)  # type: ignore
        
        iRHSFunction_func = partial(
            PetscTsPdeHelper.evalRHSFunction,
            u_solution=uh,
            G_form=g_form,
            jacobi_form=g_jacobi_form,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        iRHSJacobian_func = partial(
            PetscTsPdeHelper.evalRHSJacobian,
            jacobi_form=g_jacobi_form,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        return {
            'iRHSFunction_func': iRHSFunction_func,
            'iRHSJacobian_func': iRHSJacobian_func,
            'g_vec': g_vec,
            'g_jacobi_mat': g_jacobi_mat
        }

    @staticmethod
    def solve_implicit_by_petsc(
        comm,
        f_form: dolfinx.fem.Form|ufl.Form,
        f_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        sigma: float|None,
        uh: dolfinx.fem.Function, 
        u_dot: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        ts_type: str,
        problem_type: str,
        t0: float,
        dt: float,
        max_time: float,
        max_time_step: int = 5000,
        equation_type: str|None = None,
        snes_setting: dict|None = None, 
        ksp_setting: dict|None = None,
        criterion: dict = {},
        source_term: dolfinx.fem.Function|None = None,
        interpolate_func: Callable|None = None,
        post_step_func: Callable|None = None,
        with_snes_monitor=False,
        only_solver: bool = False
    ) -> dict:
        implicit_kwargs = TSProblemSolver.create_implicit_kwargs(
            f_form=f_form,
            f_jacobi_form=f_jacobi_form,
            sigma=sigma,
            uh=uh,
            u_dot=u_dot,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method,
            dt=dt
        )
        iFunction_func = implicit_kwargs['iFunction_func']
        iJacobian_func = implicit_kwargs['iJacobian_func']
        jacobi_mat = implicit_kwargs['jacobi_mat']
        residual_vec = implicit_kwargs['residual_vec']
        
        solver = TSProblemSolver.create_petsc_solver(
            comm=comm,
            ts_type=ts_type,
            problem_type=problem_type,
            initial_conditions=uh.x.petsc_vec,  # type: ignore
            t0=t0,
            dt=dt,
            max_time=max_time,
            max_time_step=max_time_step,
            equation_type=equation_type,
            snes_setting=snes_setting,
            ksp_setting=ksp_setting,
            criterion=criterion,
            with_snes_monitor=with_snes_monitor
        )
        solver.setIFunction(iFunction_func, residual_vec)
        solver.setIJacobian(iJacobian_func, J=jacobi_mat, P=None)
        
        if (source_term is not None) and (interpolate_func is not None):
            pre_step_function = partial(
                PetscTsPdeHelper.preStepFunction,
                source_term=source_term,
                interpolate_func=interpolate_func
            )
            solver.setPreStep(pre_step_function)
        if post_step_func is not None:
            post_step_function = partial(
                PetscTsPdeHelper.postStepFunction,
                record_func=post_step_func
            )
            solver.setPostStep(post_step_function)
        
        if with_snes_monitor:
            # solver.setMonitor(lambda _, step, time, residual: print(f"Iter:{step}/{time} residual:{residual.norm()}"))
            TSProblemSolver.log_petsc_solver(solver)
        
        if only_solver:
            return {'solver': solver}
        
        solver.solve(uh.x.petsc_vec)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        
        res_dict = {'res': uh, 'converged_reason': solver.getConvergedReason()}
        res_dict.update({'is_solved': TSProblemSolver.is_ts_converged(res_dict['converged_reason'])})
        
        solver.destroy()
        residual_vec.destroy()
        jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm) # type: ignore

        return res_dict
    
    @staticmethod
    def solve_explicit_by_petsc(
        comm,
        g_form: dolfinx.fem.Form|ufl.Form,
        g_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        uh: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        ts_type: str,
        problem_type: str,
        t0: float,
        dt: float,
        max_time: float,
        max_time_step: int = 5000,
        equation_type: str|None = None,
        snes_setting: dict|None = None, 
        ksp_setting: dict|None = None,
        criterion: dict = {},
        source_term: dolfinx.fem.Function|None = None,
        interpolate_func: Callable|None = None,
        post_step_func: Callable|None = None,
        with_snes_monitor=False,
        only_solver: bool = False
    ) -> dict:
        # TODO: still not converged, why?
        explicit_kwargs = TSProblemSolver.create_explicit_kwargs(
            g_form=g_form,
            g_jacobi_form=g_jacobi_form,
            uh=uh,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        iRHSFunction_func = explicit_kwargs['iRHSFunction_func']
        iRHSJacobian_func = explicit_kwargs['iRHSJacobian_func']
        g_vec = explicit_kwargs['g_vec']
        g_jacobi_mat = explicit_kwargs['g_jacobi_mat']
        
        solver = TSProblemSolver.create_petsc_solver(
            comm=comm,
            ts_type=ts_type,
            problem_type=problem_type,
            initial_conditions=uh.x.petsc_vec,  # type: ignore
            t0=t0,
            dt=dt,
            max_time=max_time,
            max_time_step=max_time_step,
            equation_type=equation_type,
            snes_setting=snes_setting,
            ksp_setting=ksp_setting,
            criterion=criterion,
            with_snes_monitor=with_snes_monitor
        )
        solver.setRHSFunction(iRHSFunction_func, g_vec)
        solver.setRHSJacobian(iRHSJacobian_func, J=g_jacobi_mat, P=None)
        
        if (source_term is not None) and (interpolate_func is not None):
            pre_step_function = partial(
                PetscTsPdeHelper.preStepFunction,
                source_term=source_term,
                interpolate_func=interpolate_func
            )
            solver.setPreStep(pre_step_function)
        if post_step_func is not None:
            post_step_function = partial(
                PetscTsPdeHelper.postStepFunction,
                record_func=post_step_func
            )
            solver.setPostStep(post_step_function)
        
        if with_snes_monitor:
            # solver.setMonitor(lambda _, step, time, residual: print(f"Iter:{step}/{time} residual:{residual.norm()}"))
            TSProblemSolver.log_petsc_solver(solver)
        
        if only_solver:
            return {'solver': solver}
        
        solver.solve(uh.x.petsc_vec)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        
        res_dict = {'res': uh, 'converged_reason': solver.getConvergedReason()}
        res_dict.update({'is_solved': TSProblemSolver.is_ts_converged(res_dict['converged_reason'])})
        
        solver.destroy()
        g_vec.destroy()
        g_jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm) # type: ignore

        return res_dict

    @staticmethod
    def solve_imex_by_petsc(
        comm,
        f_form: dolfinx.fem.Form|ufl.Form, 
        f_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        sigma: float|None,
        g_form: dolfinx.fem.Form|ufl.Form,
        g_jacobi_form: dolfinx.fem.Form|ufl.Form|None,
        uh: dolfinx.fem.Function,
        u_dot: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        ts_type: str,
        problem_type: str,
        t0: float,
        dt: float,
        max_time: float,
        max_time_step: int = 5000,
        equation_type: str|None = None,
        snes_setting: dict|None = None, 
        ksp_setting: dict|None = None,
        criterion: dict = {},
        source_term: dolfinx.fem.Function|None = None,
        interpolate_func: Callable|None = None,
        post_step_func: Callable|None = None,
        with_snes_monitor=False,
        only_solver: bool = False
    ) -> dict:
        implicit_kwargs = TSProblemSolver.create_implicit_kwargs(
            f_form=f_form,
            f_jacobi_form=f_jacobi_form,
            sigma=sigma,
            uh=uh,
            u_dot=u_dot,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method,
            dt=dt
        )
        iFunction_func = implicit_kwargs['iFunction_func']
        iJacobian_func = implicit_kwargs['iJacobian_func']
        f_jacobi_mat = implicit_kwargs['jacobi_mat']
        f_vec = implicit_kwargs['residual_vec']
        
        explicit_kwargs = TSProblemSolver.create_explicit_kwargs(
            g_form=g_form,
            g_jacobi_form=g_jacobi_form,
            uh=uh,
            bcs=bcs,
            mat_assemble_method=mat_assemble_method
        )
        iRHSFunction_func = explicit_kwargs['iRHSFunction_func']
        iRHSJacobian_func = explicit_kwargs['iRHSJacobian_func']
        g_vec = explicit_kwargs['g_vec']
        g_jacobi_mat = explicit_kwargs['g_jacobi_mat']
        
        solver = TSProblemSolver.create_petsc_solver(
            comm=comm,
            ts_type=ts_type,
            problem_type=problem_type,
            initial_conditions=uh.x.petsc_vec,  # type: ignore
            t0=t0,
            dt=dt,
            max_time=max_time,
            max_time_step=max_time_step,
            equation_type=equation_type,
            snes_setting=snes_setting,
            ksp_setting=ksp_setting,
            criterion=criterion,
            with_snes_monitor=with_snes_monitor
        )
        solver.setIFunction(iFunction_func, f_vec)
        solver.setIJacobian(iJacobian_func, J=f_jacobi_mat, P=None)
        solver.setRHSFunction(iRHSFunction_func, g_vec)
        solver.setRHSJacobian(iRHSJacobian_func, J=g_jacobi_mat, P=None)
        
        if (source_term is not None) and (interpolate_func is not None):
            pre_step_function = partial(
                PetscTsPdeHelper.preStepFunction,
                source_term=source_term,
                interpolate_func=interpolate_func
            )
            solver.setPreStep(pre_step_function)
        if post_step_func is not None:
            post_step_function = partial(
                PetscTsPdeHelper.postStepFunction,
                record_func=post_step_func
            )
            solver.setPostStep(post_step_function)
        
        if with_snes_monitor:
            # solver.setMonitor(lambda _, step, time, residual: print(f"Iter:{step}/{time} residual:{residual.norm()}"))
            TSProblemSolver.log_petsc_solver(solver)
        
        if only_solver:
            return {'solver': solver}
        
        solver.solve(uh.x.petsc_vec)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        
        res_dict = {'res': uh, 'converged_reason': solver.getConvergedReason()}
        res_dict.update({'is_solved': TSProblemSolver.is_ts_converged(res_dict['converged_reason'])})
        
        solver.destroy()
        f_vec.destroy()
        f_jacobi_mat.destroy()
        g_vec.destroy()
        g_jacobi_mat.destroy()
        PETSc.garbage_cleanup(comm=comm) # type: ignore

        return res_dict

    @staticmethod
    def get_context(solver: PETSc.TS) -> dict: # type: ignore
        context = {
            'current_time': solver.getTime(),
            'current_time_step': solver.getTimeStep(),
        }
        return context

    @staticmethod
    def log_petsc_solver(solver: PETSc.TS): # type: ignore
        solver.view()

    @staticmethod
    def is_ts_converged(TS_code: int):
        return TS_code in [1, 2, 3, 4, 5, 6]

    @staticmethod
    def run_step(solver: PETSc.TS) -> None:  # type: ignore
        solver.step()


class TSDolfinxProblemSolver(object):
    @staticmethod
    def solve_explicit_euler_by_dolfinx(
        comm,
        f_form: dolfinx.fem.Form|ufl.Form,
        uh: dolfinx.fem.Function,
        u_old: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
        mat_assemble_method: Literal['identity_row', 'lift'],
        is_linear: bool,
        t0: float,
        dt: float,
        max_T: float,
        snes_setting: dict|None = None, 
        ksp_setting: dict|None = None,
        criterion: dict = {},
        source_term: dolfinx.fem.Function|None = None,
        interpolate_func: Callable|None = None,
        post_step_func: Callable|None = None,
        with_residual=False,
    ):
        run_steps = int(max_T / dt)
        if is_linear:
            a_form = ufl.lhs(f_form)
            L_form = ufl.rhs(f_form)
        
        for run_iter in range(run_steps):
            t_now = t0 + run_iter * dt
            if (source_term is not None) and (interpolate_func is not None):
                source_term.interpolate(partial(interpolate_func, t=t_now))  # type: ignore
            
            if is_linear:
                res_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    uh=uh,  # type: ignore
                    a_form=a_form,  # type: ignore
                    l_form=L_form,
                    bcs=bcs,
                    mat_assemble_method=mat_assemble_method,
                    ksp_option=ksp_setting,
                    with_residual=with_residual,
                )
                if not res_dict['is_solved']:
                    raise ValueError(f"Failed to solve linear problem.")
            
            else:
                jacobi_ufl = ufl.derivative(f_form, uh, ufl.TrialFunction(uh.function_space))
                jacobi_form = DolfinUtils.ufl_to_dolfin(jacobi_ufl)
                
                res_dict = NonLinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    f_form=f_form,
                    uh=uh,
                    jacobi_form=jacobi_form,  # type: ignore
                    bcs=bcs,
                    mat_assemble_method=mat_assemble_method,
                    snes_setting=snes_setting,
                    ksp_setting=ksp_setting,
                    criterion=criterion,
                    with_residual=with_residual,
                    with_monitor=False
                )
                if not res_dict['is_solved']:
                    raise ValueError(f"Failed to solve nonlinear problem.")
            
            uh.x.array[:] = res_dict['res'].x.array.copy()  # type: ignore
            u_old.x.array[:] = uh.x.array.copy()  # type: ignore
            
            if post_step_func is not None:
                post_step_func(t_now, run_iter)
        
        return uh


def debug_implicit_ts():
    from mpi4py import MPI
    from scripts_py.mesh_utils import MeshUtils, MeshObj
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    nx = 30
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [nx, nx],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    mesh = MeshObj(domain=domain)
    V = dolfinx.fem.functionspace(domain, ("P", 1))

    # 3. Define boundary conditions
    dofs = MeshUtils.extract_entity_dofs(
        V, mesh.fdim, MeshUtils.extract_entity_indices(mesh.domain, style='facet', marker=None)
    )
    bc = dolfinx.fem.dirichletbc(0.0, dofs, V)  # type: ignore
    
    def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-2.0 * np.pi**2 * t)

    def source_term_expr(x: np.ndarray, t: float) -> np.ndarray:
        """f(x,y,t) that makes the exact solution satisfy the PDE"""
        return (
            -2.0 * np.pi**2 * exact_solution(x, t)
            + 2.0 * np.pi**2 * np.exp(-2.0 * np.pi**2 * t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

    uh = dolfinx.fem.Function(V)
    u_dot = dolfinx.fem.Function(V)

    # ------ compute initial condition
    uh.interpolate(partial(exact_solution, t=0.0))  # type: ignore

    # ------ time-stepping parameters
    T = 0.5  # final time
    dt = 0.01  # time step size
    sigma = 1.0 / dt
    
    source_term = dolfinx.fem.Function(V)
    source_term.interpolate(partial(source_term_expr, t=0.0))  # type: ignore
    
    u_test = ufl.TestFunction(V)
    F_form = (
        u_dot * u_test * ufl.dx  # type: ignore
        + ufl.inner(ufl.grad(uh), ufl.grad(u_test)) * ufl.dx
        - source_term * u_test * ufl.dx  # type: ignore
    )
    # Jacobi_form = (
    #     sigma * ufl.derivative(F_form, u_dot, ufl.TrialFunction(uh.function_space))  # type: ignore
    #     + ufl.derivative(F_form, uh, ufl.TrialFunction(uh.function_space))  # type: ignore
    # )
    
    recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t12/ts', name='ts')
    def record_func(t: float, t_step: int):
        t_count = int(t / dt)
        recorder.update_point_data(key='u', value=uh.x.array.copy())  # type: ignore
        recorder.save(step=t_count)

    res_dict = TSProblemSolver.solve_implicit_by_petsc(
        comm=domain.comm,
        f_form=F_form,
        f_jacobi_form=None,
        sigma=sigma,
        uh=uh,  # type: ignore
        u_dot=u_dot,  # type: ignore
        bcs=[bc],
        mat_assemble_method='lift',
        ts_type='backward_euler',
        problem_type='nonlinear',
        t0=0.0,
        dt=dt,
        max_time=T,
        max_time_step=int(T / dt),
        ksp_setting=ksp_option,
        source_term=source_term, # type: ignore
        interpolate_func=source_term_expr,
        post_step_func=record_func,
        with_snes_monitor=True
    )
    if not res_dict['is_solved']:
        raise ValueError(f"Failed to solve time-stepping problem.")
    uh.x.array[:] = res_dict['res'].x.array.copy()  # type: ignore
    
    # ---------- 7. Compute L² error ----------
    u_exact = dolfinx.fem.Function(V)
    u_exact.interpolate(partial(exact_solution, t=T))  # type: ignore

    error_L2 = dolfinx.fem.assemble_scalar(DolfinUtils.ufl_to_dolfin(
        ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx  # type: ignore
    ))
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2, op=MPI.SUM))

    # ---------- 8. Report ----------
    print("\n=== Diffusion test ===")
    print(f"Mesh: {nx}×{nx} triangles")
    print(f"Time interval: [0, {T}],  dt = {dt}")
    print(f"L² error at t = {T}: {error_L2:.3e}")

    # tolerance is generous for the coarse mesh; tighten when you refine
    tol = 2e-2 if nx <= 20 else 5e-3
    assert error_L2 < tol, f"Error {error_L2:.3e} exceeds tolerance {tol:.3e}"
    print("Test PASSED\n")


def debug_explicit_ts():
    # TODO: still can not pass the test, why?
    from mpi4py import MPI
    from scripts_py.mesh_utils import MeshUtils, MeshObj
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    nx = 30
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [nx, nx],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    mesh = MeshObj(domain=domain)
    V = dolfinx.fem.functionspace(domain, ("P", 1))

    # 3. Define boundary conditions
    dofs = MeshUtils.extract_entity_dofs(
        V, mesh.fdim, MeshUtils.extract_entity_indices(mesh.domain, style='facet', marker=None)
    )
    bc = dolfinx.fem.dirichletbc(0.0, dofs, V)  # type: ignore
    
    def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-2.0 * np.pi**2 * t)

    def source_term_expr(x: np.ndarray, t: float) -> np.ndarray:
        """f(x,y,t) that makes the exact solution satisfy the PDE"""
        return (
            -2.0 * np.pi**2 * exact_solution(x, t)
            + 2.0 * np.pi**2 * np.exp(-2.0 * np.pi**2 * t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

    uh = dolfinx.fem.Function(V)

    # ------ compute initial condition
    uh.interpolate(partial(exact_solution, t=0.0))  # type: ignore

    # ------ time-stepping parameters
    T = 0.5  # final time
    dt = 0.01  # time step size
    
    source_term = dolfinx.fem.Function(V)
        
    # du/dt = G(u) = -Δu + f(t)
    u_test = ufl.TestFunction(V)
    G_form = (
        - ufl.inner(ufl.grad(uh), ufl.grad(u_test)) * ufl.dx  # type: ignore
        + source_term * u_test * ufl.dx  # type: ignore
    )
    # Jacobi_form = ufl.derivative(G_form, uh, ufl.TrialFunction(uh.function_space))  # type: ignore
    
    recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t12/explicit', name='ts')
    def record_func(t: float, t_step: int):
        t_count = int(t / dt)
        recorder.update_point_data(key='u', value=uh.x.array.copy())  # type: ignore
        recorder.save(step=t_count)

    res_dict = TSProblemSolver.solve_explicit_by_petsc(
        comm=domain.comm,
        g_form=G_form,
        g_jacobi_form=None,
        uh=uh,  # type: ignore
        bcs=[bc],
        mat_assemble_method='lift',
        ts_type='euler',
        problem_type='nonlinear',
        t0=0.0,
        dt=dt,
        max_time=T,
        max_time_step=int(T / dt),
        ksp_setting=ksp_option,
        source_term=source_term, # type: ignore
        interpolate_func=source_term_expr,
        post_step_func=record_func,
        with_snes_monitor=False
    )
    if not res_dict['is_solved']:
        raise ValueError(f"Failed to solve time-stepping problem.")
    uh.x.array[:] = res_dict['res'].x.array.copy()  # type: ignore
    
    # ---------- 7. Compute L² error ----------
    u_exact = dolfinx.fem.Function(V)
    u_exact.interpolate(partial(exact_solution, t=T))  # type: ignore

    error_L2 = dolfinx.fem.assemble_scalar(DolfinUtils.ufl_to_dolfin(
        ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx  # type: ignore
    ))
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2, op=MPI.SUM))

    # ---------- 8. Report ----------
    print("\n=== Diffusion test ===")
    print(f"Mesh: {nx}×{nx} triangles")
    print(f"Time interval: [0, {T}],  dt = {dt}")
    print(f"L² error at t = {T}: {error_L2:.3e}")

    # tolerance is generous for the coarse mesh; tighten when you refine
    tol = 2e-2 if nx <= 20 else 5e-3
    assert error_L2 < tol, f"Error {error_L2:.3e} exceeds tolerance {tol:.3e}"
    print("Test PASSED\n")


def debug_explicit_dolfinx():
    from mpi4py import MPI
    from scripts_py.mesh_utils import MeshUtils, MeshObj
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    nx = 30
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [nx, nx],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    mesh = MeshObj(domain=domain)
    V = dolfinx.fem.functionspace(domain, ("P", 1))

    # 3. Define boundary conditions
    dofs = MeshUtils.extract_entity_dofs(
        V, mesh.fdim, MeshUtils.extract_entity_indices(mesh.domain, style='facet', marker=None)
    )
    bc = dolfinx.fem.dirichletbc(0.0, dofs, V)  # type: ignore
    
    def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-2.0 * np.pi**2 * t)

    def source_term_expr(x: np.ndarray, t: float) -> np.ndarray:
        """f(x,y,t) that makes the exact solution satisfy the PDE"""
        return (
            -2.0 * np.pi**2 * exact_solution(x, t)
            + 2.0 * np.pi**2 * np.exp(-2.0 * np.pi**2 * t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

    uh = dolfinx.fem.Function(V)

    # ------ compute initial condition
    u_old = dolfinx.fem.Function(V)
    u_old.interpolate(partial(exact_solution, t=0.0))  # type: ignore

    # ------ time-stepping parameters
    T = 0.5  # final time
    dt = 0.01  # time step size
    t0 = dt
    
    source_term = dolfinx.fem.Function(V)
    source_term.interpolate(partial(source_term_expr, t=t0))  # type: ignore
    
    u_trial = ufl.TrialFunction(V)
    u_test = ufl.TestFunction(V)
    f_form = (
        (u_trial - u_old) / dt * u_test * ufl.dx  # type: ignore
        + ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx  # type: ignore
        - source_term * u_test * ufl.dx  # type: ignore
    )
    
    recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t12/explicit', name='ts')
    def record_func(t: float, t_step: int):
        t_count = int(t / dt)
        recorder.update_point_data(key='u', value=uh.x.array.copy())  # type: ignore
        recorder.save(step=t_count)

    uh = TSDolfinxProblemSolver.solve_explicit_euler_by_dolfinx(
        comm=domain.comm,
        f_form=f_form,
        uh=uh,  # type: ignore
        u_old=u_old,  # type: ignore
        bcs=[bc],
        mat_assemble_method='lift',
        is_linear=True,
        t0=t0,
        dt=dt,
        max_T=T,
        ksp_setting=ksp_option,
        source_term=source_term, # type: ignore
        interpolate_func=source_term_expr,
        post_step_func=record_func,
        with_residual=False,
    )
    
    # ---------- 7. Compute L² error ----------
    u_exact = dolfinx.fem.Function(V)
    u_exact.interpolate(partial(exact_solution, t=T))  # type: ignore

    error_L2 = dolfinx.fem.assemble_scalar(DolfinUtils.ufl_to_dolfin(
        ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx  # type: ignore
    ))
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2, op=MPI.SUM))

    # ---------- 8. Report ----------
    print("\n=== Diffusion test ===")
    print(f"Mesh: {nx}×{nx} triangles")
    print(f"Time interval: [0, {T}],  dt = {dt}")
    print(f"L² error at t = {T}: {error_L2:.3e}")

    # tolerance is generous for the coarse mesh; tighten when you refine
    tol = 2e-2 if nx <= 20 else 5e-3
    assert error_L2 < tol, f"Error {error_L2:.3e} exceeds tolerance {tol:.3e}"
    print("Test PASSED\n")


if __name__ == "__main__":
    # debug_implicit_ts()
    # debug_explicit_ts()
    # debug_explicit_dolfinx()
    pass
