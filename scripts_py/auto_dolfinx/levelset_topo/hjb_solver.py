from typing import Literal

import dolfinx
import ufl

import numpy as np
import petsc4py.PETSc as PETSc

from scripts_py.eqn_ts_solver import TSProblemSolver


class HamiltonJacobiSolver(object):
    def __init__(
        self, 
        domain: dolfinx.mesh.Mesh, 
        velocity: dolfinx.fem.Function,
        phi: dolfinx.fem.Function,
        with_diffusion: bool = True,
        diffusion_coefficient: float = 1e-4,
        with_GLS_stabilization: bool = True,
    ):
        self.domain = domain
        self.velocity = velocity
        self.phi: dolfinx.fem.Function = dolfinx.fem.Function(phi.function_space, name='phi')  # type: ignore
        self.phi.x.array[:] = phi.x.array.copy()  # type: ignore
        self.phi_dot: dolfinx.fem.Function = dolfinx.fem.Function(phi.function_space, name='phi_dot')  # type: ignore
        
        self.with_diffusion = with_diffusion
        self.diffusion_coefficient = diffusion_coefficient
        self.with_GLS_stabilization = with_GLS_stabilization
        
        self.f_form = self.create_hjb_equation_form()
    
    def create_hjb_equation_form(self):
        phi_test = ufl.TestFunction(self.phi.function_space)
        
        # Level-set evolution equation (Hamilton-Jacobi equation)
        f_form = (
            self.phi_dot * phi_test * ufl.dx  # type: ignore
            + ufl.inner(self.velocity, ufl.grad(self.phi)) * phi_test * ufl.dx  # type: ignore
        )
        '''
        f_form is standard level set evolution equation plus diffusion term: 
        $\\frac{\\partial \\phi}{\\partial t} + \\boldsymbol{\\theta} \\cdot \\nabla \\phi = \\kappa \\nabla^2 \\phi$
        The diffusion term can optimize the value transfer of velocity on the level set, reducing the dependence on Reinitialization.
        diffusion_coefficient must be very small, therefore the advection term still dominates.
        '''
        if self.with_diffusion:
            f_form += self.diffusion_coefficient * ufl.inner(
                ufl.grad(self.phi), ufl.grad(phi_test)
            ) * ufl.dx  # type: ignore
            
        if self.with_GLS_stabilization:
            h = ufl.CellDiameter(self.domain)
            residual_form = (
                self.phi_dot
                + ufl.dot(self.velocity, ufl.grad(self.phi))  # type: ignore
                - self.diffusion_coefficient * ufl.div(ufl.grad(self.phi))  # type: ignore
            )
            
            beta_gls = 0.9
            tau_gls = beta_gls * (
                (4.0 * ufl.dot(self.velocity, self.velocity) / h ** 2)  # type: ignore
                + 9.0 * (4.0 * self.diffusion_coefficient / h ** 2) ** 2  # type: ignore
            ) ** (-0.5)
            
            adjoint_op = (
                ufl.dot(self.velocity, ufl.grad(phi_test)) 
                - self.diffusion_coefficient * ufl.div(ufl.grad(phi_test))  # type: ignore
            )
            
            f_form += tau_gls * ufl.inner(residual_form, adjoint_op) * ufl.dx
        
        return f_form
    
    def solve(
        self, 
        dt: float,
        run_steps: int,
        ksp_option: dict,
        snes_option: dict,
        mat_assemble_method: Literal['identity_row', 'lift'],
        only_solver: bool = False,
    ) -> np.ndarray|PETSc.TS:  # type: ignore        
        res_dict = TSProblemSolver.solve_implicit_by_petsc(
            comm=self.domain.comm,
            f_form=self.f_form,
            f_jacobi_form=None,
            sigma=None,
            uh=self.phi,  # type: ignore
            u_dot=self.phi_dot,  # type: ignore
            bcs=[],
            mat_assemble_method=mat_assemble_method,
            ts_type='backward_euler',
            problem_type='nonlinear',
            t0=0.0,
            dt=dt,
            max_time=run_steps * dt,
            max_time_step=run_steps,
            ksp_setting=ksp_option,
            snes_setting=snes_option,
            source_term=None, # type: ignore
            interpolate_func=None,
            post_step_func=None,
            with_snes_monitor=False,
            only_solver=only_solver
        )
        if only_solver:
            return res_dict['solver']

        if not res_dict['is_solved']:
            raise ValueError(f"Failed to solve Hamilton-Jacobi equation.")
        
        return res_dict['res'].x.array
    
    def get_solution(self) -> np.ndarray:
        return self.phi.x.array
    