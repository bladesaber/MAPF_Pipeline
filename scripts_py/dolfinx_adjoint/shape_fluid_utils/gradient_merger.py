import dolfinx
import ufl

import numpy as np
import pyvista
from ufl.sobolevspace import L2

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.dolfinx_adjoint.shape_fluid_utils.shape_fluid import FluidPipeBoundary
from scripts_py.eqn_solver import LinearProblemSolver


class ElasticShapeMerger(object):
    """
    Only used for Shape Optimization.
    Mesh Network构建弹性体需要考虑每个node受到的neighbours的Spring Force以及扭转力矩,导致问题变得很复杂.
    因此考虑将该问题转化为一个在弹性刚体下的有限元问题,shape gradient作为outer force
    """
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    @staticmethod
    def epsilon(u):
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        # return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)  # type: ignore

    @staticmethod
    def sigma(u, lambda_v, mu):
        return lambda_v * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * ElasticShapeMerger.epsilon(u)

    def __init__(self, mesh: MeshObj, ksp_option:dict = ksp_option):
        self.mesh: MeshObj = mesh
        
        self.point_vector_functionSpace = DolfinUtils.point_vector_function_space(self.mesh.domain)
        self.point_scalar_functionSpace = DolfinUtils.point_scalar_function_space(self.mesh.domain)
        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)
        
        self.control_boundry_infos: dict[str, FluidPipeBoundary] = {}
        self.bc_free_dofs: np.ndarray
        self.bc_fix_dofs: np.ndarray
        self.inner_dofs: np.ndarray
        self.n_fix_points: int
        self.n_free_points: int
        self.ksp_option = ksp_option
        
        self.u = ufl.TrialFunction(self.point_vector_functionSpace)
        self.v = ufl.TestFunction(self.point_vector_functionSpace)
        self.uh = dolfinx.fem.Function(self.point_vector_functionSpace, name='displacement')
        self.f = dolfinx.fem.Function(self.point_vector_functionSpace, name='shape_gradient')
        
        self.gradient_dict: dict[str, np.ndarray] = {}

    def add_control_boundary(self, name: str, boundary: FluidPipeBoundary):
        self.control_boundry_infos[name] = boundary

    def define_dofs(self):
        bc_free_dofs = []
        bc_fix_dofs = []
        for _, bry_info in self.control_boundry_infos.items():
            dofs = MeshUtils.extract_entity_dofs(
                self.point_vector_functionSpace, self.mesh.fdim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', 
                    mesh_tags=self.mesh.facet_tags, 
                    marker=bry_info.marker
                )
            )
            if bry_info.is_free:
                bc_free_dofs.extend(dofs.tolist())
            else:
                bc_fix_dofs.extend(dofs.tolist())
        
        bc_free_dofs = list(set(bc_free_dofs) - set(bc_fix_dofs))
        inner_dofs = list(set(np.arange(0, self.mesh.num_points, 1)) - set(bc_fix_dofs) - set(bc_free_dofs))
        
        self.bc_free_dofs = np.array(bc_free_dofs).astype(np.int32)  # must convert to int32
        self.bc_fix_dofs = np.array(bc_fix_dofs).astype(np.int32)
        self.inner_dofs = np.array(inner_dofs).astype(np.int32)
        self.n_fix_points = self.bc_fix_dofs.shape[0]
        self.n_free_points = self.mesh.num_points - self.n_fix_points

    def update_grad(self, key: str, grad: np.ndarray):
        assert grad.shape == (self.mesh.num_points, self.mesh.tdim), f'[Error]: shape_gradient shape error.'
        self.gradient_dict[key] = grad

    def merge_norm_gradient(self, weight: dict={}):
        '''gradient is normalized to 0.0 ~ 1.0'''
        grad_agg = 0.0
        for name, grad in self.gradient_dict.items():
            grad_agg += grad[:, :self.mesh.tdim] * weight.get(name, 1.0)
        grad_agg = grad_agg / float(len(self.gradient_dict))
        grad_norm = np.linalg.norm(grad_agg, axis=1, ord=2)
        grad_agg = grad_agg / np.max(np.abs(grad_norm))
        return grad_agg

    def compute_obstacle_gradient(
        self, 
        obs_pcd: np.ndarray, 
        obs_radius: float, 
        max_displacement_step_langth: float,
        with_norm: bool = True
    ) -> dict:
        report_dict = self.mesh.detect_obstacle(
            obs_pcd=obs_pcd, 
            obs_radius=obs_radius, 
            unit_lr=max_displacement_step_langth
        )
        if not report_dict['is_conflict']:
            return report_dict
        
        conflict_mesh_idxs: np.ndarray = report_dict['conflict_mesh_idxs']
        conflict_obs_idxs: np.ndarray = report_dict['conflict_obs_idxs']
        
        obs2mesh_vec = (
            self.mesh.points[conflict_mesh_idxs, :self.mesh.tdim] 
            - obs_pcd[conflict_obs_idxs, :self.mesh.tdim]
        )
        obs2mesh_vec_length = np.linalg.norm(obs2mesh_vec, ord=2, axis=1)
        obs2mesh_vec_norm = obs2mesh_vec / obs2mesh_vec_length[:, np.newaxis]
        
        invert_scale = (max_displacement_step_langth - (obs2mesh_vec_length - obs_radius))
        obstacle_grad = obs2mesh_vec_norm * invert_scale[:, np.newaxis]
        
        # ------ union all gradient of each mesh point to every obstacle point
        obs_aggregate_grad = np.zeros(shape=(self.mesh.num_points, self.mesh.tdim))
        for mesh_idx in np.unique(conflict_mesh_idxs):
            obs_aggregate_grad[mesh_idx, :] = np.sum(obstacle_grad[conflict_mesh_idxs == mesh_idx], axis=0)
                
        if with_norm:
            # obs_aggregate_grad = obs_aggregate_grad / np.linalg.norm(obs_aggregate_grad, ord=2)
            obs_aggregate_grad = obs_aggregate_grad / np.max(np.abs(np.linalg.norm(obs_aggregate_grad, ord=2, axis=1)))
            obs_aggregate_grad = obs_aggregate_grad * max_displacement_step_langth
        
        return {
            'is_conflict': True,
            'obs_aggregate_grad': obs_aggregate_grad
        }
        
    def linear_elastic_shape_displacement(
        self,
        length_step: float,
        obs_pcd: np.ndarray|None = None,
        obs_radius: float = 0.0,
        stiff_cof: float=50.0,
        lambda_v: float = 1.25,
        mu: float = 1.0,
        with_residual=False
    ):
        '''
        # https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html
        # linear elastic equation is navier equation
        形变会受到类似Mesh外型的影响,导致部分区域不符合预期,
        '''
        # raise DeprecationWarning('This method is deprecated.')
        
        # ------ compute shape gradient
        norm_grad = self.merge_norm_gradient()
        shape_gradient = norm_grad * length_step
        
        # ------ compute obstacle gradient
        if obs_pcd is not None:
            assert obs_radius > 0.0, f'[Error]: obs_radius must be greater than 0.0.'
            obs_grad_dict = self.compute_obstacle_gradient(
                obs_pcd=obs_pcd, 
                obs_radius=obs_radius, 
                max_displacement_step_langth=length_step,
                with_norm=True  # TODO: should i use norm?
            )
            if obs_grad_dict['is_conflict']:
                obs_aggregate_grad: np.ndarray = obs_grad_dict['obs_aggregate_grad']
                shape_gradient += obs_aggregate_grad
        
        # ------ only apply shape gradient on free boundary
        shape_gradient[self.bc_fix_dofs, :] = 0.0
        shape_gradient[self.inner_dofs, :] = 0.0
        
        # ------ compute force
        force = shape_gradient * stiff_cof
        
        # ------ define linear elastic equation
        self.f.x.array[:] = force.reshape(-1).astype(float)  # type: ignore

        a_form = ufl.inner(
            self.sigma(self.u, lambda_v=lambda_v, mu=mu),
            self.epsilon(self.v)
        ) * self.dx
        
        T = dolfinx.fem.Constant(self.mesh.domain, tuple([0.] * self.mesh.tdim))
        L_form = (
            ufl.dot(self.f, self.v) * ufl.dx 
            + ufl.dot(T, self.v) * self.ds
        )
        
        report_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.uh,  # type: ignore
            a_form=a_form,
            l_form=L_form,
            bcs=[
                dolfinx.fem.dirichletbc(
                    np.zeros(shape=(self.mesh.tdim,)), self.bc_fix_dofs, self.point_vector_functionSpace
                )
            ],
            ksp_option=self.ksp_option,
            with_residual=with_residual
        )
        
        if report_dict['is_solved']:
            return {
                'is_success': True,
                'displacement': self.uh.x.array.reshape((-1, self.mesh.tdim)).copy(),  # type: ignore
            }
        else:
            return {
                'is_success': False,
                'max_residual': report_dict['max_error'],
                'converged_reason': report_dict['converged_reason'],
            }

    def smooth_shape_displacement(self, length_step: float, with_residual=False):
        '''
        smooth shape displacement based on poisson equation, where f(force) is boundary shape gradient
        '''
        norm_grad = self.merge_norm_gradient()
        shape_gradient = norm_grad * length_step
        
        # ------ only apply shape gradient on free boundary
        shape_gradient[self.bc_fix_dofs, :] = 0.0
        shape_gradient[self.inner_dofs, :] = 0.0
        
        # ------ define boundary condition
        bc_free_func = dolfinx.fem.Function(self.point_vector_functionSpace)
        bc_free_func.x.array[:] = shape_gradient.reshape(-1)  # type: ignore
        bc_free = dolfinx.fem.dirichletbc(bc_free_func, self.bc_free_dofs)  # type: ignore
        
        bc_fix = dolfinx.fem.dirichletbc(
            np.zeros(shape=(self.mesh.tdim,)), 
            self.bc_fix_dofs, 
            self.point_vector_functionSpace
        )

        # ------ define poisson equation weak form
        a_form = ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * self.dx
        L_form = ufl.inner(self.f, self.v) * self.dx
        
        report_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.uh,  # type: ignore
            a_form=a_form,
            l_form=L_form,
            bcs=[bc_fix, bc_free],
            ksp_option=self.ksp_option,
            with_residual=with_residual
        )
        
        if report_dict['is_solved']:
            return {
                'is_success': True,
                'displacement': self.uh.x.array.reshape((-1, self.mesh.tdim)).copy(),  # type: ignore
            }
        else:
            return {
                'is_success': False,
                'max_residual': report_dict['max_error'] if with_residual else None,
                'converged_reason': report_dict['converged_reason'],
            }

    def linear_elastic_obstacle_displacement(
        self, 
        length_step: float, 
        stiff_cof: float=50.0,
        obs_pcd: np.ndarray|None = None, 
        obs_radius: float = 0.0, 
        lambda_v: float = 1.25,
        mu: float = 1.0,
        with_residual=False
    ):
        '''
        only apply obstacle gradient when mesh point is close to obstacle point
        '''
        norm_grad = self.merge_norm_gradient()
        shape_gradient = norm_grad * length_step
        
        # ------ compute obstacle gradient
        if obs_pcd is not None:
            assert obs_radius > 0.0, f'[Error]: obs_radius must be greater than 0.0.'
            obs_grad_dict = self.compute_obstacle_gradient(
                obs_pcd=obs_pcd, 
                obs_radius=obs_radius, 
                max_displacement_step_langth=length_step,
                with_norm=True  # TODO: should i use norm?
            )
            if obs_grad_dict['is_conflict']:
                obs_aggregate_grad: np.ndarray = obs_grad_dict['obs_aggregate_grad']
                
                # ------ only apply obstacle gradient on free boundary
                obs_aggregate_grad[self.bc_fix_dofs, :] = 0.0
                obs_aggregate_grad[self.inner_dofs, :] = 0.0
                
                # ------ convert to force
                obstacle_force = obs_aggregate_grad * stiff_cof
                
                # ------ define linear elastic equation
                self.f.x.array[:] = obstacle_force.reshape(-1).astype(float)  # type: ignore
                
                a_form = ufl.inner(
                    self.sigma(self.u, lambda_v=lambda_v, mu=mu),
                    self.epsilon(self.v)
                ) * self.dx
                
                T = dolfinx.fem.Constant(self.mesh.domain, tuple([0.] * self.mesh.tdim))
                L_form = (
                    ufl.dot(self.f, self.v) * ufl.dx 
                    + ufl.dot(T, self.v) * self.ds
                )
                
                report_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=self.mesh.domain.comm,
                    uh=self.uh,  # type: ignore
                    a_form=a_form,
                    l_form=L_form,
                    bcs=[
                        dolfinx.fem.dirichletbc(
                            np.zeros(shape=(self.mesh.tdim,)), 
                            self.bc_fix_dofs, 
                            self.point_vector_functionSpace
                        )
                    ],
                    ksp_option=self.ksp_option,
                    with_residual=with_residual
                )
                if with_residual:
                    print(f"[Log] Stoke Simulator isSolved:{report_dict['is_solved']}, maxResidual:{report_dict['max_residual']}.")
                
                if not report_dict['is_solved']:
                    return {
                        'is_success': False,
                        'is_obstacle_conflict': True,
                        'max_residual': report_dict['max_error'] if with_residual else None,
                        'converged_reason': report_dict['converged_reason'],
                        'log': report_dict['log'],
                    }
                
                obs_gradient = self.uh.x.array.reshape((-1, self.mesh.tdim)).copy()  # type: ignore
                assert not np.isnan(obs_gradient).any(), f'[Error]: obs_gradient is nan.'
                
                agg_gradient = shape_gradient + obs_gradient
                shape_gradient = agg_gradient / np.max(np.abs(np.linalg.norm(agg_gradient, axis=1, ord=2))) * length_step
                
                return {
                    'is_success': True,
                    'is_obstacle_conflict': True,
                    'displacement': shape_gradient,
                }
        
        return {
            'is_success': True,
            'is_obstacle_conflict': False,
            'displacement': shape_gradient,
        }        

    def draw_gradient(
        self, 
        obs_pcd: np.ndarray|None = None, 
        obs_radius: float = 0.0, 
        length_step: float = 0.1,
        with_shape_gradient: bool = True,
        with_obstacle: bool = True,
        with_obstacle_gradient: bool = True,
        shape_gradient_mag: float=1.0,
        obs_gradient_mag: float=1.0
    ):
        norm_grad = self.merge_norm_gradient()
        shape_gradient = norm_grad * length_step

        obs_grad_dict = {}
        if obs_pcd is not None:
            assert obs_radius > 0.0, f'[Error]: obs_radius must be greater than 0.0.'
            obs_grad_dict = self.compute_obstacle_gradient(
                obs_pcd=obs_pcd, 
                obs_radius=obs_radius,
                max_displacement_step_langth=length_step,
                with_norm=False
            )
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh.grid, color='blue', style='wireframe')
        if with_obstacle:
            plotter.add_mesh(pyvista.PolyData(obs_pcd), color='black')
        if with_shape_gradient:
            if shape_gradient.shape[1] == 2:
                shape_gradient = np.concatenate((shape_gradient, np.zeros((shape_gradient.shape[0], 1))), axis=1)    
            plotter.add_arrows(self.mesh.points, shape_gradient, color='red', opacity=0.6, mag=shape_gradient_mag)
        if with_obstacle_gradient:
            obs_aggregate_grad = obs_grad_dict['obs_aggregate_grad']
            if obs_aggregate_grad.shape[1] == 2:
                obs_aggregate_grad = np.concatenate(
                    (obs_aggregate_grad, np.zeros((obs_aggregate_grad.shape[0], 1))
                ), axis=1)
            plotter.add_arrows(self.mesh.points, obs_aggregate_grad, color='green', opacity=0.6, mag=obs_gradient_mag)
        plotter.show()

    def draw_boundary(self):
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh.grid, color='black', style='wireframe')
        
        bc_free_points = self.mesh.points[self.bc_free_dofs, :]
        if bc_free_points.shape[1] == 2:
            bc_free_points = np.concatenate((bc_free_points, np.zeros((bc_free_points.shape[0], 1))), axis=1)
        plotter.add_mesh(pyvista.PolyData(bc_free_points), color='green', style='points')
        
        bc_fix_points = self.mesh.points[self.bc_fix_dofs, :]
        if bc_fix_points.shape[1] == 2:
            bc_fix_points = np.concatenate((bc_fix_points, np.zeros((bc_fix_points.shape[0], 1))), axis=1)
        plotter.add_mesh(pyvista.PolyData(bc_fix_points), color='red', style='points')
        
        inner_points = self.mesh.points[self.inner_dofs, :]
        if inner_points.shape[1] == 2:
            inner_points = np.concatenate((inner_points, np.zeros((inner_points.shape[0], 1))), axis=1)
        plotter.add_mesh(pyvista.PolyData(inner_points), color='blue', style='points')
        
        plotter.show()
