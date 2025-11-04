import os
from typing import Literal

import dolfinx
import ufl
from ufl import inner, grad, dot

import numpy as np
import pyvista

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction

from thirdparty.pyadjoint.pyadjoint.tape import no_annotations

from scripts_py.dolfinx_utils import AssembleUtils, DolfinUtils
from scripts_py.mesh_utils import MeshUtils
from scripts_py.sparse_utils import PetscUtils
from scripts_py.dolfinx_utils import BoundaryUtils
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.mesh_utils import MeshObj

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary


def draw_shape_derivative(
    grid: pyvista.UnstructuredGrid, 
    shape_grad: np.ndarray, 
    scale=0.1,
    plotter: pyvista.Plotter|None = None,
    title='vis'
):
    if shape_grad.shape[1] == 2:
        shape_grad = np.concatenate([shape_grad, np.zeros((shape_grad.shape[0], 1))], axis=1)
    shape_grad = shape_grad / np.max(np.linalg.norm(shape_grad, ord=2, axis=1)) * scale
    
    if plotter is None:
        show = True
        plotter = pyvista.Plotter()
    else:
        show = False
        
    if show:
        plotter.add_mesh(grid, color='blue', style='wireframe')
    plotter.add_arrows(grid.points, shape_grad, color='red', opacity=1.0, mag=1.0)
    if show:
        plotter.show(title=title)

    
def draw_deform(
        geo_origin: pyvista.UnstructuredGrid, 
        geo_deform: pyvista.UnstructuredGrid, 
        geo_ref: pyvista.UnstructuredGrid|None = None,
        plotter: pyvista.Plotter|None = None,
        title='vis'
    ):
    if plotter is None:
        show = True
        plotter = pyvista.Plotter()
    else:
        show = False
        
    plotter.add_mesh(geo_origin, color='blue', style='wireframe')
    plotter.add_mesh(geo_deform, color='green', style='wireframe')
    if geo_ref is not None:
        plotter.add_mesh(geo_ref, color='red', style='wireframe')
                
    deform_vec = geo_deform.points - geo_origin.points
    plotter.add_arrows(geo_origin.points, deform_vec, color='red', opacity=0.2, mag=1.0)
    if show:
        plotter.show(title=title)


class NormalDerivativeRegularization(object):
    '''Solver class to regularize the shape derivatives as explained in
        Frédéric de Gournay
        Velocity Extension for the Level-set Method and Multiple Eigenvalues in Shape Optimization
        SIAM J. Control Optim., 45(1), 343–367. (25 pages)
    '''
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    @no_annotations
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        boundaries: dict[str, AutoFluidBoundary],
        beta: float = 1.0,
        gamma: float = 1.0e4,
        mat_assemble_method: Literal['identity_row', 'lift'] = 'lift'
    ):
        self.name = name
        self.mesh = mesh
        self.domain = self.mesh.domain
        self.boundaries: dict[str, AutoFluidBoundary] = boundaries
        
        self.point_scalar_func_space = DolfinUtils.point_scalar_function_space(self.domain)
        coordinate_space = self.domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.functionspace(self.domain, coordinate_space)
                
        self.bcs = []
        for name, bry_info in self.boundaries.items():
            if bry_info.is_free:
                continue
            
            bc_zero = dolfinx.fem.Function(self.deformation_space, name=f'{name}_control')
            self.bcs.append(
                BoundaryUtils.define_dirichlet_bc(
                    bc_func_space=self.deformation_space,
                    value_func_space=self.deformation_space,
                    dofs=MeshUtils.extract_entity_dofs(
                        self.deformation_space, self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=bc_zero,  # type: ignore
                    is_mixed_element=False,
                )
            )
                
        self.n_vec = MeshUtils.define_facet_norm(self.domain)
        self.ds = MeshUtils.define_ds(self.domain, facets_tag=self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.domain)
        
        hmin = self.max_mesh_diameter(self.domain)
        if beta > 20.0 * hmin:
            raise ValueError(
                f"Length scale parameter beta ({beta}) is much larger than the mesh size {hmin}"
            )
        
        self.beta: float = beta
        self.gamma: float = gamma
        
        self.beta_weight = dolfinx.fem.Constant(self.domain, self.beta)  # type: ignore
        self.gamma_weight = dolfinx.fem.Constant(self.domain, self.gamma)  # type: ignore
        
        der_trial = ufl.TrialFunction(self.deformation_space)
        der_test = ufl.TestFunction(self.deformation_space)

        # convert shape derivative to Sobolev function space H1 riesz representation
        self.a_form = (
            self.beta_weight * inner(grad(der_trial), grad(der_test)) * ufl.dx  # type: ignore
            + inner(der_trial, der_test) * ufl.dx
        )
        
        # TODO: check if this is correct ?
        # regularize tangential motion of surface to zero
        self.a_form += self.gamma_weight * (
            inner(dot(der_trial, self.n_vec), dot(der_test, self.n_vec))  # type: ignore
        ) * ufl.ds
        
        self.mat_assemble_method = mat_assemble_method
        self.a_mat = AssembleUtils.assemble_mat(self.a_form, self.bcs, method=mat_assemble_method)
        
    @staticmethod
    def max_mesh_diameter(domain: dolfinx.mesh.Mesh) -> float:
        """Calculate maximum cell diameter (longest distance between any two points) in mesh
        """
        DG0 = DolfinUtils.cell_scalar_function_space(domain, degree=0)
        a_form = ufl.CellDiameter(domain) / ufl.CellVolume(domain) * ufl.TestFunction(DG0) * ufl.dx  # type: ignore
        res = AssembleUtils.assemble_vec(a_form)
        return np.max(res.array)

    @no_annotations
    def solve(
        self, 
        dJ_dmesh: np.ndarray,
        ksp_option: dict|None = None,
        with_residual=False
    ) -> np.ndarray:
        assert dJ_dmesh.ndim == 1, "shape_derivative_original must be a 1D array"
        
        b_vec = PetscUtils.create_vec(size=dJ_dmesh.shape[0])
        b_vec.array[:] = dJ_dmesh
        BoundaryUtils.apply_boundary_to_vec(
            b_vec, 
            self.bcs, 
            DolfinUtils.ufl_to_dolfin(self.a_form), 
            clean_vec=False, 
            method=self.mat_assemble_method
        )
        
        solver = LinearProblemSolver.create_petsc_solver(
            comm = self.domain.comm, 
            solver_setting = ksp_option if ksp_option is not None else self.ksp_option, 
            mat = self.a_mat
        )
        res_dict = LinearProblemSolver.solve_by_petsc(
            b_vec=b_vec, 
            a_mat=self.a_mat, 
            solver=solver, 
            setOperators=False, 
            with_residual=with_residual
        )
        if not res_dict['is_solved']:
            raise ValueError(f"Failed to solve derivative regularization problem.")
        return res_dict['res'].array


class SmoothDerivativeRegularization(object):
    mu_ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    disp_ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    @no_annotations
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        boundaries: dict[str, AutoFluidBoundary],
        mat_assemble_method: Literal['identity_row', 'lift'] = 'lift',
        mu_max: float = 500.0,
        mu_min: float = 1.0,
    ):
        self.name = name
        self.mat_assemble_method = mat_assemble_method
        
        self.mesh = mesh
        self.domain = self.mesh.domain
        self.boundaries: dict[str, AutoFluidBoundary] = boundaries
        
        self.point_scalar_func_space = DolfinUtils.point_scalar_function_space(self.domain)
        coordinate_space = self.mesh.domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.functionspace(self.domain, coordinate_space)
        self.cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(self.mesh.domain)
        
        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)
        
        self.mu = dolfinx.fem.Function(self.point_scalar_func_space, name='mu')
        self.mu_bcs = []
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                self.mu_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=self.point_scalar_func_space,
                        value_func_space=self.point_scalar_func_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.point_scalar_func_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=mu_max,
                        is_mixed_element=False
                    )
                )
                    
            else:
                self.mu_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=self.point_scalar_func_space,
                        value_func_space=self.point_scalar_func_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.point_scalar_func_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=mu_min,
                        is_mixed_element=False
                    )
                )
        
        self.disp_bcs = []
        for bry_info in self.boundaries.values():
            if not bry_info.is_free:
                disp_zero = dolfinx.fem.Function(self.deformation_space, name=f'disp_zero_{bry_info.name}')
                disp_zero.x.array[:] = 0.0  # type: ignore
                self.disp_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=self.deformation_space,
                        value_func_space=self.deformation_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.deformation_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=disp_zero,
                        is_mixed_element=False
                    )
                )
    
    @no_annotations 
    def compute_mu(self, mat_assemble_method: Literal['identity_row', 'lift'] = 'lift', ksp_option: dict|None = None):
        mu_trial = ufl.TrialFunction(self.point_scalar_func_space)
        mu_test = ufl.TestFunction(self.point_scalar_func_space)
            
        mu_a_form = -inner(grad(mu_trial), grad(mu_test)) * ufl.dx  # type: ignore
        mu_L_form = dolfinx.fem.Constant(self.domain, 0.0) * mu_test * ufl.dx  # type: ignore

        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.domain.comm,
            uh=self.mu,  # type: ignore
            a_form=mu_a_form,
            l_form=mu_L_form,
            bcs=self.mu_bcs,
            ksp_option=ksp_option if ksp_option is not None else self.mu_ksp_option,
            mat_assemble_method=mat_assemble_method,
            with_residual=False,
        )
        if not res_dict['is_solved']:
            raise ValueError('Failed to solve mu problem.')

        # # ------ debug ------
        # self.mesh.grid.point_data['mu'] = self.mu.x.array  # type: ignore
        # self.mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t13/mu.vtu', binary=True)
        # # -------------------
    
        return self.mu.x.array  # type: ignore
    
    @no_annotations
    def solve(
        self, 
        dJ_dmesh: np.ndarray, 
        mat_assemble_method: Literal['identity_row', 'lift'] = 'lift', 
        ksp_option: dict|None = None,
        method: Literal['simple', 'mu_weight'] = 'mu_weight'
    ):
        disp = dolfinx.fem.Function(self.deformation_space, name='disp')
        disp.x.array[:] = dJ_dmesh.reshape(-1)  # type: ignore
        
        disp_u = ufl.TrialFunction(self.deformation_space)
        disp_v = ufl.TestFunction(self.deformation_space)
        
        if method == 'simple':
            # H1 riesz representation
            a_form = (
                inner(grad(disp_u), grad(disp_v)) * ufl.dx 
                + inner(disp_u, disp_v) * ufl.dx
            )
            L_form = inner(disp, disp_v) * ufl.dx
        
        elif method == 'mu_weight':
            # H1 riesz representation with mu weight
            a_form = (
                self.mu * inner(grad(disp_u), grad(disp_v)) * ufl.dx  # type: ignore
                + self.mu * inner(disp_u, disp_v) * ufl.dx  # type: ignore
            )
            
            L_form = 0.0
            for bry_info in self.boundaries.values():
                if bry_info.is_free:
                    L_form += inner(disp, disp_v) * self.ds(bry_info.marker)  # type: ignore
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.domain.comm,
            uh=disp,  # type: ignore
            a_form=a_form,
            l_form=L_form,  # type: ignore
            bcs=self.disp_bcs,
            ksp_option=ksp_option if ksp_option is not None else self.disp_ksp_option,
            mat_assemble_method=mat_assemble_method,
            with_residual=False,
        )
        if not res_dict['is_solved']:
            raise ValueError('Failed to solve mu problem.')
        return disp.x.array  # type: ignore


class StiffDerivativeRegularization(SmoothDerivativeRegularization):
    @no_annotations
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        boundaries: dict[str, AutoFluidBoundary],
        mat_assemble_method: Literal['identity_row', 'lift'] = 'lift',
        mu_max: float = 500.0,
        mu_min: float = 1.0,
    ):
        super().__init__(name, mesh, boundaries, mat_assemble_method, mu_max, mu_min)

    @staticmethod
    def epsilon(u):
        return ufl.sym(grad(u))

    @staticmethod
    def sigma(u, mu, lmb=0):
        return (
            2 * mu * StiffDerivativeRegularization.epsilon(u) 
            + lmb * ufl.tr(StiffDerivativeRegularization.epsilon(u)) * ufl.Identity(2)  # type: ignore
        )
    
    @no_annotations
    def solve(
        self, 
        dJ_dmesh: np.ndarray, 
        mat_assemble_method: Literal['identity_row', 'lift'] = 'lift', 
        ksp_option: dict|None = None
    ):
        disp = dolfinx.fem.Function(self.deformation_space, name='disp')
        disp.x.array[:] = dJ_dmesh.reshape(-1)  # type: ignore
        
        disp_u = ufl.TrialFunction(self.deformation_space)
        disp_v = ufl.TestFunction(self.deformation_space)
        
        # linear elastic equation
        disp_a_form = (
            inner(self.sigma(disp_u, mu=self.mu), grad(disp_v)) * ufl.dx
        )
        
        disp_L_form = 0.0
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                disp_L_form += inner(disp, disp_v) * self.ds(bry_info.marker)  # type: ignore
        
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.domain.comm,
            uh=disp,  # type: ignore
            a_form=disp_a_form,
            l_form=disp_L_form,  # type: ignore
            bcs=self.disp_bcs,
            ksp_option=ksp_option if ksp_option is not None else self.disp_ksp_option,
            mat_assemble_method=mat_assemble_method,
            with_residual=False,
        )
        if not res_dict['is_solved']:
            raise ValueError('Failed to solve mu problem.')
        return disp.x.array  # type: ignore

