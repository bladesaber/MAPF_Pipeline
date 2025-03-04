import numpy as np
import dolfinx
from typing import Dict, List
import ufl
from petsc4py import PETSc

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshObj, MeshUtils
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import t_grad, t_div
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils


class CurvatureGrad(object):
    # Todo 失败方法，原因不明，这个方法放到之后再解决
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }

    def __init__(
            self,
            mesh: MeshObj,
            weight: float,
            ksp_option_curvature: Dict = None,
            ksp_option_norm: Dict = None
    ):
        self.mesh = mesh
        self.weight = weight
        self.ksp_option_curvature = ksp_option_curvature if ksp_option_curvature is not None else self.ksp_option
        self.ksp_option_norm = ksp_option_norm if ksp_option_norm is not None else self.ksp_option

        self.vector_point_functionSpace = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("CG", 1))
        self.scalar_point_functionSpace = dolfinx.fem.FunctionSpace(self.mesh.domain, ("CG", 1))

        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.coord = MeshUtils.define_coordinate(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain)
        self.test_v = ufl.TestFunction(self.vector_point_functionSpace)

        self.kappa_curvature = dolfinx.fem.Function(self.vector_point_functionSpace)
        self.n_vec_fun = dolfinx.fem.Function(self.vector_point_functionSpace)

        self.a_curvature_form = ufl.inner(ufl.TrialFunction(self.vector_point_functionSpace), self.test_v) * self.ds
        self.l_curvature_form = ufl.inner(t_grad(self.coord, self.n_vec), t_grad(self.test_v, self.n_vec)) * self.ds

        self.a_norm_form = ufl.inner(
            ufl.TrialFunction(self.vector_point_functionSpace),
            ufl.TestFunction(self.vector_point_functionSpace)
        ) * self.ds
        self.l_norm_form = ufl.inner(self.n_vec, ufl.TestFunction(self.vector_point_functionSpace)) * self.ds

        self.loss_form = ufl.inner(self.kappa_curvature, self.kappa_curvature) * self.ds

        self.shape_form = ufl.derivative(self.weight * self.loss_form, self.coord, self.test_v)
        # identity = ufl.Identity(self.mesh.domain.geometry.dim)
        # coord_tang = t_grad(self.coord, self.n_vec)
        # self.shape_form = (
        #         self.weight *
        #         ufl.inner(
        #             (identity - (coord_tang + coord_tang.T)) * t_grad(self.test_v, self.n_vec),
        #             t_grad(self.kappa_curvature, self.n_vec)
        #         ) * self.ds
        #         + 0.5 * t_div(self.test_v, self.n_vec) * t_div(self.kappa_curvature, self.n_vec) * self.ds
        # )

        self.fix_dofs: np.ndarray = np.array([])
        self.grad_vec: np.ndarray = None
        self.is_updated = False

    def set_fix_boundary(self, fix_dofs: np.ndarray):
        self.fix_dofs = fix_dofs

    def compute_shape_derivative(self, max_step: float):
        if self.is_updated:
            grad_vec_petsc = AssembleUtils.assemble_vec(dolfinx.fem.form(self.shape_form))
            self.grad_vec = grad_vec_petsc.array.reshape((-1, self.mesh.topology_dim))
            if self.fix_dofs.shape[0] > 0:
                self.grad_vec[self.fix_dofs, :] = 0.0
            n_vec = self.n_vec_fun.x.array.reshape((-1, self.mesh.topology_dim))
            self.grad_vec = np.sum(self.grad_vec * n_vec, axis=1, keepdims=True) * n_vec
            # self.grad_vec = self.grad_vec / (np.max(np.linalg.norm(self.grad_vec, axis=1)) + 1e-8) * max_step

        else:
            raise ValueError('[Error]: compute shape derivative without updating.')

    def compute_loss(self):
        if self.is_updated:
            curvature_val = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.loss_form))
            loss_curvature = 0.5 * self.weight * curvature_val
            return loss_curvature
        else:
            raise ValueError('[Error]: compute loss without updating.')

    def _compute_curvature(self, with_debug=False):
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.kappa_curvature,
            a_form=dolfinx.fem.form(self.a_curvature_form),
            l_form=dolfinx.fem.form(self.l_curvature_form),
            bcs=[],
            mat_assemble_method='lift',
            ksp_option=self.ksp_option_curvature,
            with_debug=with_debug
        )
        return res_dict['is_solved']

    def _compute_surface_norm_vec(self):
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.n_vec_fun,
            a_form=dolfinx.fem.form(self.a_norm_form),
            l_form=dolfinx.fem.form(self.l_norm_form),
            bcs=[],
            mat_assemble_method='lift',
            ksp_option=self.ksp_option_norm,
            with_debug=True
        )
        if res_dict['is_solved']:
            n_vec = self.n_vec_fun.x.array.reshape((-1, self.mesh.topology_dim))
            n_vec = n_vec / (np.linalg.norm(n_vec, axis=1, keepdims=True) + 1e-8)
            self.n_vec_fun.x.array[:] = n_vec.reshape(-1)

        return res_dict['is_solved']

    def update(self, with_debug=False):
        self.is_updated = False
        is_valid = self._compute_curvature(with_debug=with_debug)
        if not is_valid:
            print('[Error]: compute curvature fail.')
            return False

        is_valid = self._compute_surface_norm_vec()
        if not is_valid:
            print('[Error]: compute surface norm vector fail.')
            return False

        self.is_updated = True
        return True

    def draw_problem(self, with_grad=True, with_surface_norm=True, vis: VisUtils = None, **kwargs):
        if vis is None:
            vis = VisUtils()
            show_vis = True

        vis.plot(self.mesh.grid, style='wireframe')

        if with_grad:
            grad_vec = self.grad_vec.reshape((-1, self.mesh.topology_dim))
            if self.mesh.topology_dim == 2:
                grad_vec = np.concatenate([grad_vec, np.zeros(shape=(grad_vec.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.points, grad_vec, mag=kwargs.get('grad_mag', 1.0), color=(0., 0., 1.), opacity=0.25)

        if with_surface_norm:
            n_vec_np = self.n_vec_fun.x.array.reshape((-1, self.mesh.topology_dim))
            if self.mesh.topology_dim == 2:
                n_vec_np = np.concatenate([n_vec_np, np.zeros(shape=(n_vec_np.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.points, n_vec_np, mag=kwargs.get('norm_mag', 1.0), color=(0., 1., 0.), opacity=0.5)

        if show_vis:
            vis.show()


def debug():
    mesh = MeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/model.msh', dim=2
    )
    grader = CurvatureGrad(mesh, weight=1.0)
    grader.update()
    grader.compute_shape_derivative(max_step=0.1)
    grader.draw_problem(
        with_grad=True, with_surface_norm=True, norm_mag=0.1, grad_mag=0.001
    )


if __name__ == '__main__':
    debug()
