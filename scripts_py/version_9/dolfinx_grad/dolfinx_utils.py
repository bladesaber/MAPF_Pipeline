import numpy as np
from typing import Union, Callable, List, Sequence, Literal

import dolfinx
from dolfinx.fem import petsc
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import create_unit_square

import ufl
from ufl.core import expr


class DolfinUtils(object):
    """
    ufl.TrialFunction: represents the unknown function to solve for
    ufl.TestFunction: represents the weight function(an arbitrary function) in the weak formulation
    """

    @staticmethod
    def ufl_to_dolfin(form: ufl.form.Form):
        return dolfinx.fem.form(form)

    @staticmethod
    def create_expression(form: expr, V: dolfinx.fem.FunctionSpace):
        exp = dolfinx.fem.Expression(form, V.element.interpolation_points())
        return exp

    @staticmethod
    def derivative(form: ufl.Form, coefficient: ufl.Coefficient, direction: Union[ufl.Coefficient, ufl.Argument]):
        """
        coefficient: can not be Argument(TrialFunction or TestFunction)
        derivative here is variation(泛函变分) rather than gradient.
        ufl.derivative(f, u, dire) = grad(f, u) * direction
            when direction is TrialFunction/TestFunction (当direction不是一个具体变分方向时)
            TrialFunction: mean second variation(Jacobian)，指对函数f相对于u的变分，为一阶梯度的equation，常用于梯度矩阵组建
            TestFunction: mean first variation which is residual equation(
                想象direction是一个变量,ufl.derivative(f, u, dire)是基于这个变量的等式
            )，这是一个式子
        """
        return ufl.derivative(form, coefficient, direction)

    @staticmethod
    def function_eval(uh: dolfinx.fem.Function, pcd_xyz: np.ndarray, cell_indices: List[int]):
        """
        Evaluate Function at points x, where x has shape (num_points, 3),
        and cells has shape (num_points,) and cell[i] is the index of the
        cell containing point x[i]. If the cell index is negative the
        point is ignored.
        """
        values = uh.eval(pcd_xyz, cell_indices)
        return values


class BoundaryUtils(object):
    @staticmethod
    def apply_boundary_to_vec(
            b_vec: PETSc.Vec, bcs: list[dolfinx.fem.DirichletBC], a_form: dolfinx.fem.Form,
            clean_vec: bool = False, scalar_forward=True, method='lift'
    ):
        assert method in ['Identity_row', 'lift']
        if clean_vec:
            with b_vec.localForm() as loc_b:
                loc_b.set(0)

        if method == 'lift':
            dolfinx.fem.apply_lifting(b_vec, [a_form], [bcs])
            b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b_vec, bcs)

        if scalar_forward:
            b_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    @staticmethod
    def apply_boundary_to_matrix_explicit(a_mat: PETSc.Mat, bcs: list[dolfinx.fem.DirichletBC]):
        """
        But it will cause matrix become unSymmetric
        """
        for bc in bcs:
            dofs, _ = bc._cpp_object.dof_indices()
            # zeroRowsLocal: zero selected row of matrix and insert scalar value to diag
            a_mat.zeroRowsLocal(dofs, diag=1.0)
        # print(f"[Debug]apply_boundary_to_matrix_explicit: is A_mat Symmetric{a_mat.isSymmetric(1e-15)}")

    @staticmethod
    def define_dirichlet_cell(
            V: dolfinx.fem.functionspace, dofs: np.ndarray,
            bc_value: Union[Callable, float, np.ndarray, dolfinx.fem.Function]
    ):
        assert dofs.shape[0] > 0
        if isinstance(bc_value, Callable):
            u_d = dolfinx.fem.Function(V)
            u_d.interpolate(bc_value)
            dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs)
        elif isinstance(bc_value, dolfinx.fem.Function):
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs)
        elif isinstance(bc_value, np.ndarray):
            u_d = dolfinx.fem.Function(V)
            u_d.x.array[:] = bc_value
            dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs)
        else:
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=V)
        return {'type': 'dirichlet', 'boundary': dirichlet_bc}

    @staticmethod
    def create_homogenize_bc(V: dolfinx.fem.functionspace, dofs: np.ndarray, value_type=None):
        if value_type is not None:
            if isinstance(value_type, float):
                return dolfinx.fem.dirichletbc(0.0, dofs, V)
            elif isinstance(value_type, np.ndarray):
                new_value = dolfinx.fem.Function(V)
                new_value.x.array[:] = value_type
                return dolfinx.fem.dirichletbc(new_value, dofs, V)
            elif isinstance(value_type, dolfinx.fem.Function):
                new_value = dolfinx.fem.Function(value_type.function_space)
                new_value.x.array[:] = 0.0
                # todo 不知为什么当V为不是类似流体那种复合空间时会报错，必须使用None
                return dolfinx.fem.dirichletbc(new_value, dofs, V)
            else:
                raise NotImplementedError
        else:
            new_value = dolfinx.fem.Function(V)
            new_value.x.array[:] = 0.0
            return dolfinx.fem.dirichletbc(new_value, dofs, V)


class AssembleUtils(object):
    @staticmethod
    def assemble_mat(
            a_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC],
            a_mat: PETSc.Mat = None, method='lift', check_symmetric=False, diagonal=1.0,
    ):
        """
        assemble_matrix 用于组建a_form(Bilinear)的中间矩阵,即Ax=b(的A矩阵)
        a_form必须包含TrialFunction和TestFunction
        reference: https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html

        Ref: https://jsdokken.com/FEniCS23-tutorial/src/lifting.html
        Identity_row: 这是原始方法，但会导致A_mat不在对称，致使求解困难，很大部分的求解器不再适用
        """
        assert method in ['Identity_row', 'lift']

        if method == 'Identity_row':
            if a_mat is None:
                a_mat = dolfinx.fem.petsc.assemble_matrix(a_form, diagonal=diagonal)
            else:
                dolfinx.fem.petsc.assemble_matrix_mat(a_mat, a_form, diagonal=diagonal)
            a_mat.assemble()
            BoundaryUtils.apply_boundary_to_matrix_explicit(a_mat, bcs)

        elif method == 'lift':
            if a_mat is None:
                a_mat = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs, diagonal=diagonal)
            else:
                dolfinx.fem.petsc.assemble_matrix_mat(a_mat, a_form, bcs=bcs, diagonal=diagonal)
            a_mat.assemble()

        else:
            raise NotImplementedError

        if check_symmetric:
            print(f"[Debug AssembleUtils] A_mat Symmetric: {a_mat.isSymmetric(1e-15)}")

        return a_mat

    @staticmethod
    def assemble_vec(L_form: dolfinx.fem.Form, b_vec: PETSc.Vec = None, clear_vec=False) -> PETSc.Vec:
        """
        assemble_matrix用于计算L_form在FunctionSpace中每一个element的积分值
        L_form必须包含TestFunction
        """
        if b_vec is None:
            b_vec = dolfinx.fem.petsc.assemble_vector(L_form)
            # b_vec = dolfinx.fem.assemble_vector(L_form)
        else:
            if clear_vec:
                with b_vec.localForm() as loc:
                    loc.set(0)
            # dolfinx.fem.petsc._assemble_vector_vec
            dolfinx.fem.petsc.assemble_vector(b_vec, L_form)
        return b_vec

    @staticmethod
    def assemble_form_vec(
            l_form: ufl.Form, function_space: dolfinx.fem.FunctionSpace,
            b_vec: PETSc.Vec = None, clear_vec=False
    ) -> PETSc.Vec:
        """
        l_form must without ufl.dx or ufl.ds
        function_space: scalar function space or vector function space
        """
        agg_form = ufl.inner(l_form, ufl.TestFunction(function_space)) * ufl.dx
        return AssembleUtils.assemble_vec(dolfinx.fem.form(agg_form), b_vec=b_vec, clear_vec=clear_vec)

    @staticmethod
    def assemble_scalar(form: dolfinx.fem.Form):
        """计算form的积分值,form必须是积分式"""
        return dolfinx.fem.assemble_scalar(form)

    @staticmethod
    def create_vector(L_form: dolfinx.fem.Form):
        """only return 0 vector"""
        return dolfinx.fem.petsc.create_vector(L_form)

    @staticmethod
    def create_matrix(a_form: dolfinx.fem.Form):
        """only return 0 matrix"""
        return dolfinx.fem.petsc.create_matrix(a_form)


def debug():
    from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj

    mesh = MeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model_01_2D.msh', dim=2
    )

    V = dolfinx.fem.FunctionSpace(mesh.domain, ("CG", 1))

    inlet_entity_idxs = MeshUtils.extract_entity_indices(
        mesh.domain, style='facet', marker=13, mesh_tags=mesh.facet_tags
    )
    inlet_dofs = MeshUtils.extract_entity_dofs(V, mesh.facet_dim, inlet_entity_idxs)

    u = dolfinx.fem.Function(V, name='t')
    value_t = np.ones(u.x.array.shape[0])
    dolfinx.fem.dirichletbc(value_t, inlet_dofs, V)


if __name__ == '__main__':
    debug()
