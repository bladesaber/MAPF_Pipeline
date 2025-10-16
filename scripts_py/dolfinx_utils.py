import numpy as np
from typing import Union, Callable, List, Sequence, Literal

import dolfinx
from dolfinx.fem import petsc as dolfinx_petsc
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import create_unit_square

import ufl
import ufl.form
from ufl.core import expr
from basix.ufl import element


class DolfinUtils(object):
    """
    ufl.TrialFunction: represents the unknown function to solve for
    ufl.TestFunction: represents the weight function(an arbitrary function) in the weak formulation
    """

    @staticmethod
    def ufl_to_dolfin(form: ufl.Form) -> dolfinx.fem.Form:
        return dolfinx.fem.form(form)  # type: ignore

    @staticmethod
    def create_expression(form: expr, V: dolfinx.fem.FunctionSpace):  # type: ignore
        return dolfinx.fem.Expression(form, V.element.interpolation_points())

    @staticmethod
    def derivative(
        form: ufl.Form,
        coefficient: ufl.Coefficient,
        direction: Union[ufl.Coefficient, ufl.Argument],
    ):
        """
        todo
        coefficient: can not be Argument(TrialFunction or TestFunction)
        derivative here is variation(泛函变分) rather than gradient.
        ufl.derivative(f, u, dire) = grad(f, u) * direction
            when direction is TrialFunction/TestFunction (当direction不是一个具体变分方向时)
            TrialFunction: mean second variation(Jacobian),指对函数f相对于u的变分,为一阶梯度的equation,常用于梯度矩阵组建
            TestFunction: mean first variation which is residual equation(
                想象direction是一个变量,ufl.derivative(f, u, dire)是基于这个变量的等式
            ),这是一个式子
        """
        return ufl.derivative(form, coefficient, direction)

    @staticmethod
    def function_eval(
        uh: dolfinx.fem.Function, pcd_xyz: np.ndarray, cell_indices: List[int]
    ):
        """
        evaluate function at points x, where x has shape (num_points, 3),
        and cells has shape (num_points,) and cell[i] is the index of the
        cell containing point x[i]. If the cell index is negative the
        point is ignored.
        """
        values = uh.eval(pcd_xyz, cell_indices)
        return values

    @staticmethod
    def tabulate_dof_coordinates(func_space: dolfinx.fem.FunctionSpace) -> np.ndarray:
        return func_space.tabulate_dof_coordinates()

    @staticmethod
    def point_vector_function_space(
        domain: dolfinx.mesh.Mesh, degree: int = 1
    ) -> dolfinx.fem.FunctionSpace:
        point_vector_element = element("CG", domain.basix_cell(), degree=degree, shape=(domain.topology.dim,))  # type: ignore
        return dolfinx.fem.functionspace(domain, point_vector_element)

    @staticmethod
    def point_scalar_function_space(
        domain: dolfinx.mesh.Mesh, degree: int = 1
    ) -> dolfinx.fem.FunctionSpace:
        point_scalar_element = element("CG", domain.basix_cell(), degree=degree)  # type: ignore
        return dolfinx.fem.functionspace(domain, point_scalar_element)

    @staticmethod
    def cell_vector_function_space(
        domain: dolfinx.mesh.Mesh, degree: int = 0
    ) -> dolfinx.fem.FunctionSpace:
        cell_vector_element = element("DG", domain.basix_cell(), degree=degree, shape=(domain.topology.dim,))  # type: ignore
        return dolfinx.fem.functionspace(domain, cell_vector_element)

    @staticmethod
    def cell_scalar_function_space(
        domain: dolfinx.mesh.Mesh, degree: int = 0
    ) -> dolfinx.fem.FunctionSpace:
        cell_scalar_element = element("DG", domain.basix_cell(), degree=degree)  # type: ignore
        return dolfinx.fem.functionspace(domain, cell_scalar_element)


class BoundaryUtils(object):
    @staticmethod
    def apply_boundary_to_vec(
        b_vec: PETSc.Vec,
        bcs: list[dolfinx.fem.DirichletBC],
        a_form: dolfinx.fem.Form,
        clean_vec: bool = False,
        scalar_forward=True,
        method="lift",
    ):
        assert method in ["identity_row", "lift"]
        if clean_vec:
            with b_vec.localForm() as loc_b:
                loc_b.set(0)

        if method == "lift":
            dolfinx.fem.apply_lifting(b_vec, [a_form], [bcs])
            b_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
        dolfinx.fem.set_bc(b_vec, bcs)

        if scalar_forward:
            b_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

    @staticmethod
    def apply_boundary_to_matrix_explicit(
        a_mat: PETSc.Mat, bcs: list[dolfinx.fem.DirichletBC]
    ):
        """
        the matrix is not symmetric
        reference: https://jsdokken.com/FEniCS23-tutorial/src/lifting.html
        """
        for bc in bcs:
            dofs, _ = bc._cpp_object.dof_indices()
            # zeroRowsLocal: zero selected row of matrix and insert scalar value to diag
            a_mat.zeroRowsLocal(dofs, diag=1.0)
        # print(f"[Debug]apply_boundary_to_matrix_explicit: is A_mat Symmetric{a_mat.isSymmetric(1e-15)}")

    @staticmethod
    def define_dirichlet_bc(
        bc_func_space: dolfinx.fem.FunctionSpace,
        value_func_space: dolfinx.fem.FunctionSpace|tuple[dolfinx.fem.FunctionSpace, dolfinx.fem.FunctionSpace],
        dofs: np.ndarray|list[np.ndarray],
        bc_value: Union[Callable, float, np.ndarray, dolfinx.fem.Function, dolfinx.fem.Constant],
        is_mixed_element: bool
    ) -> dolfinx.fem.DirichletBC:
        '''
        Args:
            - bc_func_space: boundary function space
            - value_func_space: value function space
            
            If value functin sapce is from mixed element, eg:
            ```python
                velocity_element = element("Lagrange", mesh.domain.basix_cell(), degree=2, shape=(mesh.tdim,))
                pressure_element = element("Lagrange", mesh.domain.basix_cell(), degree=1)
                mix_element = mixed_element([velocity_element, pressure_element])

                W = dolfinx.fem.functionspace(domain, mix_element)
                W0, W1 = W.sub(0), W.sub(1)
                V, V_to_W_dofs = W0.collapse()
                Q, Q_to_W_dofs = W1.collapse()
                
                # dofs is from mixed element function space
                dofs = MeshUtils.extract_entity_dofs(
                    (W0, V), mesh.fdim,
                    MeshUtils.extract_entity_indices(
                        mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=24
                    )
                )
                
                bc_value = dolfinx.fem.Function(V)  # comes from mixed element function space
                dirichlet_bc = dolfinx.fem.dirichletbc(bc_value, dofs, V=W0)  # must specify V=W0 since different from bc_func_space
            ```
            
            If value function space is from simple element, eg:
            ```python
                point_scalar_element = element("CG", mesh.domain.basix_cell(), degree=1)
                scalar_space = dolfinx.fem.functionspace(mesh.domain, point_scalar_element)
                
                # dofs is from same element function space
                dofs = MeshUtils.extract_entity_dofs(
                    scalar_space, mesh.fdim,
                    MeshUtils.extract_entity_indices(
                        mesh.domain, style='facet', marker=24, mesh_tags=mesh.facet_tags
                    )
                )
                
                bc_value = dolfinx.fem.Function(scalar_space)
                dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=None)  # V must be None since same element
                ```
            
            If bc_value is a float, eg:
            ```python
                point_scalar_element = element("CG", mesh.domain.basix_cell(), degree=1)
                scalar_space = dolfinx.fem.functionspace(mesh.domain, point_scalar_element)
                dirichlet_bc = dolfinx.fem.dirichletbc(value=10.0, dofs=dofs, V=scalar_space)  # V must be define
                
                point_vector_element = element("CG", mesh.domain.basix_cell(), degree=1, shape=(mesh.tdim,))
                vector_space = dolfinx.fem.functionspace(mesh.domain, point_vector_element)
                dirichlet_bc = dolfinx.fem.dirichletbc(value=np.array([1., 0.]), dofs=dofs, V=vector_space)
            ```
            
            If bc_value is a dolfinx.fem.Constant, eg:
            ```python
                point_scalar_element = element("CG", mesh.domain.basix_cell(), degree=1)
                scalar_space = dolfinx.fem.functionspace(mesh.domain, point_scalar_element)
                dirichlet_bc = dolfinx.fem.dirichletbc(value=dolfinx.fem.Constant(mesh.domain, 10.0), dofs=dofs, V=scalar_space)  # V must be define
            ```
            
        Returns:
            - dirichlet_bc: 边界条件
        '''
        if isinstance(dofs, list):
            assert is_mixed_element is True, "is_mixed_element must be True"
            for dof in dofs:
                assert dof.shape[0] > 0, 'dofs must be not empty'
                assert dof.dtype == np.int32, 'dofs must be int32'
        elif isinstance(dofs, np.ndarray):
            assert is_mixed_element is False, "is_mixed_element must be False"
            assert dofs.shape[0] > 0, 'dofs must be not empty'
            assert dofs.dtype == np.int32, 'dofs must be int32'
        else:
            raise ValueError(f"Unsupported dofs type: {type(dofs)}")
        
        if isinstance(bc_value, Callable):
            assert value_func_space is not None, "value_func_space is not None"
            
            u_d = dolfinx.fem.Function(value_func_space)
            u_d.interpolate(bc_value)  # type: ignore
            if is_mixed_element:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs, V=bc_func_space)  # type: ignore
            else:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs)  # type: ignore

        elif isinstance(bc_value, dolfinx.fem.Function):
            if is_mixed_element:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=bc_func_space)  # type: ignore
            else:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs)  # type: ignore
            
        elif isinstance(bc_value, np.ndarray):
            assert value_func_space is not None, "value_func_space is not None"
            u_d = dolfinx.fem.Function(value_func_space)
            u_d.x.array[:] = bc_value  # type: ignore
            if is_mixed_element:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs, V=bc_func_space)  # type: ignore
            else:
                dirichlet_bc = dolfinx.fem.dirichletbc(value=u_d, dofs=dofs)  # type: ignore
            
        elif isinstance(bc_value, float):
            assert bc_func_space is not None, "bc_func_space is not None"
            assert is_mixed_element is False, "is_mixed_element must be False"
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=bc_func_space)  # type: ignore
        
        elif isinstance(bc_value, dolfinx.fem.Constant):
            assert bc_func_space is not None, "bc_func_space is not None"
            assert is_mixed_element is False, "is_mixed_element must be False"
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=bc_func_space)  # type: ignore
        
        else:
            raise ValueError(f"Unsupported bc_value type: {type(bc_value)}")
        return dirichlet_bc

    @staticmethod
    def create_homogenize_bc(
        bc_func_space: dolfinx.fem.FunctionSpace,
        value_func_space: dolfinx.fem.FunctionSpace,
        dofs: np.ndarray,
        is_mixed_element: bool,
        value_type: Union[float, np.ndarray, dolfinx.fem.Function, dolfinx.fem.Constant, Callable] | None = None
    ):
        if value_type is not None:
            if isinstance(value_type, float):
                assert bc_func_space is not None, "bc_func_space is not None"
                assert is_mixed_element is False, "is_mixed_element must be False"
                return dolfinx.fem.dirichletbc(0.0, dofs, bc_func_space)  # type: ignore
            
            elif isinstance(value_type, np.ndarray):
                # new_value = dolfinx.fem.Function(value_func_space)
                # return dolfinx.fem.dirichletbc(new_value, dofs, bc_func_space)
                raise NotImplementedError
            
            elif isinstance(value_type, dolfinx.fem.Function):
                new_value = dolfinx.fem.Function(value_type.function_space)                
                if is_mixed_element:
                    return dolfinx.fem.dirichletbc(new_value, dofs, bc_func_space)  # type: ignore
                else:
                    return dolfinx.fem.dirichletbc(new_value, dofs)  # type: ignore
            
            elif isinstance(value_type, dolfinx.fem.Constant):
                assert bc_func_space is not None, "bc_func_space is not None"
                assert is_mixed_element is False, "is_mixed_element must be False"
                return dolfinx.fem.dirichletbc(0.0, dofs, bc_func_space)  # type: ignore
            
            elif isinstance(value_type, Callable):
                new_value = dolfinx.fem.Function(value_func_space)
                if is_mixed_element:
                    return dolfinx.fem.dirichletbc(new_value, dofs, bc_func_space)  # type: ignore
                else:
                    return dolfinx.fem.dirichletbc(new_value, dofs)  # type: ignore
            
            else:
                raise ValueError(f"Unsupported value_type type: {type(value_type)}")
        else:
            assert value_func_space is not None, "value_func_space is not None"
            new_value = dolfinx.fem.Function(value_func_space)
            if is_mixed_element:
                return dolfinx.fem.dirichletbc(new_value, dofs, bc_func_space)  # type: ignore
            else:
                return dolfinx.fem.dirichletbc(new_value, dofs)  # type: ignore


class AssembleUtils(object):
    @staticmethod
    def assemble_mat(
        a_form: dolfinx.fem.Form | ufl.Form,
        bcs: list[dolfinx.fem.DirichletBC],
        a_mat: PETSc.Mat = None,
        method="lift",
        diagonal=1.0,
        check_symmetric=False,
    ) -> PETSc.Mat:
        """
        assemble_matrix 用于组建a_form(Bilinear)的中间矩阵,即Ax=b(的A矩阵)
        a_form必须包含TrialFunction和TestFunction
        reference: https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html

        Ref: https://jsdokken.com/FEniCS23-tutorial/src/lifting.html
        identity_row: 这是原始方法,但会导致A_mat不在对称致使求解困难,很大部分的求解器不再适用
        """
        assert method in ["identity_row", "lift"]
        if isinstance(a_form, ufl.Form):
            a_form = DolfinUtils.ufl_to_dolfin(a_form)

        if method == "identity_row":
            if a_mat is None:
                a_mat = dolfinx_petsc.assemble_matrix(a_form, diagonal=diagonal)
            else:
                dolfinx_petsc.assemble_matrix_mat(a_mat, a_form, diagonal=diagonal)
            a_mat.assemble()
            BoundaryUtils.apply_boundary_to_matrix_explicit(a_mat, bcs)
        elif method == "lift":
            if a_mat is None:
                a_mat = dolfinx_petsc.assemble_matrix(
                    a_form, bcs=bcs, diagonal=diagonal
                )
            else:
                dolfinx_petsc.assemble_matrix_mat(
                    a_mat, a_form, bcs=bcs, diagonal=diagonal
                )
            a_mat.assemble()
        else:
            raise NotImplementedError

        if check_symmetric:
            print(f"[Debug AssembleUtils] A_mat Symmetric: {a_mat.isSymmetric(1e-15)}")

        return a_mat

    @staticmethod
    def assemble_vec(
        L_form: dolfinx.fem.Form | ufl.Form, b_vec: PETSc.Vec = None, clear_vec=False
    ) -> PETSc.Vec:
        """
        todo
        assemble_matrix用于计算L_form在FunctionSpace中每一个element的积分值
        L_form必须包含TestFunction
        """
        if isinstance(L_form, ufl.Form):
            L_form = DolfinUtils.ufl_to_dolfin(L_form)

        if b_vec is None:
            b_vec = dolfinx_petsc.assemble_vector(L_form)
            # b_vec = dolfinx.fem.assemble_vector(L_form)
        else:
            if clear_vec:
                with b_vec.localForm() as loc:
                    loc.set(0)
            # dolfinx.fem.petsc._assemble_vector_vec
            dolfinx_petsc.assemble_vector(b_vec, L_form)
        return b_vec

    @staticmethod
    def assemble_form_vec(
        l_form: ufl.Form,
        function_space: dolfinx.fem.FunctionSpace,
        b_vec: PETSc.Vec = None,
        clear_vec=False,
    ) -> PETSc.Vec:
        """
        todo
        l_form must without ufl.dx or ufl.ds
        function_space: scalar function space or vector function space
        """
        agg_form = ufl.inner(l_form, ufl.TestFunction(function_space)) * ufl.dx
        return AssembleUtils.assemble_vec(
            DolfinUtils.ufl_to_dolfin(agg_form), b_vec=b_vec, clear_vec=clear_vec
        )

    @staticmethod
    def assemble_scalar(form: dolfinx.fem.Form | ufl.Form):
        """计算form的积分值,form必须是积分式"""
        if isinstance(form, ufl.Form):
            form = DolfinUtils.ufl_to_dolfin(form)
        return dolfinx.fem.assemble_scalar(form)

    @staticmethod
    def create_vector(L_form: dolfinx.fem.Form | ufl.Form):
        """only return 0 vector"""
        if isinstance(L_form, ufl.Form):
            L_form = DolfinUtils.ufl_to_dolfin(L_form)
        return dolfinx_petsc.create_vector(L_form)

    @staticmethod
    def create_matrix(a_form: dolfinx.fem.Form | ufl.Form):
        """only return 0 matrix"""
        if isinstance(a_form, ufl.Form):
            a_form = DolfinUtils.ufl_to_dolfin(a_form)
        return dolfinx_petsc.create_matrix(a_form)


if __name__ == "__main__":
    pass
