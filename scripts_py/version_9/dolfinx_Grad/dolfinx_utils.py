import dolfinx
import numpy as np
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
from typing import Union, Callable, List
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from ufl.core import expr


class AssembleUtils(object):
    @staticmethod
    def ufl_to_dolfinx(form: ufl.form.Form):
        return dolfinx.fem.form(form)

    @staticmethod
    def assemble_mat(
            a_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC], A_mat: PETSc.Mat = None
    ):
        """
        assemble_matrix 用于组建a_form(Bilinear)的中间矩阵,即Ax=b(的A矩阵)
        a_form必须包含TrialFunction和TestFunction
        reference: https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
        """
        if A_mat is None:
            A_mat = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
        else:
            dolfinx.fem.petsc.assemble_matrix_mat(A_mat, a_form, bcs=bcs)
        A_mat.assemble()
        return A_mat

    @staticmethod
    def assemble_vec(L_form: dolfinx.fem.Form, b_vec: PETSc.Vec = None) -> PETSc.Vec:
        """
        assemble_matrix用于计算L_form在FunctionSpace中每一个element的积分值
        L_form必须包含TestFunction
        """
        if b_vec is None:
            b_vec = dolfinx.fem.petsc.assemble_vector(L_form)
            # b_vec = dolfinx.fem.assemble_vector(L_form)
        else:
            # dolfinx.fem.petsc._assemble_vector_vec
            dolfinx.fem.petsc.assemble_vector(b_vec, L_form)
        return b_vec

    @staticmethod
    def assemble_scalar(form: dolfinx.fem.Form):
        """
        计算form的积分值,form必须是积分式
        """
        return dolfinx.fem.assemble_scalar(form)

    @staticmethod
    def create_vector(L_form: dolfinx.fem.Form):
        return dolfinx.fem.petsc.create_vector(L_form)

    @staticmethod
    def create_matrix(a_form: dolfinx.fem.Form):
        return dolfinx.fem.petsc.create_matrix(a_form)

    @staticmethod
    def derivative(form: ufl.Form, coefficient: ufl.Coefficient, direction: Union[ufl.Coefficient, ufl.Argument]):
        """
        coefficient: can not be Argument(TrialFunction or TestFunction)
        derivative here is variation(泛函变分) rather than gradient.
        ufl.derivative(f, u, dire) = grad(f, u) * dire
        """
        return ufl.derivative(form, coefficient, direction)

    @staticmethod
    def function_eval(uh: dolfinx.fem.Function, point_xyzs: np.ndarray, cells: List[int]):
        """
        Evaluate Function at points x, where x has shape (num_points, 3),
        and cells has shape (num_points,) and cell[i] is the index of the
        cell containing point x[i]. If the cell index is negative the
        point is ignored.
        """
        values = uh.eval(point_xyzs, cells)
        return values


class MeshUtils(object):
    @staticmethod
    def msh_to_XDMF(msh_file: str, output_file: str, name, dim=3):
        """
        msh_file: .msh, .gmsh2, .gmsh
        if operation fail, please check:
            if dim=3: whether all necessary volumes had been added to physical group
            if dim=2: whether all necessary surfaces had been added to physical group
        """
        assert output_file.endswith('.xdmf')
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)
        mesh.name = name
        cell_tags.name = f"{mesh.name}_cells"
        facet_tags.name = f"{mesh.name}_facets"
        with XDMFFile(mesh.comm, output_file, 'w') as f:
            mesh.topology.create_connectivity(dim - 1, dim)
            f.write_mesh(mesh)
            f.write_meshtags(
                cell_tags, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )
            f.write_meshtags(
                facet_tags, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )

    @staticmethod
    def read_XDMF(file: str, mesh_name, cellTag_name, facetTag_name):
        assert file.endswith('.xdmf')
        with XDMFFile(MPI.COMM_WORLD, file, "r") as f:
            domain = f.read_mesh(name=mesh_name)
            cell_tags = f.read_meshtags(domain, name=cellTag_name)
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
        # only after creating the connectivity, facet_tags can be extracted
        with XDMFFile(MPI.COMM_WORLD, file, "r") as f:
            facet_tags = f.read_meshtags(domain, name=facetTag_name)
        return domain, cell_tags, facet_tags

    @staticmethod
    def get_topology_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim

    @staticmethod
    def get_facet_dim(domain: dolfinx.mesh.Mesh):
        return domain.topology.dim - 1

    @staticmethod
    def extract_facet_entities(
            domain: dolfinx.mesh.Mesh, mesh_tags: dolfinx.mesh.MeshTags = None, marker: Union[Callable, int] = None
    ):
        tdim = domain.topology.dim
        if marker is None:
            domain.topology.create_connectivity(tdim - 1, tdim)
            entities_idxs = dolfinx.mesh.exterior_facet_indices(domain.topology)
        else:
            if isinstance(marker, Callable):
                entities_idxs = dolfinx.mesh.locate_entities_boundary(domain, tdim - 1, marker=marker)
            elif isinstance(marker, int):
                entities_idxs = mesh_tags.find(marker)
            else:
                raise NotImplementedError
        return entities_idxs

    @staticmethod
    def extract_entity_dofs(V: dolfinx.fem.functionspace, dim: int, entities_idxs: np.array):
        dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=dim, entities=entities_idxs)
        return dofs

    @staticmethod
    def extract_entities(domain: dolfinx.mesh.Mesh, marker: Callable, dim: int):
        """
        return the index of response area in dolfin.fem.Function
        find more info: https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html
        """
        entities_idxs = dolfinx.mesh.locate_entities(domain, dim, marker=marker)
        return entities_idxs

    @staticmethod
    def define_facet_norm(domain: dolfinx.mesh.Mesh):
        return ufl.FacetNormal(domain)

    @staticmethod
    def define_coordinate(domain: dolfinx.mesh.Mesh):
        return ufl.SpatialCoordinate(domain)

    @staticmethod
    def define_ds(domain: dolfinx.mesh.Mesh, facets_tag: dolfinx.mesh.MeshTags = None):
        """define area of boundary"""
        return ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)

    @staticmethod
    def define_dx(domain: dolfinx.mesh.Mesh):
        """define volume of mesh"""
        return ufl.Measure("dx", domain=domain)

    @staticmethod
    def extract_inverse_boundary_entities(
            domain: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags, other_marker: int, dim: int
    ):
        facet_indices = MeshUtils.extract_facet_entities(domain)
        marker_tags = np.full_like(facet_indices, fill_value=other_marker)
        idxs = np.nonzero(np.in1d(facet_indices, facet_tags.indices))[0]
        marker_tags[idxs] = facet_tags.values
        mesh_tags = dolfinx.mesh.meshtags(domain, dim, facet_indices, marker_tags)
        return mesh_tags

    @staticmethod
    def define_meshtag(
            domain: dolfinx.mesh.Mesh, indices_list: List[np.ndarray], markers_list: List[np.ndarray], dim: int
    ):
        indices = np.concatenate(indices_list, axis=-1).astype(np.int32)
        markers = np.concatenate(markers_list, axis=-1).astype(np.int32)
        meshtag = dolfinx.mesh.meshtags(domain, dim, indices, markers)
        return meshtag

    @staticmethod
    def move(domain: dolfinx.mesh.Mesh, coodr_dif: np.ndarray):
        assert domain.geometry.x.shape == coodr_dif.shape
        domain.geometry.x[:] += coodr_dif

    @staticmethod
    def num_of_entities(domain: dolfinx.mesh.Mesh, dim: int):
        nums = domain.topology.index_map(dim).size_local
        return nums


class BoundaryUtils(object):
    @staticmethod
    def apply_boundary_to_vec(
            b_vec: PETSc.Vec, bcs: list[dolfinx.fem.DirichletBC], a_form: dolfinx.fem.form, clean_vec: bool = False
    ):
        if clean_vec:
            with b_vec.localForm() as loc_b:
                loc_b.set(0)

        dolfinx.fem.apply_lifting(b_vec, [a_form], [bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b_vec, bcs)

    @staticmethod
    def define_dirichlet_cell(
            V: dolfinx.fem.functionspace, dofs: np.array,
            bc_value: Union[Callable, float, np.array, dolfinx.fem.Function]
    ):
        assert dofs.shape[0] > 0
        if isinstance(bc_value, Callable):
            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(bc_value)
            dirichlet_bc = dolfinx.fem.dirichletbc(value=u_D, dofs=dofs)
        elif isinstance(bc_value, dolfinx.fem.Function):
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs)
        else:
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=V)
        return {'type': 'dirichlet', 'boundary': dirichlet_bc}

    @staticmethod
    def define_neuuman_cell(
            L_form: ufl.Form, expr_form: ufl.Form, ds: ufl.Measure,
            marker: int, mesh_tags: dolfinx.mesh.MeshTags,
    ):
        assert mesh_tags.find(marker).shape[0] > 0
        L_form += expr_form * ds(marker)

    @staticmethod
    def create_homogenize_bc(V: dolfinx.fem.functionspace, dofs: np.ndarray, value_type=None):
        if value_type is not None:
            if isinstance(value_type, float):
                return dolfinx.fem.dirichletbc(0.0, dofs, V)
            elif isinstance(value_type, np.ndarray):
                return dolfinx.fem.dirichletbc(np.zeros_like(value_type), dofs, V)
            elif isinstance(value_type, dolfinx.fem.Function):
                new_value = dolfinx.fem.Function(value_type.function_space)
                new_value.x.array[:] = 0.0
                return dolfinx.fem.dirichletbc(new_value, dofs, V)
            else:
                raise NotImplementedError

        else:
            new_value = dolfinx.fem.Function(V)
            new_value.x.array[:] = 0.0
            return dolfinx.fem.dirichletbc(new_value, dofs, V)


class UFLUtils(object):
    @staticmethod
    def create_expression(
            form: expr,
            V: dolfinx.fem.FunctionSpaceBase
    ):
        exp = dolfinx.fem.Expression(form, V.element.interpolation_points())
        return exp
