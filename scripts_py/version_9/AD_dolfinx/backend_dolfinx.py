import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
import pyvista
from typing import Union, Callable
from tensorboardX import SummaryWriter

from .petsc_utils import PETScUtils


class SolverUtils(object):

    @staticmethod
    def solve_linear_variational_problem(
            uh: dolfinx.fem.function.Function,
            a_form: ufl.form.Form, L_form: ufl.form.Form, bcs: list[dolfinx.fem.DirichletBC],
            petsc_options: dict = {},
            **kwargs,
    ):
        problem = LinearProblem(a_form, L_form, bcs, u=uh, petsc_options=petsc_options)
        problem.solve()

        if kwargs.pop('check_correct', False):
            a_form_dolfinx = dolfinx.fem.form(a_form)
            L_form_dolfinx = dolfinx.fem.form(L_form)
            A_mat = SolverUtils.assemble_mat(a_form_dolfinx, bcs)
            b_vec = SolverUtils.assemble_vec(L_form_dolfinx)
            SolverUtils.apply_boundary_to_vec(b_vec, bcs, a_form_dolfinx, clean_vec=False)
            solver = PETScUtils.create_PETSc_solver(solver_setting=petsc_options)
            res_dict = PETScUtils.solve_linear_system_by_PETSc(A_mat, b_vec, solver)

            residual = uh.x.array - res_dict['res'].array
            print(
                f"[### Linear Variational Problem] "
                f"error_mean:{res_dict['error_mean']:.5f} error_max:{res_dict['error_max']:.5f} "
                f"residual:{np.max(np.abs(residual)):.5f} cost:time:{res_dict['cost_time']:.3f}"
            )

        if kwargs.pop('check_valid', False):
            assert not np.any(np.isnan(uh.x.array)) or np.any(np.isinf(uh.x.array))

        return uh

    @staticmethod
    def solve_nonlinear_variational_problem(
            uh: dolfinx.fem.function.Function,
            F_form: ufl.form.Form, bcs: list[dolfinx.fem.DirichletBC],
            rtol=1e-6, petsc_options: dict = None
    ):
        problem = NonlinearProblem(F_form, uh, bcs=bcs)

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = 'incremental'
        solver.rtol = rtol

        if petsc_options is not None:
            # solver.report = True
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            for key in petsc_options.keys():
                opts[f"{option_prefix}{key}"] = petsc_options[key]
            # opts[f"{option_prefix}ksp_type"] = "cg"
            # opts[f"{option_prefix}pc_type"] = "gamg"
            # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()

        run_times, is_converged = solver.solve(uh)

        return run_times, is_converged, uh

    @staticmethod
    def solve_linear_system_problem(
            uh: dolfinx.fem.Function, b_vec: PETSc.Vec, A_mat: PETSc.Mat = None,
            comm=None, solver: PETSc.KSP = None, solver_setting: dict = {}
    ):
        if solver is None:
            solver = PETScUtils.create_PETSc_solver(comm, solver_setting)
            solver.setOperators(A_mat)
        solver.solve(b_vec, uh.vector)
        uh.x.scatter_forward()
        return uh

    @staticmethod
    def uflForm_to_dolfinxForm(form: ufl.form.Form):
        return dolfinx.fem.form(form)

    @staticmethod
    def assemble_mat(
            a_form: dolfinx.fem.Form, bcs: list[dolfinx.fem.DirichletBC], A_mat: PETSc.Mat = None
    ):
        """
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
        if b_vec is None:
            b_vec = dolfinx.fem.petsc.assemble_vector(L_form)
            # b_vec = dolfinx.fem.assemble_vector(L_form)
        else:
            # dolfinx.fem.petsc._assemble_vector_vec
            dolfinx.fem.petsc.assemble_vector(b_vec, L_form)
        return b_vec

    @staticmethod
    def apply_boundary_to_vec(
            b_vec: PETSc.Vec, bcs: list[dolfinx.fem.DirichletBC], a_form: dolfinx.fem.form,
            clean_vec: bool = False
    ):
        if clean_vec:
            with b_vec.localForm() as loc_b:
                loc_b.set(0)

        dolfinx.fem.apply_lifting(b_vec, [a_form], [bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b_vec, bcs)

    @staticmethod
    def create_vector(L_form: dolfinx.fem.Form):
        b_vec = dolfinx.fem.petsc.create_vector(L_form)
        return b_vec

    @staticmethod
    def create_matrix(a_form: dolfinx.fem.Form):
        A_mat = dolfinx.fem.petsc.create_matrix(a_form)
        return A_mat


class MeshUtils(object):
    @staticmethod
    def msh_to_XDMF(msh_file: str, output_file: str, name, dim=3):
        """
        if operation fail, please check:
            if dim=3: whether all necessary volumes had been added to physical group
            if dim=2: whether all necessary surfaces had been added to physical group
        """
        assert output_file.endswith('.xdmf') and msh_file.endswith('.msh')
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
    def extract_boundary_entities(
            domain: dolfinx.mesh.Mesh, marker: Union[Callable, int] = None,
            mesh_tags: dolfinx.mesh.MeshTags = None
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
        entities_idxs = dolfinx.mesh.locate_entities(domain, dim, marker=marker)
        return entities_idxs

    @staticmethod
    def define_subdomain_facets_tag(domain: dolfinx.mesh.Mesh, marker_idxs_dict: dict, dim: int):
        indices_set = []
        markers_set = []
        for marker in marker_idxs_dict.keys():
            sub_domain_indices = marker_idxs_dict[marker]
            indices_set.append(sub_domain_indices)
            markers_set.append(np.full_like(sub_domain_indices, marker))

        indices_set = np.array(indices_set).reshape(-1).astype(np.int32)
        markers_set = np.array(markers_set).reshape(-1).astype(np.int32)
        mesh_tags = dolfinx.mesh.meshtags(domain, dim, indices_set, markers_set)
        return mesh_tags

    @staticmethod
    def define_facet_norm(domain: dolfinx.mesh.Mesh):
        n = ufl.FacetNormal(domain)
        return n

    @staticmethod
    def define_coordinate(domain: dolfinx.mesh.Mesh):
        x = ufl.SpatialCoordinate(domain)
        return x

    @staticmethod
    def define_ds(domain: dolfinx.mesh.Mesh, facets_tag: dolfinx.mesh.MeshTags = None):
        """define area of boundary"""
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)
        return ds

    @staticmethod
    def define_dx(domain: dolfinx.mesh.Mesh):
        """define volume of mesh"""
        dx = ufl.Measure("dx", domain=domain)
        return dx


class BoundaryUtils(object):
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
            V: dolfinx.fem.functionspace, domain: dolfinx.mesh.Mesh,
            sub_domain_fun: Callable, marker: int,
            bc_value: Union[Callable, float, np.array, dolfinx.fem.Function]
    ):
        entities = MeshUtils.extract_boundary_entities(domain, marker=sub_domain_fun)
        entities = entities.astype(np.int32)

        if isinstance(bc_value, Callable):
            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(bc_value)
            bc_value = u_D

        return {'type': 'neuuman', 'entity': entities, 'bc_value': bc_value, 'marker': marker}

    @staticmethod
    def create_bc(
            value: Union[np.array, float, dolfinx.fem.Function, dolfinx.fem.Constant],
            dofs, V, homogenize=False
    ):
        if isinstance(value, float):
            if homogenize:
                bc = dolfinx.fem.dirichletbc(0.0, dofs, V)
            else:
                bc = dolfinx.fem.dirichletbc(value, dofs, V)

        elif isinstance(value, np.ndarray):
            if homogenize:
                bc = dolfinx.fem.dirichletbc(np.zeros_like(value), dofs, V)
            else:
                bc = dolfinx.fem.dirichletbc(value, dofs, V)

        elif isinstance(value, dolfinx.fem.Constant):
            # value: dolfinx.fem.Constant
            # if homogenize:
            #     bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(0.0), dofs, V)
            # else:
            #     bc = dolfinx.fem.dirichletbc(value, dofs, V)
            raise NotImplementedError

        elif isinstance(value, dolfinx.fem.Function):
            new_value = dolfinx.fem.Function(value.function_space)
            if homogenize:
                new_value.x.array[:] = 0.0
                bc = dolfinx.fem.dirichletbc(new_value, dofs, V)
            else:
                new_value.x.array[:] = value.x.array
                bc = dolfinx.fem.dirichletbc(new_value, dofs, V)

        else:
            raise NotImplementedError

        return bc


class ComputeUtils(object):
    @staticmethod
    def compute_integral(form: ufl.form.Form):
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))


class VisUtils(object):
    def __init__(self):
        self.ploter = pyvista.Plotter()
        self.ploter.set_background('white')

    def plot(self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0, style=None, show_edges=False):
        self.ploter.add_mesh(mesh, color=color, opacity=opacity, style=style, show_edges=show_edges)

    def show(self):
        self.ploter.show()

    @staticmethod
    def convert_to_grid(domain: dolfinx.mesh.Mesh):
        cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain)
        grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
        return grid

    @staticmethod
    def show_scalar_res_vtk(
            grid: pyvista.UnstructuredGrid, scalar_tag: str, res_fun: dolfinx.fem.function.Function
    ):
        grid.point_data[scalar_tag] = res_fun.x.array.real
        grid.set_active_scalars(scalar_tag)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.show()
        return grid


class XDMFRecorder(object):
    def __init__(self, file: str, comm=None):
        assert file.endswith('.xdmf')
        if comm is None:
            comm = MPI.COMM_WORLD

        self.writter = XDMFFile(comm, file, 'w')

    def write_mesh(self, doamin: dolfinx.mesh.Mesh):
        self.writter.write_mesh(doamin)

    def write_function(self, function: dolfinx.fem.Function, step):
        self.writter.write_function(function, step)


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

