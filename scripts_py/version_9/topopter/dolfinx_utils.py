import dolfinx
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI
import ufl
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from typing import Callable, Union
import numpy as np
import pyvista


class DolfinxUtils(object):
    @staticmethod
    def convert_msh2XDMF(msh_file: str, output_file: str, name, dim=3):
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
    def solve_finite_element_problem(a_fun, L_fun, bcs: list):
        problem = LinearProblem(a_fun, L_fun, bcs=bcs)
        res_fun: dolfinx.fem.function.Function = problem.solve()
        return res_fun

    @staticmethod
    def define_dirichlet_boundary_from_fun(
            V: dolfinx.fem.functionspace, domain: dolfinx.mesh.Mesh,
            bc_value: Union[Callable, float, np.array],
            bc_fun: Callable = None
    ):
        """
        boundary_value: if is function, please use lambda, like:
            define_dirichlet_boundary(V, domain, lambda x: lambda x: 1 + x[0]**2 + 2 * x[1]**2, None)
            bc_value is function or float or vector
        boundary_fun is None, mean all boundary
        """
        if bc_fun is None:
            facets = dolfinx.mesh.exterior_facet_indices(domain)
            dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)
        else:
            facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, marker=bc_fun)
            dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)

            # use this function, you must sure surface is a plane
            # dofs = dolfinx.fem.locate_dofs_geometrical(V, bc_fun)

        print(f"[Debug] Create Dirichlet Boundary Based on {len(facets)} faces")
        assert len(facets) > 0
        if isinstance(bc_value, Callable):
            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(bc_value)
            dirichlet_bc = dolfinx.fem.dirichletbc(value=u_D, dofs=dofs)
        else:
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=V)
        return dirichlet_bc

    @staticmethod
    def define_dirichlet_boundary_marker(
            V: dolfinx.fem.functionspace, domain: dolfinx.mesh.Mesh,
            marker, mesh_tags: dolfinx.mesh.MeshTags,
            bc_value: Union[Callable, float, np.array]
    ):
        facets = mesh_tags.find(marker)
        print(f"[Debug] Create Dirichlet Boundary Based on {len(facets)} faces")
        assert len(facets) > 0

        dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)
        if isinstance(bc_value, Callable):
            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(bc_value)
            dirichlet_bc = dolfinx.fem.dirichletbc(value=u_D, dofs=dofs)
        else:
            dirichlet_bc = dolfinx.fem.dirichletbc(value=bc_value, dofs=dofs, V=V)
        return dirichlet_bc

    @staticmethod
    def define_neuuman_boundary_from_fun(domain: dolfinx.mesh.Mesh, bc_fun: Callable):
        facet_indices = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, bc_fun)
        facet_indices = facet_indices.astype(np.int32)
        print(f"[Debug] Create Neuuman Boundary Based on {len(facet_indices)} faces")
        assert len(facet_indices) > 0
        return facet_indices

    @staticmethod
    def define_neuuman_boundary_from_marker(marker: int, mesh_tags: dolfinx.mesh.MeshTags):
        assert marker > 1
        facet_indices = mesh_tags.find(marker)
        print(f"[Debug] Create Neuuman Boundary Based on {len(facet_indices)} faces")
        facet_indices = facet_indices.astype(np.int32)
        assert len(facet_indices) > 0
        return facet_indices

    @staticmethod
    def define_subdomain_facets_tag(domain: dolfinx.mesh.Mesh, indice_tag_dict: dict):
        """
        indice_tag_dict: {name:{
                            indices:[list[int]],
                            marker: int
                        }}
        """
        # facet_indices = dolfinx.mesh.exterior_facet_indices(domain.topology)
        facet_indices = []
        facet_markers = []
        for name in indice_tag_dict.keys():
            sub_facet_indices = indice_tag_dict[name]['indices']
            marker = indice_tag_dict[name]['marker']
            facet_indices.append(sub_facet_indices)
            facet_markers.append(np.full_like(sub_facet_indices, marker))

        facet_indices = np.array(facet_indices).reshape(-1).astype(np.int32)
        facet_markers = np.array(facet_markers).reshape(-1).astype(np.int32)
        facets_tag = dolfinx.mesh.meshtags(domain, domain.topology.dim - 1, facet_indices, facet_markers)
        return facets_tag

    @staticmethod
    def define_boundary_area(domain: dolfinx.mesh.Mesh, facets_tag: dolfinx.mesh.MeshTags):
        """
        the subdomain of ds is ds(marker)
        """
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)
        return ds

    @staticmethod
    def define_facet_norm(domain: dolfinx.mesh.Mesh):
        n = ufl.FacetNormal(domain)
        return n

    @staticmethod
    def define_coordinate(domain: dolfinx.mesh.Mesh):
        x = ufl.SpatialCoordinate(domain)
        return x

    @staticmethod
    def define_constant_tensor(domain: dolfinx.mesh.Mesh, value: Union[float, np.array]):
        """
        value: can be value or vector
        """
        c = dolfinx.fem.Constant(domain, c=value)
        return c

    @staticmethod
    def define_interpolate_fun(Q: dolfinx.fem.FunctionSpaceBase, ufl_expression):
        """
        Q: usually Discontinued Lagrange
        """
        project_fun = dolfinx.fem.Function(Q)
        ufl_expr = dolfinx.fem.Expression(ufl_expression, Q.element.interpolation_points())
        project_fun.interpolate(ufl_expr)
        return project_fun

    @staticmethod
    def compute_integral_scalar(ufl_expression):
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl_expression))

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

    @staticmethod
    def show_tag_vtk(grid: pyvista.UnstructuredGrid, mesh_tag: dolfinx.mesh.MeshTags):
        # num_local_cells = domain.topology.index_map(domain.topology.dim).size_local
        # grid.cell_data["Marker"] = mesh_tag.values[mesh_tag.indices < num_local_cells]
        grid.cell_data["Marker"] = mesh_tag.values
        grid.set_active_scalars("Marker")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.show()
        return grid


if __name__ == '__main__':
    DolfinxUtils.convert_msh2XDMF(
        name='t2',
        msh_file='/home/admin123456/Desktop/work/test/3D_top2/t2.msh',
        output_file='/home/admin123456/Desktop/work/test/3D_top2/t2.xdmf',
        dim=3
    )

    # domain, cell_tags, facet_tags = DolfinxUtils.read_XDMF(
    #     file='/home/admin123456/Desktop/work/test/t1.xdmf', cellTag_name='t1_cells', facetTag_name='t1_facets'
    # )

    pass
