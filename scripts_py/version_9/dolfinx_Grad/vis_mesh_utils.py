"""
More Info can be found: reference: https://github.com/nate-sime/febug.git
"""

import pyvista
import dolfinx
import numpy as np


class VisUtils(object):
    def __init__(self):
        self.plotter = pyvista.Plotter()
        self.plotter.set_background('white')

    def plot(
            self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0,
            style=None, show_edges=False, show_scalar_bar=False
    ):
        self.plotter.add_mesh(
            mesh, color=color, opacity=opacity, style=style,
            show_edges=show_edges, show_scalar_bar=show_scalar_bar
        )

    def show(self, dim: int = None):
        if dim is not None:
            if dim == 2:
                self.plotter.view_xy()
        self.plotter.show()

    @staticmethod
    def convert_to_grid(domain: dolfinx.mesh.Mesh, tdim: int = None, entities=None) -> pyvista.DataSet:
        cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim, entities)
        grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
        return grid

    @staticmethod
    def show_scalar_res_vtk(
            grid: pyvista.UnstructuredGrid, scalar_tag: str, res_fun: dolfinx.fem.function.Function,
            is_point_data=True
    ):
        if is_point_data:
            grid.point_data[scalar_tag] = res_fun.x.array.real
        else:
            grid.cell_data[scalar_tag] = res_fun.x.array.real

        grid.set_active_scalars(scalar_tag)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.show()
        return grid

    @staticmethod
    def show_vector_res_vtk(
            grid: pyvista.UnstructuredGrid, uh: dolfinx.fem.function, dim, with_wrap=False, factor=1.0,
            plotter: pyvista.Plotter = None
    ):
        if plotter is None:
            plotter = pyvista.Plotter()

        if dim == 2:
            u_value = uh.x.array.reshape((-1, dim))
            grid['u'] = np.concatenate([u_value, np.zeros(shape=(u_value.shape[0], 1))], axis=1)
        else:
            grid['u'] = uh.x.array.reshape((-1, dim))

        if with_wrap:
            vis_mesh = grid.warp_by_vector("u", factor=factor)
            plotter.add_mesh(grid, style="wireframe", color="k")
            plotter.add_mesh(vis_mesh, show_edges=True)
        else:
            grid.set_active_scalars('u')
            plotter.add_mesh(grid, show_edges=True)

        return plotter

    @staticmethod
    def show_arrow_res_vtk(
            grid: pyvista.UnstructuredGrid, uh: dolfinx.fem.function,
            V: dolfinx.fem.FunctionSpaceBase, scale=0.2,
            plotter: pyvista.Plotter = None
    ):
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
        values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
        values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

        # Create a point cloud of glyphs
        function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        function_grid[uh.name] = values
        glyphs = function_grid.glyph(orient=uh.name, scale=uh.name, factor=scale)

        if plotter is None:
            plotter = pyvista.Plotter()

        plotter.add_mesh(grid, style="wireframe", color="k")
        plotter.add_mesh(glyphs)
        # plotter.view_xy()

        return plotter

    @staticmethod
    def show_vtu(vtu_file: str, field_name='f', scale=0.2):
        assert vtu_file.endswith('.vtu')
        grid = pyvista.read(vtu_file)
        u_vecs = grid[field_name]
        u_cents = grid.points

        plotter = pyvista.Plotter()
        plotter.add_arrows(u_cents, u_vecs, mag=scale)
        plotter.show()

    @staticmethod
    def function_to_grid(fun: dolfinx.fem.Function, plotter: pyvista.Plotter = None) -> pyvista.UnstructuredGrid:
        """
        reference: https://github.com/nate-sime/febug.git
        """
        V = fun.function_space
        mesh = V.mesh

        bs = V.dofmap.index_map_bs
        dof_values = fun.x.array.reshape(V.tabulate_dof_coordinates().shape[0], V.dofmap.index_map_bs)

        if bs == 2:
            dof_values = np.hstack((dof_values, np.zeros((dof_values.shape[0], 3 - bs))))

        if np.iscomplexobj(dof_values):
            dof_values = dof_values.real

        if V.ufl_element().degree() == 0:
            grid = VisUtils.convert_to_grid(mesh, mesh.topology.dim)
            num_dofs_local = V.dofmap.index_map.size_local
            grid.cell_data[fun.name] = dof_values[:num_dofs_local]
        else:
            grid = VisUtils.convert_to_grid(V)
            grid.point_data[fun.name] = dof_values

        grid.set_active_scalars(fun.name)

        if plotter is not None:
            plotter.add_mesh(grid, scalars=fun.name, show_scalar_bar=True)

        return grid

    @staticmethod
    def meshtags_to_grid(domain: dolfinx.mesh.Mesh, meshtags: dolfinx.mesh.MeshTags, plotter: pyvista.Plotter = None):
        if np.issubdtype(meshtags.values.dtype, np.integer):
            unique_vals = np.unique(meshtags.values)
            annotations = dict(zip(unique_vals, map(str, unique_vals)))
        else:
            annotations = None

        print(annotations)

        if meshtags.dim > 0:
            entities = VisUtils.convert_to_grid(domain, meshtags.dim, meshtags.indices)
            entities.cell_data[meshtags.name] = meshtags.values
            entities.set_active_scalars(meshtags.name)
        else:
            x = domain.geometry.x[meshtags.indices]
            entities = pyvista.PolyData(x)
            entities[meshtags.name] = meshtags.values

        if plotter is not None:
            plotter.add_mesh(entities, show_scalar_bar=True, annotations=annotations)

        return entities

    @staticmethod
    def plot_mesh_quality(
            mesh: dolfinx.mesh.Mesh, tdim: int,
            quality_measure: str = "scaled_jacobian",
            entities=None,
            progress_bar: bool = True,
            plotter: pyvista.Plotter = None,
    ):
        """
        quality_measure:
            1. area
            2. radius_ratio
            3. skew
            4. volume
            5. max_angle
            6. min_angle
            7. condition
        """
        if mesh.topology.index_map(tdim) is None:
            mesh.topology.create_entities(tdim)

        grid = VisUtils.convert_to_grid(mesh, tdim, entities)
        qual = grid.compute_cell_quality(quality_measure=quality_measure, progress_bar=progress_bar)
        qual.set_active_scalars("CellQuality")

        if plotter is None:
            plotter = pyvista.Plotter()

        plotter.add_mesh(grid, style='wireframe')
        plotter.add_mesh(qual, show_scalar_bar=True)

        return plotter

    @staticmethod
    def show_arrow_from_grid(grid: pyvista.DataSet, name: str, scale=1.0, plotter: pyvista.Plotter = None):
        if plotter is None:
            plotter = pyvista.Plotter()

        plotter.add_arrows(grid.points, grid[name], mag=scale)
        return plotter

    @staticmethod
    def show_scalar_from_grid(grid: pyvista.DataSet, name: str, plotter: pyvista.Plotter = None):
        if plotter is None:
            plotter = pyvista.Plotter()

        grid.set_active_scalars(name)
        plotter.add_mesh(grid, show_edges=True)
        return plotter
