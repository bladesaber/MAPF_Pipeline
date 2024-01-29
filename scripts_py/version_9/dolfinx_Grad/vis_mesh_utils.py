import pyvista
import dolfinx
import numpy as np


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
    def show_vector_res_vtk(grid: pyvista.UnstructuredGrid, uh: dolfinx.fem.function, dim, with_wrap=False, factor=1.0):
        if dim == 2:
            u_value = uh.x.array.reshape((-1, dim))
            grid['u'] = np.concatenate([u_value, np.zeros(shape=(u_value.shape[0], 1))], axis=1)
        else:
            grid['u'] = uh.x.array.reshape((-1, dim))

        plotter = pyvista.Plotter()
        if with_wrap:
            vis_mesh = grid.warp_by_vector("u", factor=factor)
            plotter.add_mesh(grid, style="wireframe", color="k")
            plotter.add_mesh(vis_mesh, show_edges=True)
        else:
            grid.set_active_scalars('u')
            plotter.add_mesh(grid, show_edges=True)

        plotter.show()

    @staticmethod
    def show_arrow_res_vtk(
            grid: pyvista.UnstructuredGrid, uh: dolfinx.fem.function,
            V: dolfinx.fem.FunctionSpaceBase, scale=0.2
    ):
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
        values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
        values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

        # Create a point cloud of glyphs
        function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["u"] = values
        glyphs = function_grid.glyph(orient="u", factor=scale)

        # Create plotter
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, style="wireframe", color="k")
        plotter.add_mesh(glyphs)
        plotter.view_xy()
        plotter.show()

    @staticmethod
    def show_vtu(vtu_file: str, field_name='f', scale=0.2):
        assert vtu_file.endswith('.vtu')
        grid = pyvista.read(vtu_file)
        u_vecs = grid[field_name]
        u_cents = grid.points

        plotter = pyvista.Plotter()
        plotter.add_arrows(u_cents, u_vecs, mag=scale)
        plotter.show()
