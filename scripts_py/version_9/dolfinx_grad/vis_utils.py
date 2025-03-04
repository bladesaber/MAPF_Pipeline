import pyvista
import numpy as np


class VisUtils(object):
    def __init__(self):
        self.plotter: pyvista.Plotter = pyvista.Plotter()
        self.plotter.set_background('white')

    def clear(self):
        self.plotter.clear()

    def render(self):
        self.plotter.render()

    def show(self, dim: int = None):
        if dim is not None:
            if dim == 2:
                self.plotter.view_xy()
        self.plotter.show()

    def plot(
            self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0,
            style=None, show_edges=False, show_scalar_bar=False,
            point_size=None, line_width=None, scalars=None, cmap='viridis'
    ):
        self.plotter.add_mesh(
            mesh, color=color, opacity=opacity, style=style,
            show_edges=show_edges, show_scalar_bar=show_scalar_bar,
            point_size=point_size, line_width=line_width,
            scalars=scalars, cmap=cmap
        )

    def plot_grad(
            self, pos_xyz: np.ndarray, grad: np.ndarray, plotter: pyvista.Plotter = None,
            mag=1.0, color=(1., 0., 0.), opacity=1.0
    ):
        if plotter is None:
            plotter = self.plotter
        plotter.add_arrows(pos_xyz, grad, mag=mag, color=color, opacity=opacity)

    @staticmethod
    def set_point_data(grid: pyvista.UnstructuredGrid, data: np.ndarray, name: str):
        grid.point_data[name] = data
        return grid

    @staticmethod
    def set_cell_data(grid: pyvista.UnstructuredGrid, data: np.ndarray, name: str):
        grid.cell_data[name] = data
        return grid
