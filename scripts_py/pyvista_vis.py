import pyvista
import numpy as np


class PyVistaVisUtils(object):
    def __init__(self):
        self.plotter: pyvista.Plotter = pyvista.Plotter()
        self.plotter.set_background('white')  # type: ignore

    def clear(self):
        self.plotter.clear()

    def render(self):
        self.plotter.render()
    
    def show(self, dim: int | None = None):
        if dim is not None:
            if dim == 2:
                self.plotter.view_xy()  # type: ignore
        self.plotter.show()
    
    def plot_mesh(
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
    
    def plot_arrow(self, start: np.ndarray, end: np.ndarray, color=(0.5, 0.1, 0.8), opacity=1.0, line_width=None):
        self.plotter.add_arrows(start, end, color=color, opacity=opacity, line_width=line_width)