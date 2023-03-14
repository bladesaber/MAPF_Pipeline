# import numpy as np
# import pyvista
#
# # Make the xyz points
# theta = np.linspace(-10 * np.pi, 10 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# points = np.column_stack((x, y, z))
# print(points.shape)
#
# # spline = pyvista.Spline(points, 500).tube(radius=0.1)
# # spline.plot(scalars='arc_length', show_scalar_bar=False)

### --------------------------------------------------------------
# import numpy as np
# from mayavi.mlab import *
# from mayavi import mlab
#
# def test_plot3d():
#     """Generates a pretty set of lines."""
#     n_mer, n_long = 6, 11
#     dphi = np.pi / 1000.0
#     phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
#     mu = phi * n_mer
#     x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
#     y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
#     z = np.sin(n_long * mu / n_mer) * 0.5
#
#     l = plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
#     return l
#
# l = test_plot3d()
# mlab.show()

## -------------------
# import numpy as np
# import pyvista as pv
# from pyvista import examples
#
# kpos = [(-6.68, 11.9, 11.6), (3.5, 2.5, 1.26), (0.45, -0.4, 0.8)]
# mesh = examples.download_kitchen()
# kitchen = examples.download_kitchen(split=True)
#
# streamlines = mesh.streamlines(n_points=40, source_center=(0.08, 3, 0.71))
#
# p = pv.Plotter()
# p.add_mesh(mesh.outline(), color="k")
# p.add_mesh(kitchen, color=True)
# p.add_mesh(streamlines.tube(radius=0.01), scalars="velocity", lighting=False)
# p.camera_position = kpos
# p.show()

