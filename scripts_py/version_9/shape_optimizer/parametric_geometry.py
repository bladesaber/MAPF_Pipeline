import numpy as np
import pyvista
from pyspline import Curve, Surface, Volume
import re

from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class TecplotIO(object):
    @staticmethod
    def read_dat(file: str):
        if not file.endswith('.dat'):
            return False

        data = {}
        zone = None
        with open(file, 'r') as f:
            for line in f.readlines():
                if line.startswith('VARIABLES'):
                    pass
                elif line.startswith('Zone'):
                    match = re.search(r'T="([^"]+)"', line)
                    zone = match.group(1)
                    data[zone] = []

                elif line.startswith('DATAPACKING'):
                    data_packing = re.match(r'DATAPACKING=([^"]+)', line).group(1)
                else:
                    points = [float(i) for i in re.split(r'\s+', line.strip())]
                    data[zone].append(points)

        for key in data.keys():
            data[key] = np.array(data[key])

        return data


""" 1D spline
n = 200
theta = np.linspace(0.0, 2 * np.pi, n)
x = np.cos(theta)
y = np.sin(theta)
z = np.linspace(0., 1., n)
curve = Curve(x=x, y=y, z=z, k=3, nCtl=16, nIter=100)
curve.writeTecplot('/home/admin123456/Desktop/work/para_geo/helix.dat')

xyz = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))], axis=1)
res_t, res_distance = curve.projectPoint(xyz, nIter=100)
refit_xyz = curve.getValue(res_t)

data = TecplotIO.read_dat('/home/admin123456/Desktop/work/para_geo/helix.dat')
plotter = pyvista.Plotter()
plotter.add_mesh(pyvista.PointSet(data['interpolated']), color=(0., 0., 1.))
plotter.add_mesh(pyvista.PointSet(data['control_pts']), color=(0., 1., 0.))
plotter.add_mesh(pyvista.PointSet(data['orig_data']), color=(1., 0., 0.))
plotter.show()
"""

"""
# ------ 2D spline
nu, nv = 30, 30
u = np.linspace(-4, 4, nu)
v = np.linspace(-4, 4, nv)
V, U = np.meshgrid(u, v)
Z = np.cos(U) * np.sin(V)
surf = Surface(x=U, y=V, z=Z, ku=3, kv=3, nCtlu=15, nCtlv=15, nIter=100)
surf.writeTecplot('/home/admin123456/Desktop/work/para_geo/surf.dat')

xyz = np.concatenate([U.reshape((-1, 1)), V.reshape((-1, 1)), Z.reshape((-1, 1))], axis=1)
res_u, res_v, res_distance = surf.projectPoint(xyz, nIter=100)
refit_xyz = surf.getValue(res_u, res_v)
print(f"max error: {np.max(np.abs(xyz - refit_xyz))}")

data = TecplotIO.read_dat('/home/admin123456/Desktop/work/para_geo/surf.dat')

plotter = pyvista.Plotter()
plotter.add_mesh(pyvista.PointSet(data['interpolated']), color=(0., 0., 1.))
plotter.add_mesh(pyvista.PointSet(data['control_pts']), color=(0., 1., 0.))
plotter.add_mesh(pyvista.PointSet(data['orig_data']), color=(1., 0., 0.))
plotter.show()
"""

""" 3 spline
mesh = pyvista.read('/home/admin123456/Desktop/work/para_geo/model.vtk')
mesh_pcd = np.asarray(mesh.points)
mesh_range = np.max(mesh_pcd, axis=0) - np.min(mesh_pcd, axis=0)
cube_min, cube_max = np.min(mesh_pcd, axis=0) - mesh_range * 0.025, np.max(mesh_pcd, axis=0) + mesh_range * 0.025
num = 4
xs = np.linspace(cube_min[0], cube_max[0], num=num)
ys = np.linspace(cube_min[1], cube_max[1], num=num)
zs = np.linspace(cube_min[2], cube_max[2], num=num)
xs, ys, zs = np.meshgrid(xs, ys, zs)
cube_pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)

vol = Volume(X=cube_pcd, ku=3, kv=3, kw=3, nCtlu=num, nCtlv=num, nCtlw=num)
vol.writeTecplot('/home/admin123456/Desktop/work/para_geo/vol.dat')

res_u, res_v, res_w, res_distance = vol.projectPoint(mesh_pcd, nIter=100)
refit_xyz = vol.getValue(res_u, res_v, res_w)
print(f"max error: {np.max(np.abs(mesh_pcd - refit_xyz))}")

data = TecplotIO.read_dat('/home/admin123456/Desktop/work/para_geo/vol.dat')
# for key in data.keys():
#     print(f'key:{key} shape:{data[key].shape}')

plotter = pyvista.Plotter()
# plotter.add_mesh(pyvista.PointSet(data['interpolated']), color=(0., 0., 1.))
plotter.add_mesh(pyvista.PointSet(data['control_pts']), color=(1., 0., 0.))
# plotter.add_mesh(pyvista.PointSet(mesh_pcd), color=(0., 1., 0.))
plotter.add_mesh(pyvista.PointSet(refit_xyz), color=(0., 0., 1.))

vol_shape = vol.coef.shape
control_pcd = vol.coef.reshape((-1, 3))
control_pcd[10:30, 0] += 10.0
refit_xyz_2 = vol.getValue(res_u, res_v, res_w)
plotter.add_mesh(pyvista.PointSet(refit_xyz_2), color=(0., 1., 0.))
plotter.add_mesh(pyvista.PointSet(vol.coef.reshape((-1, 3))), color=(0., 0., 0.))

plotter.show()
"""
