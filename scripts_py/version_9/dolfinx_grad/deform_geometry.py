import numpy as np
from typing import List, Dict, Union
import pyvista
import re

from pyspline import Curve, Surface, Volume
from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils


class TecplotIO(object):
    @staticmethod
    def decode(file: str):
        """zone: interpolated, control_pts, orig_data"""
        assert file.endswith('.dat')
        zone_dict, state = {}, None
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('VARIABLES'):
                    continue
                elif line.startswith('Zone'):
                    zone = re.search(r'T="([^"]+)"', line).group(1)
                    zone_dict[zone] = {
                        'data': []
                    }
                    state = zone
                    continue
                elif line.startswith('DATAPACKING'):
                    zone_dict[state]['data_packing'] = re.match(r'DATAPACKING=([^"]+)', line).group(1)
                    continue

                points = [float(i.strip()) for i in line.strip()]
                zone_dict[state]['data'].append(points)

        for key in zone_dict.keys():
            zone_dict[key]['data'] = np.array(zone_dict[key]['data'])

        return zone_dict

    @staticmethod
    def encode(zone_dict: Dict[str, Union[np.ndarray, str]], file: str):
        assert file.endswith('.dat')
        with open(file, 'w') as f:
            f.write('VARIABLES = "CoordinateX", "CoordinateY", "CoordinateZ"\n')
            f.write(f'Zone T="interpolated" I={zone_dict["interpolated"]["data"].shape[0]}\n')
            f.write(f'DATAPACKING={zone_dict["interpolated"]["data_packing"]}\n')
            for point in zone_dict["interpolated"]["data"]:
                f.write(f'{" ".join([str(i) for i in point])}\n')
            f.write(f'Zone T="control_pts" I={zone_dict["control_pts"]["data"].shape[0]}\n')
            f.write(f'DATAPACKING={zone_dict["control_pts"]["data_packing"]}\n')
            for point in zone_dict["interpolated"]["data"]:
                f.write(f'{" ".join([str(i) for i in point])}\n')
            f.write(f'Zone T="orig_data" I={zone_dict["orig_data"]["data"].shape[0]}\n')
            f.write(f'DATAPACKING={zone_dict["orig_data"]["data_packing"]}\n')
            for point in zone_dict["orig_data"]["data"]:
                f.write(f'{" ".join([str(i) for i in point])}\n')


class SurfaceBspline(object):
    def __init__(self, pcd_surface: np.ndarray, uv_num: List[int], uv_degree: List[int], fit_nIter=100):
        self.uv_num = uv_num
        self.uv_degree = uv_degree

        surf = Surface(
            x=pcd_surface[:, :, 0], y=pcd_surface[:, :, 1], z=pcd_surface[:, :, 2],
            ku=uv_degree[0] + 1, kv=uv_degree[1] + 1,  # pyspline的配置需要加1，knot才能匹配
            nCtlu=uv_num[0], nCtlv=uv_num[1],
            nIter=fit_nIter
        )
        self.coef = surf.coef

        # self.bspline_cfg = {
        #     'u': BsplineUtils.create_knot_cfg(uv_num[0], uv_degree[0], True, True),
        #     'v': BsplineUtils.create_knot_cfg(uv_num[1], uv_degree[1], True, True),
        # }
        self.knot = {'u': surf.tu, 'v': surf.tv}

        pcd = pcd_surface.reshape((-1, 3))
        pcd_u, pcd_v, resid_distance = surf.projectPoint(pcd, nIter=fit_nIter)
        print(f'[Info]:Maximum Residual:{np.max(resid_distance)}')
        self.pcd_uv = np.concatenate([pcd_u.reshape((-1, 1)), pcd_v.reshape((-1, 1))], axis=1)

        # 由于Bspline的compute_Bip函数缺陷导致不能含有0.0或1.0这两个值
        self.pcd_uv = np.maximum(np.minimum(self.pcd_uv, 1 - 1e-5), 1e-5)

        # surf.getDerivative(self.pcd_uv[0, 0], self.pcd_uv[0, 1])  # 可用于计算表面曲率或切线信息
        self.mat: np.ndarray = self.compute_mat(
            self.pcd_uv, self.uv_num, self.uv_degree,
            self.knot['u'], self.knot['v']
        )

    @staticmethod
    def compute_mat(uvs: np.ndarray, uv_num, uv_degree, u_knot, v_knot):
        u_mat = BsplineUtils.compute_mat(uvs[:, 0], uv_num[0], uv_degree[0], u_knot, fill_last_one=False)
        v_mat = BsplineUtils.compute_mat(uvs[:, 1], uv_num[1], uv_degree[1], v_knot, fill_last_one=False)
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        return mat

    def evaluate_uv(self, u, v):
        point = np.zeros(3)
        for i in range(self.uv_num[0]):
            for j in range(self.uv_num[1]):
                bu = BsplineUtils.compute_Bip(u, i, self.uv_degree[0], self.knot['u'])
                bv = BsplineUtils.compute_Bip(v, j, self.uv_degree[1], self.knot['v'])
                point += bu * bv * self.coef[i, j, :]
        return point

    def get_pcd(self):
        res = self.mat[..., np.newaxis] * self.coef
        return np.sum(np.sum(res, axis=1), axis=1)


class VolumeBspline(object):
    """for 3d solid"""
    def __init__(
            self,
            cube_range: List[List[float]], cube_num: List[int], cube_degree: List[int],
            volume_pcd: np.ndarray, fit_nIter=100
    ):
        self.uvw_num = cube_num
        self.uvw_degree = cube_degree

        xs = np.linspace(cube_range[0][0], cube_range[0][1], num=cube_num[0])
        ys = np.linspace(cube_range[1][0], cube_range[1][1], num=cube_num[1])
        zs = np.linspace(cube_range[2][0], cube_range[2][1], num=cube_num[2])
        xs, ys, zs = np.meshgrid(xs, ys, zs)
        cube_pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)

        vol = Volume(
            X=cube_pcd,
            ku=cube_degree[0] + 1, kv=cube_degree[1] + 1, kw=cube_degree[2] + 1,
            nCtlu=cube_num[0], nCtlv=cube_num[1], nCtlw=cube_num[2],
            nIter=fit_nIter
        )
        vol.coef = cube_pcd
        self.coef = vol.coef
        self.knot = {'u': vol.tu, 'v': vol.tv, 'w': vol.tw}

        pcd_u, pcd_v, pcd_w, resid_distance = vol.projectPoint(volume_pcd, nIter=fit_nIter)
        print(f'[Info]:Maximum Residual:{np.max(resid_distance)}')
        self.pcd_uvw = np.concatenate([
            pcd_u.reshape((-1, 1)), pcd_v.reshape((-1, 1)), pcd_w.reshape((-1, 1))
        ], axis=1)

        self.mat: np.ndarray = self.compute_mat(
            self.pcd_uvw, self.uvw_num, self.uvw_degree,
            self.knot['u'], self.knot['v'], self.knot['w']
        )

    @staticmethod
    def compute_mat(uvws: np.ndarray, uvw_num, uvw_degree, u_knot, v_knot, w_knot):
        u_mat = BsplineUtils.compute_mat(uvws[:, 0], uvw_num[0], uvw_degree[0], u_knot, fill_last_one=False)
        v_mat = BsplineUtils.compute_mat(uvws[:, 1], uvw_num[1], uvw_degree[1], v_knot, fill_last_one=False)
        w_mat = BsplineUtils.compute_mat(uvws[:, 2], uvw_num[2], uvw_degree[2], w_knot, fill_last_one=False)
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        mat = mat[:, :, :, np.newaxis] * w_mat[:, np.newaxis, np.newaxis, :]
        return mat

    def evaluate_uvw(self, u, v, w):
        point = np.zeros(3)
        for i in range(self.uvw_num[0]):
            for j in range(self.uvw_num[1]):
                for k in range(self.uvw_num[2]):
                    bu = BsplineUtils.compute_Bip(u, i, self.uvw_degree[0], self.knot['u'])
                    bv = BsplineUtils.compute_Bip(v, j, self.uvw_degree[1], self.knot['v'])
                    bw = BsplineUtils.compute_Bip(w, k, self.uvw_degree[2], self.knot['w'])
                    point += bu * bv * bw * self.coef[i, j, k, :]
        return point

    def get_pcd(self):
        res = self.mat[..., np.newaxis] * self.coef
        return np.sum(np.sum(np.sum(res, axis=1), axis=1), axis=1)


class AreaBspline(object):
    """for 2d solid"""
    def __init__(
            self,
            uv_range: List[List[float]], uv_num: List[int], uv_degree: List[int],
            pcd_area: np.ndarray, fit_nIter=100
    ):
        """pcd_area: shape is [n, 2]"""
        assert pcd_area.shape[-1] == 2

        self.uv_num = uv_num
        self.uv_degree = uv_degree

        xs = np.linspace(uv_range[0][0], uv_range[0][1], num=uv_num[0])
        ys = np.linspace(uv_range[1][0], uv_range[1][1], num=uv_num[1])
        xs, ys = np.meshgrid(xs, ys)
        square_pcd = np.concatenate([
            xs[..., np.newaxis], ys[..., np.newaxis],
            np.zeros((xs.shape[0], xs.shape[1], 1))
        ], axis=-1)

        surf = Surface(
            x=square_pcd[:, :, 0], y=square_pcd[:, :, 1], z=square_pcd[:, :, 2],
            ku=uv_degree[0] + 1, kv=uv_degree[1] + 1,  # pyspline的配置需要加1，knot才能匹配
            nCtlu=uv_num[0], nCtlv=uv_num[1],
            nIter=fit_nIter
        )
        self.coef = surf.coef
        self.knot = {'u': surf.tu, 'v': surf.tv}

        pcd_2d = pcd_area.reshape((-1, 2))
        pcd_3d = np.concatenate([pcd_2d, np.zeros((pcd_2d.shape[0], 1))], axis=1)
        pcd_u, pcd_v, resid_distance = surf.projectPoint(pcd_3d, nIter=fit_nIter)
        print(f'[Info]:Maximum Residual:{np.max(resid_distance)}')
        self.pcd_uv = np.concatenate([pcd_u.reshape((-1, 1)), pcd_v.reshape((-1, 1))], axis=1)

        # 由于Bspline的compute_Bip函数缺陷导致不能含有0.0或1.0这两个值
        self.pcd_uv = np.maximum(np.minimum(self.pcd_uv, 1 - 1e-6), 1e-6)

        # surf.getDerivative(self.pcd_uv[0, 0], self.pcd_uv[0, 1])  # 可用于计算表面曲率或切线信息
        self.mat: np.ndarray = self.compute_mat(
            self.pcd_uv, self.uv_num, self.uv_degree,
            self.knot['u'], self.knot['v']
        )

    @staticmethod
    def compute_mat(uvs: np.ndarray, uv_num, uv_degree, u_knot, v_knot):
        u_mat = BsplineUtils.compute_mat(uvs[:, 0], uv_num[0], uv_degree[0], u_knot, fill_last_one=False)
        v_mat = BsplineUtils.compute_mat(uvs[:, 1], uv_num[1], uv_degree[1], v_knot, fill_last_one=False)
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        return mat

    def get_pcd(self):
        res = self.mat[..., np.newaxis] * self.coef
        return np.sum(np.sum(res, axis=1), axis=1)


def debug_surface():
    # TecplotIO.decode('/home/admin123456/Desktop/work/para_geo/helix.dat')

    x = np.linspace(-4, 4, 30)
    y = np.linspace(-4, 4, 30)
    x, y = np.meshgrid(x, y)
    z = np.cos(x) * np.sin(y)
    pcd_surface = np.concatenate([
        x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis],
    ], axis=-1)

    mesh = SurfaceBspline(pcd_surface, uv_num=[15, 15], uv_degree=[3, 3])
    reproject_pcd = mesh.get_pcd()
    pcd = pcd_surface.reshape((-1, 3))

    error = np.linalg.norm(reproject_pcd - pcd, ord=2, axis=1)
    print(f'Max Error:{np.max(error)}')

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.PointSet(pcd), color=(1., 0., 0.))
    plotter.add_mesh(pyvista.PointSet(reproject_pcd), color=(0., 1., 0.))
    plotter.show()


def debug_area():
    mesh = pyvista.read('/home/admin123456/Desktop/work/topopt_exps/new_test/example03/cache/result_vtu/res_00000.vtu')
    mesh_points = np.asarray(mesh.points)[:, :2]
    mesh_min = np.min(mesh_points, axis=0)
    mesh_max = np.max(mesh_points, axis=0)
    uv_range = [
        [mesh_min[0] - 1., mesh_max[0] + 1.],
        [mesh_min[1] - 1., mesh_max[1] + 1.],
    ]
    model = AreaBspline(uv_range=uv_range, uv_num=[10, 10], uv_degree=[3, 3], pcd_area=mesh_points, fit_nIter=100)
    reproject_pcd = model.get_pcd()

    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh, color=(1., 0., 0.), style='wireframe')
    plotter.add_mesh(pyvista.PointSet(reproject_pcd), color=(0., 1., 0.))
    plotter.show()


def debug_volume():
    mesh = pyvista.read('/home/admin123456/Desktop/work/para_geo/model.vtk')
    mesh_pcd = np.asarray(mesh.points)

    mesh_min = np.min(mesh_pcd, axis=0)
    mesh_max = np.max(mesh_pcd, axis=0)
    cube_range = [
        [mesh_min[0] - 1.0, mesh_max[0] + 1.0],
        [mesh_min[1] - 1.0, mesh_max[1] + 1.0],
        [mesh_min[2] - 1.0, mesh_max[2] + 1.0],
    ]
    cube_num = [5, 5, 5]
    cube_degree = [3, 3, 3]
    mesh = VolumeBspline(cube_range, cube_num, cube_degree, mesh_pcd)
    reproject_pcd = mesh.get_pcd()

    error = np.linalg.norm(reproject_pcd - mesh_pcd, ord=2, axis=1)
    print(f'Max Error:{np.max(error)}')

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.PointSet(mesh_pcd), color=(1., 0., 0.))
    plotter.add_mesh(pyvista.PointSet(reproject_pcd), color=(0., 1., 0.))
    plotter.show()


if __name__ == '__main__':
    # debug_surface()
    debug_area()
    # debug_volume()
    pass
