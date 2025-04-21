import numpy as np
from pyspline import Curve
import pyvista
from typing import Dict, Union
import pandas as pd
from sklearn.neighbors import KDTree


class MathServerUtils:
    @staticmethod
    def refit_spline_from_curve(curve_pcd: np.ndarray, degree: int, n_control: int, n_iter=100):
        curve = Curve(
            x=curve_pcd[:, 0], y=curve_pcd[:, 1], z=curve_pcd[:, 2],
            k=degree, nCtl=n_control, nIter=n_iter
        )
        ts, resid_distance = curve.projectPoint(curve_pcd)
        max_resid_error = np.max(resid_distance)

        coef = curve.coef
        coef[0, :] = curve_pcd[0, :]
        coef[-1, :] = curve_pcd[-1, :]

        return {'status': True, 'coef': coef, 'knots': curve.t, 'ts': ts, 'max_error': max_resid_error}

    @staticmethod
    def refit_spline_with_less_control_points(
            curve_pcd: np.ndarray, degree: int, min_control: int, control_step=5,
            n_iter=100, max_trial=100, tol=1e-2
    ):
        iteration = 1
        cur_n_control = min_control
        while True:
            res = MathServerUtils.refit_spline_from_curve(
                curve_pcd=curve_pcd, degree=degree, n_control=cur_n_control, n_iter=n_iter
            )
            if res['max_error'] < tol:
                return res

            cur_n_control += control_step
            iteration += 1
            if iteration > max_trial:
                break
        return {'status': False}

    @staticmethod
    def simplify_group_spline(
            group_curve: Dict[str, Dict[str, Union[np.ndarray, float]]], degree: int, min_control: int, control_step=5,
            refit_tol=1e-2, max_trial=100, n_iter=100, coef_decimals=3
    ):
        spline_dict = {}
        for curve_name, curve_cfg in group_curve.items():
            res = MathServerUtils.refit_spline_with_less_control_points(
                curve_pcd=curve_cfg['xyz'],
                degree=degree + 1, #  pyspline的配置需要加1，knot才能匹配
                min_control=min_control, control_step=control_step,
                n_iter=n_iter, max_trial=max_trial, tol=refit_tol
            )
            if res['status']:
                res.update({
                    'coef': np.round(res['coef'], decimals=coef_decimals),
                    'radius': curve_cfg['radius']
                })
                spline_dict[curve_name] = res

                # # ------ visual debug
                # import pyvista
                # plotter = pyvista.Plotter()
                # curve_mesh = pyvista.PointSet(curve_cfg['xyz'])
                # coef_mesh = pyvista.PointSet(res['coef'])
                # plotter.add_mesh(curve_mesh, color=(0., 1., 0.))
                # plotter.add_mesh(coef_mesh, color=(1., 0., 0.))
                # plotter.show()
                # # ------

            else:
                return {'status': False, 'msg': f'[Error]: Failed to simplify curve {curve_name}.'}

        coefs = np.concatenate([spline_cfg['coef'] for spline_cfg in spline_dict.values()], axis=0)
        df = pd.DataFrame(coefs, columns=['x', 'y', 'z'])
        df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        coefs = df.values

        tree = KDTree(coefs)
        for spline_name, spline_cfg in spline_dict.items():
            idxs_list = tree.query(spline_cfg['coef'], k=1, return_distance=False)
            idxs_list = np.concatenate(idxs_list, axis=-1)
            if np.unique(idxs_list).shape[0] != idxs_list.shape[0]:
                return {'status': False, 'msg': "[Error]: curve has duplicated points"}
            spline_cfg['coef'] = idxs_list
            del spline_cfg['ts']

        return {'status': True, 'coef': coefs, 'spline_dict': spline_dict}


def debug_pyspline():
    n = 100
    theta = np.linspace(0.0000, 2 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.linspace(0, 1, n)
    xyz = np.column_stack((x, y, z))

    curve = Curve(x=x, y=y, z=z, k=4, nCtl=16, nIter=100)
    ts, resid_distance = curve.projectPoint(xyz)
    print(ts)

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.PointSet(curve.coef), color=(1., 0., 0.))
    plotter.add_mesh(pyvista.PointSet(xyz), color=(0., 1., 0.))
    plotter.show()


def debug_simplify_group_spline():
    import h5py
    from tqdm import tqdm

    data = h5py.File('/home/admin123456/Desktop/work/flask_server/test.hdf5')
    group_res = {}
    for group_name, group_curves_cfg in tqdm(data.items()):
        curves_dict = {}
        for curve_name, curve_cfg in group_curves_cfg.items():
            curves_dict[curve_name] = {
                'xyz': np.array(curve_cfg['xyzr'])[:, :3],
                'radius': np.mean(np.array(curve_cfg['xyzr'])[:, -1]),
            }
        res = MathServerUtils.simplify_group_spline(
            curves_dict, 3, 5, 5, 1e-2, 15, 100
        )
        if not res['status']:
            print(res['msg'])
            return

        group_res[group_name] = res

    writer = h5py.File('/home/admin123456/Desktop/work/flask_server/test_res.hdf5', mode='w')
    for group_name, res in group_res.items():
        group_writer = writer.create_group(group_name)

        coef: np.ndarray = res['coef']
        group_writer.create_dataset("coefficients", data=coef)

        seg_writer = group_writer.create_group("segments")
        for spline_name, spline_res in res['spline_dict'].items():
            spline_writer = seg_writer.create_group(spline_name)
            spline_writer.create_dataset('coefficient_idxs', data=spline_res['coef'])
            spline_writer.create_dataset('knots', data=spline_res['knots'])
            spline_writer.create_dataset('radius', data=[spline_res['radius']])
            spline_writer.create_dataset('degree', data=[3])


if __name__ == "__main__":
    # debug_pyspline()
    debug_simplify_group_spline()

    from scripts_py.version_9.mapf_pkg.utils import Hdf5IO
    import h5py
    data = h5py.File('/home/admin123456/Desktop/work/flask_server/test_res.hdf5')
    Hdf5IO.print_h5df_tree(data)

    pass