import numpy as np
import dolfinx
from sklearn.neighbors import KDTree
import h5py
from typing import List
import pandas as pd

"""
Gmsh -> Ansys Fluent: bdf格式
Ansys Fluent -> Gmsh: cgns格式
"""


class CrossSimulatorUtil(object):
    @staticmethod
    def convert_to_simple_function(
            domain: dolfinx.mesh.Mesh, coords: np.ndarray,
            value_list: List[np.ndarray], r=1e-3
    ) -> List[dolfinx.fem.Function]:
        tree = KDTree(coords)
        idxs_list, dists_list = tree.query_radius(domain.geometry.x, r=r, return_distance=True)
        coord_idxs = []
        for idxs, dists in zip(idxs_list, dists_list):
            if len(idxs) != 1:
                raise ValueError('[ERROR]: Finite Element Mesh is not complete match')
            coord_idxs.append(idxs[0])
        coord_idxs = np.array(coord_idxs)

        # coords = coords[coord_idxs]
        fun_list = []
        for value in value_list:
            if value.ndim == 1:
                function_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))
                value = value[coord_idxs].reshape(-1)
            elif value.ndim == 2:
                function_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
                value = value[coord_idxs, :].reshape(-1)
            else:
                raise ValueError("[ERROR]: Non-Valid Shape")

            fun = dolfinx.fem.Function(function_space)
            fun.x.array[:] = value
            fun_list.append(fun)

        return fun_list

    @staticmethod
    def read_hdf5(file: str) -> h5py.File:
        assert file.split('.')[-1] in ['cgns', 'hdf5']
        return h5py.File(file)

    @staticmethod
    def read_data_from_hdf5(data_h5: h5py.File, tag_sequence: List[str]):
        data = data_h5
        for tag in tag_sequence:
            data = data[tag]
        return data

    @staticmethod
    def print_h5df_tree(data_h5: h5py.File, pre=''):
        items = len(data_h5)
        for key, val in data_h5.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    CrossSimulatorUtil.print_h5df_tree(val, pre + '    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    CrossSimulatorUtil.print_h5df_tree(val, pre + '│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    @staticmethod
    def read_ansys_fluent_data(
            data: h5py.File, read_tags: List[str], tdim: int, coord_scale=1.0, save_csv_file: str = None
    ):
        assert tdim in [2, 3]

        res_dict = {}
        for tag in read_tags:
            if tag == 'coord':
                coord_list = [
                    np.array(data['Base']['Zone']['GridCoordinates']['CoordinateX'][' data']).reshape((-1, 1)),
                    np.array(data['Base']['Zone']['GridCoordinates']['CoordinateY'][' data']).reshape((-1, 1))
                ]
                if tdim == 3:
                    coord_list.append(
                        np.array(data['Base']['Zone']['GridCoordinates']['CoordinateZ'][' data']).reshape((-1, 1))
                    )
                xyz_coord = np.concatenate(coord_list, axis=1)
                res_dict[tag] = xyz_coord * coord_scale

            elif tag == 'velocity':
                vel_list = [
                    np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityX'][' data']).reshape((-1, 1)),
                    np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityY'][' data']).reshape((-1, 1)),
                ]
                if tdim == 3:
                    vel_list.append(
                        np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityZ'][' data']).reshape((-1, 1))
                    )
                vel = np.concatenate(vel_list, axis=1)
                res_dict[tag] = vel

            elif tag == 'pressure':
                pressure = np.array(data['Base']['Zone']['FlowSolution.N:1']['Pressure'][' data']).reshape((-1, 1))
                res_dict[tag] = pressure

            else:
                raise ValueError("[ERROR]: Non-Valid Tag")

        if save_csv_file is not None:
            assert save_csv_file.endswith('.csv')
            df = pd.DataFrame()
            for tag in res_dict.keys():
                data = res_dict[tag]
                if tag == 'coord':
                    if tdim == 2:
                        df[['coord_x', 'coord_y']] = data
                    else:
                        df[['coord_x', 'coord_y', 'coord_z']] = data
                elif tag == 'velocity':
                    if tdim == 2:
                        df[['vel_x', 'vel_y']] = data
                    else:
                        df[['vel_x', 'vel_y', 'vel_z']] = data
                elif tag == 'pressure':
                    df['pressure'] = data

            df.to_csv(save_csv_file)

        return res_dict
