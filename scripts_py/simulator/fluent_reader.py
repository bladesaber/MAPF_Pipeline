import numpy as np
from sklearn.neighbors import KDTree
import h5py
from typing import List, Union
import pandas as pd
import os


class FluentReader(object):
    tags = {
        'x': 'Base/Zone/GridCoordinates/CoordinateX/ data',
        'y': 'Base/Zone/GridCoordinates/CoordinateY/ data',
        'z': 'Base/Zone/GridCoordinates/CoordinateZ/ data',
        'vel_x': 'Base/Zone/FlowSolution.N:1/CoordinateX/ data',
        'vel_y': 'Base/Zone/FlowSolution.N:1/CoordinateY/ data',
        'vel_z': 'Base/Zone/FlowSolution.N:1/CoordinateZ/ data',
        'pressure': 'Base/Zone/FlowSolution.N:1/PressureStagnation/ data'
    }

    def __init__(self, result_file: str):
        assert result_file.split('.')[-1] in ['cgns', 'hdf5', 'h5']
        self.data = h5py.File(result_file)

    @staticmethod
    def read_data(data_h5: h5py.File, tag_sequence: List[str]):
        data = data_h5
        for tag in tag_sequence:
            data = data[tag]
        return np.array(data)

    @staticmethod
    def print_h5df_tree(data_h5: h5py.File, pre=''):
        items = len(data_h5)
        for key, val in data_h5.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    FluentReader.print_h5df_tree(val, pre + '    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    FluentReader.print_h5df_tree(val, pre + '│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))