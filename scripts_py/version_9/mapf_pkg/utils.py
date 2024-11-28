import h5py
from typing import List, Tuple, Dict
from tensorboardX import SummaryWriter


class Hdf5IO(object):
    @staticmethod
    def read_h5df(file: str):
        return h5py.File(file)

    @staticmethod
    def print_h5df_tree(data_h5: h5py.File, pre=''):
        items = len(data_h5)
        for key, val in data_h5.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    Hdf5IO.print_h5df_tree(val, pre + '    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    Hdf5IO.print_h5df_tree(val, pre + '│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    def write_scalars(self, tag, scalar_dict: Dict[str, float], step):
        if len(scalar_dict) == 0:
            return
        self.writer.add_scalars(tag, scalar_dict, step)
