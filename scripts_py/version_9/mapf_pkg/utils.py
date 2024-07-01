import h5py


def read_h5df(file: str):
    return h5py.File(file)


def print_h5df_tree(data_h5: h5py.File, pre=''):
    items = len(data_h5)
    for key, val in data_h5.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                print_h5df_tree(val, pre + '    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                print_h5df_tree(val, pre + '│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))
