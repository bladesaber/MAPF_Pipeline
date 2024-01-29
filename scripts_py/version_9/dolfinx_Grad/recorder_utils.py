from dolfinx.io import gmshio, XDMFFile, VTKFile
import dolfinx
from mpi4py import MPI
from tensorboardX import SummaryWriter
from typing import Union, List


class XDMFRecorder(object):
    def __init__(self, file: str, comm=None):
        assert file.endswith('.xdmf')
        if comm is None:
            comm = MPI.COMM_WORLD

        self.writter = XDMFFile(comm, file, 'w')

    def write_mesh(self, doamin: dolfinx.mesh.Mesh):
        self.writter.write_mesh(doamin)

    def write_function(self, function: dolfinx.fem.Function, step):
        self.writter.write_function(function, step)

    def close(self):
        self.writter.close()


class VTKRecorder(object):
    def __init__(self, file: str, comm=None):
        assert file.endswith('.pvd')
        if comm is None:
            comm = MPI.COMM_WORLD

        self.writter = VTKFile(comm, file, 'w')

    def write_mesh(self, doamin: dolfinx.mesh.Mesh, step):
        self.writter.write_mesh(doamin, step)

    def write_function(self, function: Union[dolfinx.fem.Function, List[dolfinx.fem.Function]], step):
        self.writter.write_function(function, step)

    def close(self):
        self.writter.close()


class TensorBoardRecorder(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)
