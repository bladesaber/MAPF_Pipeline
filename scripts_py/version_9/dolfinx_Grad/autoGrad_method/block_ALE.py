import dolfinx
from petsc4py import PETSc

from Thirdparty.pyadjoint import pyadjoint
from Thirdparty.pyadjoint.pyadjoint import annotate_tape, get_working_tape, stop_annotating

from .type_Mesh import Mesh
from .type_Function import Function
from ..petsc_utils import PETScUtils

"""
       block_variable
type <=================> blockVariable ------> checkpoint
            output
"""


def move_mesh_boundary(mesh: Mesh, u: dolfinx.fem.Function):
    dim = mesh.geometry.dim
    mesh.geometry.x[:, :dim] = mesh.geometry.x[:, :dim] + u.x.array.reshape((-1, dim))
    return mesh


def move(domain: Mesh, vector: Function, **kwargs):
    annotate = annotate_tape(kwargs)

    if annotate:
        assert isinstance(domain, Mesh) and isinstance(vector, Function)
        tape = get_working_tape()
        block = ALEMoveBlock(domain, vector, **kwargs)
        tape.add_block(block)

    with stop_annotating():
        output = move_mesh_boundary(domain, vector)

    if annotate:
        block.add_output(domain.create_block_variable())

    return output


class ALEMoveBlock(pyadjoint.Block):
    def __init__(self, domain: Mesh, vector: Function, **kwargs):
        super(ALEMoveBlock, self).__init__()
        self.add_dependency(domain)
        self.add_dependency(vector)

    def evaluate_adj(self, markings=False):
        adj_value: PETSc.Vec = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        adj_value_copy = PETScUtils.create_vec(adj_value.size)
        # adj_value_copy.array[:] = adj_value.array
        adj_value_copy.aypx(0.0, adj_value)

        self.get_dependencies()[0].add_adj_output(adj_value_copy)
        self.get_dependencies()[1].add_adj_output(adj_value)

    def evaluate_tlm(self, markings=False):
        vector: Function = self.get_dependencies()[1]
        if vector.tlm_value is None:
            return

        tlm_output = vector._ad_copy()
        self.get_outputs()[0].add_tlm_output(tlm_output)

    def recompute(self, markings=False):
        mesh: Mesh = self.get_dependencies()[0].saved_output
        vector: Function = self.get_dependencies()[1].saved_output
        move_mesh_boundary(mesh, vector)
        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()
