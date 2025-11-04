from os import name
from typing import Union, Callable, Any

import dolfinx

import numpy as np
from petsc4py import PETSc

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.block_variable import BlockVariable
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh


def move_mesh(mesh: DolfinxMesh, u: DolfinxFunction):
    dim = mesh.geometry.dim
    mesh.geometry.x[:, :dim] = mesh.geometry.x[:, :dim] + u.x.array.reshape((-1, dim))
    return mesh


def deform(
    domain: DolfinxMesh,
    deform_fun: DolfinxFunction,
    tape: Tape = None,  # type: ignore
    with_block: bool = False
):
    if tape is None:
        with_annotate = annotate_tape()
    else:
        with_annotate = True
    
    if with_annotate:
        block = MeshDeformBlock(domain, deform_fun)
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
    
    with stop_annotating():
        output = move_mesh(domain, deform_fun)

    if with_annotate:
        block.add_output(output.create_block_variable())
    
    if with_block and with_annotate:
        return output, block
    return output


class MeshDeformBlock(Block):
    def __init__(self, domain: DolfinxMesh, deform_fun: DolfinxFunction):
        super(MeshDeformBlock, self).__init__()
        self.add_dependency(domain)
        self.add_dependency(deform_fun)

    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if isinstance(adj_value, PETSc.Vec):  # type: ignore
            adj_value_np = adj_value.array
        elif isinstance(adj_value, np.ndarray):
            adj_value_np = adj_value
        elif isinstance(adj_value, (dolfinx.fem.Function, DolfinxFunction)):
            adj_value_np = adj_value.x.array
        else:
            raise ValueError(f"Unsupported adjoint value type: {type(adj_value)}")

        self.get_dependencies()[0].add_adj_output(adj_value_np.copy())
        self.get_dependencies()[1].add_adj_output(adj_value_np.copy())

    def evaluate_tlm(self, markings=False):
        tlm_output: np.ndarray = 0.0  # type: ignore
        for i in range(2):
            dep_i: BlockVariable = self.get_dependencies()[i]
            if dep_i.tlm_value is None:
                continue
            
            if isinstance(dep_i.tlm_value, PETSc.Vec):  # type: ignore
                tlm_input_np = dep_i.tlm_value.array
            elif isinstance(dep_i.tlm_value, np.ndarray):
                tlm_input_np = dep_i.tlm_value
            elif isinstance(dep_i.tlm_value, (dolfinx.fem.Function, DolfinxFunction)):
                tlm_input_np = dep_i.tlm_value.x.array
            else:
                raise ValueError(f"Unsupported tlm value type: {type(dep_i.tlm_value)}")

            tlm_output += tlm_input_np.copy()

        self.get_outputs()[0].add_tlm_output(tlm_output)

    def recompute(self, markings=False):
        domain: DolfinxMesh = self.get_dependencies()[0].saved_output
        deform_fun: DolfinxFunction = self.get_dependencies()[1].saved_output
        move_mesh(domain, deform_fun)
        self.get_outputs()[0].checkpoint = domain._ad_create_checkpoint()

