from os import name
from typing import Union, Callable, Any

import dolfinx
import ufl
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_arguments

import numpy as np
from petsc4py import PETSc

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint import OverloadedType, AdjFloat
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction


def assign(
    func: DolfinxFunction, 
    other: Union[DolfinxFunction, AdjFloat], 
    tape: Tape = None,  # type: ignore
    with_block: bool = False
):
    if tape is None:
        with_annotate = annotate_tape()
    else:
        with_annotate = True
    
    if with_annotate:
        block = FunctionAssignBlock(func, other)
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
    
    with stop_annotating():
        func.assign(other)
    
    if with_annotate:
        block.add_output(func.create_block_variable())
    
    if with_block and with_annotate:
        return None, block
    return None


class FunctionAssignBlock(Block):
    def __init__(self, func, other, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.func = func
        self.other: DolfinxFunction|AdjFloat = None  # type: ignore
        
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif isinstance(other, float) or isinstance(other, int):
            other = AdjFloat(other)
            self.add_dependency(other, no_duplicates=True)
        else:
            raise ValueError(f"Unsupported other type: {type(other)}")
    
    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared) -> np.ndarray | float:
        assert len(adj_inputs) == 1, "Only one adjoint input is supported"
        
        if isinstance(adj_inputs[0], PETSc.Vec):  # type: ignore
            adj_input_np = adj_inputs[0].array.copy()
        elif isinstance(adj_inputs[0], np.ndarray):
            adj_input_np = adj_inputs[0].copy()
        elif isinstance(adj_inputs[0], dolfinx.fem.Function):
            adj_input_np = adj_inputs[0].x.array.copy()
        else:
            raise ValueError(f"Unsupported adjoint input type: {type(adj_inputs[0])}")
        
        if isinstance(block_variable.output, (float, AdjFloat)):
            return adj_input_np.sum()
        elif isinstance(block_variable.output, DolfinxFunction):
            return adj_input_np
        else:
            raise ValueError(f"Unsupported output type: {type(block_variable.output)}")
    
    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        assert len(tlm_inputs) == 1, "Only one tlm input is supported"
        
        tlm_input = tlm_inputs[0]
        if isinstance(tlm_input, (float, AdjFloat)):
            tlm_input_func = dolfinx.fem.Function(self.func.function_space)
            tlm_input_func.x.array[:] = np.ones_like(block_variable.output.x.array.shape) * tlm_input  # type: ignore
            return tlm_input_func
        
        elif isinstance(tlm_input, DolfinxFunction):
            return tlm_input
        else:
            raise ValueError(f"Unsupported tlm input type: {type(tlm_input)}")
                
    def recompute_component(self, inputs, block_variable, idx, prepared):
        input_value = inputs[0]
        if isinstance(input_value, (float, AdjFloat)):
            output_func = dolfinx.fem.Function(self.func.function_space)
            output_func.x.array[:] = np.ones_like(block_variable.output.x.array.shape) * input_value  # type: ignore
            return output_func
        elif isinstance(input_value, (DolfinxFunction, dolfinx.fem.Function)):
            return input_value
        else:
            raise ValueError(f"Unsupported input type: {type(input_value)}")
        