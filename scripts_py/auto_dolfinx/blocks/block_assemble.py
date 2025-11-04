from typing import Union, Callable, Any

import dolfinx
import ufl
from ufl.algorithms import expand_derivatives

import numpy as np

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import AssembleUtils


def assemble_scalar(form: ufl.Form, domain: dolfinx.mesh.Mesh, tape: Tape = None, with_block: bool = False):  # type: ignore
    if tape is None:
        with_annotate = annotate_tape()
    else:
        with_annotate = True
    
    with stop_annotating():
        output: float = AssembleUtils.assemble_scalar(form)
    output = create_overloaded_object(output)  # type: ignore
    
    if with_annotate:
        block = AssembleScalarBlock(form, domain)
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.create_block_variable())  # type: ignore
    
    if with_block and with_annotate:
        return output, block
    return output


class AssembleScalarBlock(Block):
    '''
    tlm_value:
        dolfinx.fem.Function
    adj_value:
        np.ndarray
    '''
    def __init__(self, form: ufl.Form, domain: dolfinx.mesh.Mesh):
        super(AssembleScalarBlock, self).__init__()
        self.form: ufl.Form = form
        self.domain = domain

        self.add_dependency(domain)
        coeffs = self.form.coefficients()
        if coeffs is not None:
            for c in coeffs:
                self.add_dependency(c, no_duplicates=True)
        else:
            raise ValueError("form must have coefficients")

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies) -> ufl.Form:
        ''' updates the form with latest result, return updated form
        Args:
            inputs: dependencies' values: type[original value type, i.e float, numpy.ndarray, Dolfinx function, etc.]
            adj_inputs: adjoint values from outputs
            relevant_dependencies: relevant dependencies: type[BlockVariable]
        '''
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
        form = ufl.replace(self.form, replaced_coeffs)
        return form  # type: ignore
    
    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs) -> ufl.Form:
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):
        raise NotImplementedError(type(self))

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared_form: ufl.Form) -> float:
        '''
        Args:
            inputs: dependencies' values: type[original value type, i.e float, numpy.ndarray, Dolfinx function, etc.]
            tlm_inputs: tlm values from outputs
            block_variable: block variable of the output corresponding to index `idx`
            idx: index of the component to compute
            prepared_form: form from prepare_evaluate_tlm method
        Returns:
            An object of the same type as `block_variable.saved_output`: The resulting product.
        '''
        dform: ufl.Form = 0.0  # type: ignore
        dform_shape: ufl.Form = 0.0  # type: ignore
        
        for bv in self.get_dependencies():
            c_rep: dolfinx.fem.Function = bv.saved_output
            tlm_value = bv.tlm_value
            if tlm_value is None:
                continue
            
            if isinstance(c_rep, dolfinx.mesh.Mesh):
                # tlm_value is dolfinx.fem.Function
                # function space is: 
                # coordinate_element = coord.ufl_domain().ufl_coordinate_element()
                # coordinate_func_space = dolfinx.fem.functionspace(mesh, coordinate_element)
                coord = MeshUtils.define_coordinate(c_rep)
                dform_shape += ufl.derivative(prepared_form, coord, tlm_value)
            
            elif isinstance(c_rep, dolfinx.fem.Function):
                # tlm_value is dolfinx.fem.Function, funcation space is the same as c_rep
                dform += ufl.derivative(prepared_form, c_rep, tlm_value)

            else:
                raise ValueError(f"Unsupported dependency type: {type(c_rep)}")
        
        if not isinstance(dform, float):
            # expand_derivatives is used to symbolically expand and simplify the derivative expression before assembly
            dform = expand_derivatives(dform)  # type: ignore
            dform_value: float = AssembleUtils.assemble_scalar(dform)  # type: ignore

        if not isinstance(dform_shape, float):
            dform_shape_value: float = AssembleUtils.assemble_scalar(dform_shape)  # type: ignore

        return dform_value + dform_shape_value  # type: ignore
  
    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared_form: ufl.Form) -> np.ndarray:
        '''
        Reverse Automatic Differentiation (RAD)
        Args:
            inputs: dependencies' values: type[original value type, i.e float, numpy.ndarray, Dolfinx function, etc.]
            adj_inputs: adjoint values from outputs
            block_variable: block variable of the dependencies corresponding to index `idx`
            idx: index of the component to compute
            prepared_form: form from prepare_evaluate_adj method
        Returns:
            An object of the same type as `block_variable.saved_output`: The resulting product.
        '''
        adj_input: float = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output
        
        if isinstance(c, dolfinx.fem.Function):
            # 当以TestFunction作为泛函，即求解梯度
            dform = ufl.derivative(prepared_form, c_rep, ufl.TestFunction(c.function_space))
        
        elif isinstance(c, dolfinx.mesh.Mesh):
            coord = MeshUtils.define_coordinate(c_rep)
            coordinate_element = coord.ufl_domain().ufl_coordinate_element()  # type: ignore
            coordinate_func_space = dolfinx.fem.functionspace(c, coordinate_element)
            dform = ufl.derivative(prepared_form, coord, ufl.TestFunction(coordinate_func_space))
        
        else:
            raise ValueError(f"Unsupported dependency type: {type(c)}")
        
        output = AssembleUtils.assemble_vec(dform)
        return output.array * adj_input

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)
    
    def recompute_component(self, inputs, block_variable, idx, prepared) -> float:
        form: ufl.Form = prepared
        output = AssembleUtils.assemble_scalar(form)
        output = create_overloaded_object(output)
        return output  # type: ignore