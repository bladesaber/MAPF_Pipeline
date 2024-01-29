import dolfinx
import ufl
from petsc4py import PETSc

from Thirdparty.pyadjoint import pyadjoint
from Thirdparty.pyadjoint.pyadjoint import annotate_tape, get_working_tape, stop_annotating
from Thirdparty.pyadjoint.pyadjoint import create_overloaded_object
from Thirdparty.pyadjoint.pyadjoint.tape import no_annotations

from .type_Function import Function
from ..dolfinx_utils import MeshUtils, AssembleUtils

"""
output is pyadjoint.adjFloat
AssembleBlock.add_output
    -> AssembleBlock.will_add_as_output
        -> BlockVariable._ad_will_add_as_output
            -> OverloadType._ad_will_add_as_dependency [False]
AssembleBlock._outputs.append(obj)
"""


def assemble(form: ufl.form.Form, domain: dolfinx.mesh.Mesh, **kwargs):
    annotate = annotate_tape(kwargs)

    with stop_annotating():
        output = AssembleUtils.assemble_scalar(dolfinx.fem.form(form))

    output = create_overloaded_object(output)

    if annotate:
        block = AssembleBlock(form, domain, **kwargs)
        tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.block_variable)

    return output


class AssembleBlock(pyadjoint.Block):
    def __init__(self, form: ufl.form.Form, domain: dolfinx.mesh.Mesh, **kwargs):
        super(AssembleBlock, self).__init__()
        self.form = form

        self.add_dependency(domain)
        for c in self.form.coefficients():
            self.add_dependency(c, no_duplicates=True)

    def __str__(self):
        return str(self.form)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        """
        Update Form with latest result
        """
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        """
        adj_inputs: input from pre-block
        block_variable: relevant input
        """
        form: ufl.form.Form = prepared
        adj_input: PETSc.Vec = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, dolfinx.fem.Function):
            # 当以TestFunction作为泛函，即求解梯度
            dc = ufl.TestFunction(c.function_space)
            dform = ufl.derivative(form, c_rep, dc)

        elif isinstance(c, dolfinx.mesh.Mesh):
            X = MeshUtils.define_coordinate(c_rep)
            coordinate_space = X.ufl_domain().ufl_coordinate_element()
            function_space = dolfinx.fem.FunctionSpace(c, coordinate_space)
            du = ufl.TestFunction(function_space)

            dform = ufl.derivative(form, X, du)

        else:
            raise NotImplementedError

        output: PETSc.Vec = AssembleUtils.assemble_vec(dolfinx.fem.form(dform))
        output.array[:] = output.array * adj_input
        return output

    @no_annotations
    def evaluate_adj(self, markings=False):
        super().evaluate_adj(markings)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        """
        block_variable: relevant output
        idx: the index of relevant output
        """
        form: ufl.form.Form = prepared
        dform: ufl.form.Form = 0.0
        dform_shape: ufl.form.Form = 0.0

        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue

            if isinstance(c_rep, dolfinx.mesh.Mesh):
                X = MeshUtils.define_coordinate(c_rep)
                dform_shape += ufl.derivative(form, X, tlm_value)

            elif isinstance(c_rep, Function):
                dform += ufl.derivative(form, c_rep, tlm_value)

            else:
                raise NotImplementedError

        if not isinstance(dform, float):
            dform = AssembleUtils.assemble_scalar(dolfinx.fem.form(dform))

        if not isinstance(dform_shape, float):
            dform_shape = AssembleUtils.assemble_scalar(dolfinx.fem.form(dform_shape))

        return dform + dform_shape

    @no_annotations
    def evaluate_tlm(self, markings=False):
        super().evaluate_tlm(markings)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form: ufl.form.Form = prepared
        output = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
        output = create_overloaded_object(output)
        return output

    def recompute(self, markings=False):
        """
        Updated result will be saved to OverloadType.checkpoint
        """
        super().recompute(markings)
