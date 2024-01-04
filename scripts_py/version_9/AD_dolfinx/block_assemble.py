import numpy as np
import dolfinx
import ufl

from Thirdparty.pyadjoint import pyadjoint
from Thirdparty.pyadjoint.pyadjoint import annotate_tape, get_working_tape, stop_annotating
from Thirdparty.pyadjoint.pyadjoint import create_overloaded_object

from .backend_dolfinx import ComputeUtils, SolverUtils
from .type_Mesh import Mesh
from .type_Function import Function


def assemble(form: ufl.form.Form, domain: dolfinx.mesh.Mesh, **kwargs):
    annotate = annotate_tape(kwargs)

    with stop_annotating():
        output = ComputeUtils.compute_integral(form)

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
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        form: ufl.form.Form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, dolfinx.fem.Function):
            dc = ufl.TestFunction(c.function_space)
        else:
            raise NotImplementedError

        # dc作为1个变量被嵌入dform内，assemble_vector将求取dc的值由于为等式约束，dc即为相应的泛函
        dform = ufl.derivative(form, c_rep, dc)
        output = SolverUtils.assemble_vec(dolfinx.fem.form(dform))

        # ------ Debug
        # print(f"Is A_mat Inf: {np.any(np.isinf(output.array))} or Nan: {np.any(np.isnan(output.array))}")
        # -----------------

        output.array[:] = output.array * adj_input
        return output

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        form: ufl.form.Form = prepared
        dform: ufl.form.Form = 0.0
        dform_shape = 0.0

        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue

            if isinstance(c_rep, Mesh):
                raise NotImplementedError
            elif isinstance(c_rep, Function):
                dform += ufl.derivative(form, c_rep, tlm_value)

        if not isinstance(dform, float):
            dform = ComputeUtils.compute_integral(dform)

        return dform + dform_shape

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form: ufl.form.Form = prepared
        output = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
        output = create_overloaded_object(output)
        return output
