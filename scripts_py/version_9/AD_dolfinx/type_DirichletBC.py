import dolfinx
from copy import deepcopy
from petsc4py import PETSc
from typing import Union, Optional
import numpy as np
import numbers
from dolfinx import cpp as _cpp

from Thirdparty.pyadjoint.pyadjoint.overloaded_type import (
    OverloadedType, register_overloaded_type, get_overloaded_class
)
from Thirdparty.pyadjoint.pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from Thirdparty.pyadjoint import pyadjoint

from .type_utils import AuxiliaryType
from .type_Function import Function


def get_bc_value(
        value: Union[Function, dolfinx.fem.Function, dolfinx.fem.Constant, np.ndarray],
        dofs: np.typing.NDArray[np.int32],
        V: Optional[dolfinx.fem.FunctionSpaceBase] = None
):
    """
    Copy from dolfinx.fem.bcs
    """

    if isinstance(value, numbers.Number):
        value = np.asarray(value)

    try:
        dtype = value.dtype
        if dtype == np.float32:
            bctype = _cpp.fem.DirichletBC_float32
        elif dtype == np.float64:
            bctype = _cpp.fem.DirichletBC_float64
        elif dtype == np.complex64:
            bctype = _cpp.fem.DirichletBC_complex64
        elif dtype == np.complex128:
            bctype = _cpp.fem.DirichletBC_complex128
        else:
            raise NotImplementedError(f"Type {value.dtype} not supported.")
    except AttributeError:
        raise AttributeError("Boundary condition value must have a dtype attribute.")

    # Unwrap value object, if required
    if isinstance(value, np.ndarray):
        _value = value
    else:
        try:
            _value = value._cpp_object  # type: ignore
        except AttributeError:
            _value = value

    if V is not None:
        try:
            bc = bctype(_value, dofs, V)
        except TypeError:
            bc = bctype(_value, dofs, V._cpp_object)
    else:
        bc = bctype(_value, dofs)

    return bc


class DirichletBC(AuxiliaryType, dolfinx.fem.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)
        AuxiliaryType.__init__(
            self, *args,
            block_class=DirichletBCBlock,
            _ad_args=args,
            _ad_kwargs=kwargs,
            _ad_floating_active=True,
            annotate=kwargs.pop("annotate", True),
            **kwargs
        )
        dolfinx.fem.DirichletBC.__init__(self, args[0])

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        if len(deps) <= 0:
            return None
        return deps[0]

    def set_value(self, value):
        bc = get_bc_value(value, self.block.dofs, self.block.function_space)
        self._cpp_object = bc

    def _ad_restore_at_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.set_value(checkpoint.saved_output)
        return self


class DirichletBCBlock(pyadjoint.Block):
    def __init__(self, *args, **kwargs):
        pyadjoint.Block.__init__(self)

        if isinstance(args[0], OverloadedType):
            self.add_dependency(args[0])

        self.dofs = args[1]
        self.function_space = args[2]
        self.original_value = args[3]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        bv = self.get_dependencies()[0]
        tlm_input = bv.tlm_value
        if tlm_input is None:
            return None

        bc = dolfinx.fem.dirichletbc(tlm_input, self.dofs, self.function_space)
        return bc

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        raise NotImplementedError

    @no_annotations
    def recompute(self):
        return

    def __str__(self):
        return "DirichletBC block"


def dirichletbc(
        value: Union[Function, dolfinx.fem.Function, dolfinx.fem.Constant, np.ndarray],
        dofs: np.typing.NDArray[np.int32],
        V: Optional[dolfinx.fem.FunctionSpaceBase] = None
) -> DirichletBC:
    bc = get_bc_value(value, dofs, V)
    return DirichletBC(bc, dofs, V, value)
