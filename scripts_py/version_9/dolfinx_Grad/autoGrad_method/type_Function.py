import dolfinx
import numpy as np
from petsc4py import PETSc

from Thirdparty.pyadjoint.pyadjoint.overloaded_type import (
    register_overloaded_type, get_overloaded_class
)
from Thirdparty.pyadjoint.pyadjoint.tape import no_annotations

from .type_utils import AuxiliaryType


@register_overloaded_type
class Function(AuxiliaryType, dolfinx.fem.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args,
                                       _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                       block_class=kwargs.pop("block_class", None),
                                       _ad_args=kwargs.pop("_ad_args", None),
                                       # output_block_class=kwargs.pop("output_block_class", None),
                                       # _ad_output_args=kwargs.pop("_ad_output_args", None),
                                       # _ad_outputs=kwargs.pop("_ad_outputs", None),
                                       annotate=kwargs.pop("annotate", True),
                                       **kwargs)
        dolfinx.fem.Function.__init__(self, *args, **kwargs)

    def assign(self, f: dolfinx.fem.Function):
        self.x.array[:] = f.x.array

    def _ad_create_checkpoint(self):
        new_f: Function = Function(self.function_space, name=f"{self.name}_checkpoint")
        new_f.x.array[:] = self.x.array
        return new_f

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_convert_type(self, value, options={}):
        # use to output the derivative as the new obj
        if isinstance(value, PETSc.Vec):
            value: PETSc.Vec
            ret: Function = Function(self.function_space)
            ret.x.array[:] = value.array.copy()
            return ret
        else:
            raise NotImplementedError

    def _ad_copy(self):
        ret = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        ret.x.array[:] = self.x.array
        return ret

    @staticmethod
    def _ad_to_list(m: dolfinx.fem.Function):
        # convert var to list in order to used in optimize
        m_list: np.ndarray = m.x.array[:]
        return m_list.tolist()

    @staticmethod
    def _ad_assign_numpy(dst: dolfinx.fem.Function, src: np.array, offset):
        # assign new var to type
        num = dst.vector.size
        m_a_local = src[offset:offset + num]
        dst.x.array[:] = m_a_local
        offset += num
        return dst, offset

    @no_annotations
    def _ad_mul(self, other):
        # r: Function = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        # r.x.array[:] = self.x.array * other
        # return r
        raise NotImplementedError

    @no_annotations
    def _ad_add(self, other):
        # r: Function = get_overloaded_class(dolfinx.fem.Function)(self.function_space)
        # r.x.array[:] = self.x.array + other
        # return r
        raise NotImplementedError
