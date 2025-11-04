import typing

import dolfinx
import ufl

import numpy as np

from thirdparty.pyadjoint.pyadjoint.overloaded_type import (
    register_overloaded_type,
    get_overloaded_class,
    OverloadedType
)
from thirdparty.pyadjoint.pyadjoint.tape import (
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)


'''
OverloadedType被用于维持对象Object, 但Object在不同的时间点/steps下可能会有不同的值, 
因此Block(Block相当于不同的step)会记录BlockVariable来维持该时刻下Object的值, 在Tape遍历时, 会根据BlockVariable来恢复Object的值.
所以只需要维持单一的dolfinx.mesh.Mesh对象, 实际上这个单一的dolfinx.mesh.Mesh表述在不同时间点下的Mesh
BlockVariable总是指向其OverloadedType,而OverloadedType则指向最新的BlockVariable

when add as dep or output, block will create and record a BlockVariable, and new BlockVariable will point the OverloadedType,
and the OverloadedType will point to the new BlockVariable at the same time.
'''

@register_overloaded_type
class DolfinxMesh(dolfinx.mesh.Mesh, OverloadedType):
    def __init__(self, domain: dolfinx.mesh.Mesh):
        super(DolfinxMesh, self).__init__(msh=domain._cpp_object, domain=domain._ufl_domain)  # type: ignore
        OverloadedType.__init__(self)

    @classmethod
    def _ad_init_object(cls, obj: dolfinx.mesh.Mesh):
        new_obj = cls(obj)
        new_obj.geometry.x[:] = obj.geometry.x.copy()
        return new_obj

    def _ad_create_checkpoint(self) -> np.ndarray:
        return self.geometry.x.copy()
    
    def _ad_restore_at_checkpoint(self, checkpoint: np.ndarray) -> dolfinx.mesh.Mesh:
        self.geometry.x[:] = checkpoint
        return self
    
    def _ad_copy(self) -> dolfinx.mesh.Mesh:
        ret = get_overloaded_class(dolfinx.mesh.Mesh)(self)
        ret.geometry.x[:] = self.geometry.x.copy()
        return ret
    
    @staticmethod
    def _ad_to_list(m):
        raise NotImplementedError
    
    def _ad_convert_riesz(self, value, riesz_map=None):
        # TODO: plase adjust later
        return value