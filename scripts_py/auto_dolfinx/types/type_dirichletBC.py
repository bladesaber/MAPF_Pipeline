from typing import Union, Callable, Any

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
from scripts_py.dolfinx_utils import BoundaryUtils


@register_overloaded_type
class DolfinxDirichletBC(dolfinx.fem.DirichletBC, OverloadedType):
    def __init__(
        self,
        name: str,
        bc_func_space: dolfinx.fem.FunctionSpace,
        value_func_space: dolfinx.fem.FunctionSpace,
        dofs: np.ndarray|list[np.ndarray],
        bc_value: dolfinx.fem.Function,
        is_mixed_element: bool,
    ):
        self.name = name
        self.bc_func_space = bc_func_space
        self.value_func_space = value_func_space
        self.dofs = dofs
        self.bc_value = bc_value
        self.is_mixed_element = is_mixed_element
        super(DolfinxDirichletBC, self).__init__(
            BoundaryUtils.define_dirichlet_bc(
                bc_func_space=bc_func_space,
                value_func_space=value_func_space,
                dofs=dofs,
                bc_value=bc_value,
                is_mixed_element=is_mixed_element,
                wrap_dirichlet_bc=False
            )
        )
        OverloadedType.__init__(self)

    @classmethod
    def _ad_init_object(cls, obj: 'DolfinxDirichletBC'):
        new_obj = cls(
            name=obj.name,
            bc_func_space=obj.bc_func_space,
            value_func_space=obj.value_func_space,
            dofs=obj.dofs,
            bc_value=obj.bc_value,
            is_mixed_element=obj.is_mixed_element
        )
        return new_obj

    def _ad_create_checkpoint(self) -> dolfinx.fem.Function:
        new_f: dolfinx.fem.Function = dolfinx.fem.Function(self.value_func_space, name=f"{self.name}_checkpoint")  # type: ignore
        new_f.x.array[:] = self.bc_value.x.array.copy()
        return new_f
    
    def _ad_restore_at_checkpoint(self, checkpoint: dolfinx.fem.Function) -> dolfinx.fem.DirichletBC:
        self._cpp_object = BoundaryUtils.define_dirichlet_bc(
            bc_func_space=self.bc_func_space,
            value_func_space=self.value_func_space,
            dofs=self.dofs,
            bc_value=checkpoint,
            is_mixed_element=self.is_mixed_element,
            wrap_dirichlet_bc=False
        )
        return self

    def _ad_copy(self) -> dolfinx.fem.DirichletBC:
        ret = get_overloaded_class(dolfinx.fem.DirichletBC)(
            bc_func_space=self.bc_func_space,
            value_func_space=self.value_func_space,
            dofs=self.dofs,
            bc_value=self.bc_value,
            is_mixed_element=self.is_mixed_element
        )
        return ret
    
    @staticmethod
    def _ad_to_list(m):
        raise NotImplementedError
    