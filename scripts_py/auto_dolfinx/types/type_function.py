import re
import typing

import dolfinx
from dolfinx import la
import numpy.typing as npt
import ufl

import numpy as np
from petsc4py import PETSc

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
from thirdparty.pyadjoint.pyadjoint import AdjFloat

from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.sparse_utils import PetscUtils


@register_overloaded_type
class DolfinxFunction(dolfinx.fem.Function, OverloadedType):
    riesz_ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    def __init__(
        self, 
        V: dolfinx.fem.FunctionSpace,
        x: typing.Optional[la.Vector] = None,
        name: typing.Optional[str] = None,
        dtype: typing.Optional[npt.DTypeLike] = None
    ):
        super(DolfinxFunction, self).__init__(V, x, name, dtype)
        OverloadedType.__init__(self)

    @classmethod
    def _ad_init_object(cls, obj: np.ndarray):
        return obj

    def _ad_create_checkpoint(self) -> dolfinx.fem.Function:
        new_f: DolfinxFunction = DolfinxFunction(self.function_space, name=f"{self.name}_checkpoint")  # type: ignore
        new_f.x.array[:] = self.x.array.copy()
        return new_f
    
    def _ad_restore_at_checkpoint(self, checkpoint: dolfinx.fem.Function) -> dolfinx.fem.Function:
        # return checkpoint  # TODO: check if this is correct
        self.x.array[:] = checkpoint.x.array.copy()
        return self
    
    def _ad_copy(self) -> dolfinx.fem.Function:
        ret = get_overloaded_class(dolfinx.fem.Function)(self.function_space, name=f"{self.name}_copy")
        ret.x.array[:] = self.x.array.copy()
        return ret
    
    @staticmethod
    def _ad_to_list(m):
        raise NotImplementedError
    
    def assign(self, other: typing.Union['DolfinxFunction', AdjFloat]) -> None:
        if isinstance(other, (float, AdjFloat)):
            self.x.array[:] = np.ones_like(self.x.array.shape) * other
        elif isinstance(other, DolfinxFunction):
            self.interpolate(other)
        else:
            raise ValueError(f"Unsupported other type: {type(other)}")
    
    @no_annotations
    def _ad_convert_riesz(self, value, riesz_map: dict|None=None):
        ''' Reference: Riesz Representation Theorem
        Every continuous linear functional $ \\ell $ on a Hilbert space $ V $ can be uniquely represented as 
        an inner product with a fixed element $ u_h \\in V $ (the Riesz representative), 
        evaluated against any test function $ v \\in V $
        '''
        
        if isinstance(value, dolfinx.fem.Function):
            value_np = value.x.array.copy()
        elif isinstance(value, np.ndarray):
            value_np = value
        elif isinstance(value, PETSc.Vec):  # type: ignore
            value_np = value.array
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")
        
        if riesz_map is None:
            return value_np
        
        riesz_representation = riesz_map["riesz_representation"]

        if riesz_representation == "l2":
            return value_np
        
        elif riesz_representation == "L2":
            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)

            a_form = ufl.inner(u, v) * ufl.dx
            a_mat = AssembleUtils.assemble_mat(a_form, [], method='lift')
            b_vec = PetscUtils.create_vec(size=value_np.shape[0])
            b_vec.array[:] = value_np
            
            solver = LinearProblemSolver.create_petsc_solver(
                comm = riesz_map['comm'], 
                solver_setting = riesz_map.get('ksp_option', self.riesz_ksp_option), 
                mat = a_mat
            )
            res_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b_vec, 
                a_mat=a_mat, 
                solver=solver, 
                setOperators=False, 
                with_residual=riesz_map.get('with_residual', False)
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve Riesz problem.")
            value_np = res_dict['res'].array

        elif riesz_representation == "H1":
            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)

            a_form = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            a_mat = AssembleUtils.assemble_mat(a_form, [], method='lift')
            b_vec = PetscUtils.create_vec(size=value_np.shape[0])
            b_vec.array[:] = value_np
            
            solver = LinearProblemSolver.create_petsc_solver(
                comm = riesz_map['comm'], 
                solver_setting = riesz_map.get('ksp_option', self.riesz_ksp_option), 
                mat = a_mat
            )
            res_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b_vec, 
                a_mat=a_mat, 
                solver=solver, 
                setOperators=False, 
                with_residual=riesz_map.get('with_residual', False)
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve Riesz problem.")
            value_np = res_dict['res'].array

        else:
            raise NotImplementedError(f"Unknown Riesz representation: {riesz_representation}")
        
        return value_np
