from typing import Union, Callable, Any

import dolfinx
import ufl
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_arguments

import numpy as np
from petsc4py import PETSc

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.dolfinx_utils import BoundaryUtils
from scripts_py.mesh_utils import MeshUtils
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.sparse_utils import PetscUtils
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction


def equation_solve(
    domain: dolfinx.mesh.Mesh,
    F_form: ufl.Form,
    state: dolfinx.fem.Function,
    adjoint: dolfinx.fem.Function,
    bcs: list[dolfinx.fem.DirichletBC],
    is_linear: bool,
    tlm_ksp_option={
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    },
    adj_ksp_option={
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    },
    forward_ksp_option={
        'ksp_setting': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        'snes_setting': {}
    },
    tlm_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    adj_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    forward_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    tape: Tape = None,  # type: ignore
    with_block: bool = False
):
    if tape is None:
        with_annotate = annotate_tape()
    else:
        with_annotate = True
    
    with stop_annotating():
        if is_linear:
            F_bilinear_form = ufl.replace(
                F_form, {
                    state: ufl.TrialFunction(state.function_space),  # type: ignore
                    adjoint: ufl.TestFunction(adjoint.function_space),  # type: ignore
                }
            )
            lhs: ufl.Form = ufl.lhs(F_bilinear_form)  # type: ignore
            rhs: ufl.Form = ufl.rhs(F_bilinear_form)
            
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=domain.comm, 
                uh=state, 
                a_form=lhs,
                l_form=rhs, 
                bcs=bcs, 
                ksp_option=forward_ksp_option['ksp_setting'],
                mat_assemble_method=forward_kwargs['mat_assemble_method'],
                with_residual=forward_kwargs['with_residual']
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve recompute problem.")
            
            output = res_dict['res']
            
        else:
            F_recompute_linear_form: ufl.Form = ufl.replace(
                F_form, {adjoint: ufl.TestFunction(adjoint.function_space)}  # type: ignore
            )
            
            jacobi_form = ufl.derivative(
                F_recompute_linear_form,
                state,
                ufl.TrialFunction(state.function_space),
            )
            jacobi_form_dolfin = DolfinUtils.ufl_to_dolfin(jacobi_form)
            
            res_dict = NonLinearProblemSolver.solve_fea_petsc(
                comm=domain.comm,
                f_form=F_recompute_linear_form,
                uh=state,
                jacobi_form=jacobi_form_dolfin,  # type: ignore
                bcs=bcs,
                mat_assemble_method=forward_kwargs['mat_assemble_method'],
                snes_setting=forward_ksp_option['snes_setting'],
                ksp_setting=forward_ksp_option['ksp_setting'],
                criterion={},
                with_residual=forward_kwargs['with_residual'],
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve recompute problem.")

            output = res_dict['res']
    
    if with_annotate:
        block = EquationSolveBlock(
            domain=domain,
            F_form=F_form,
            state=state,
            adjoint=adjoint,
            bcs=bcs, # type: ignore
            is_linear=is_linear,
            tlm_ksp_option=tlm_ksp_option,
            adj_ksp_option=adj_ksp_option,
            forward_ksp_option=forward_ksp_option,
            tlm_kwargs=tlm_kwargs,
            adj_kwargs=adj_kwargs,
            forward_kwargs=forward_kwargs,
        )
        
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.create_block_variable())  # type: ignore
    
    if with_block and with_annotate:
        return output, block
    return output


class EquationSolveBlock(Block):
    def __init__(
        self, 
        domain: dolfinx.mesh.Mesh,
        F_form: ufl.Form,
        state: dolfinx.fem.Function,
        adjoint: dolfinx.fem.Function,
        bcs: list[DolfinxDirichletBC],
        is_linear: bool,
        tlm_ksp_option: dict, 
        adj_ksp_option: dict, 
        forward_ksp_option: dict,
        tlm_kwargs: dict,
        adj_kwargs: dict,
        forward_kwargs: dict,
    ):
        super().__init__()
        assert state in F_form.coefficients(), "state must be in F_form.coefficients()"  # type: ignore
        assert adjoint in F_form.coefficients(), "adjoint must be in F_form.coefficients()"  # type: ignore
        assert len(extract_arguments(F_form)) == 0, "F_form must be bilinear form or non-linear form."
        
        self.domain = domain
        self.state = state
        self.adjoint = adjoint
        self.state_trial = ufl.TrialFunction(state.function_space)  # type: ignore
        self.adjoint_test = ufl.TestFunction(adjoint.function_space)  # type: ignore
        self.F_form = F_form
        self.is_linear = is_linear
        
        # why bilinear version of F_form is necessary?
        # because computation of derivative of F_form needs to have TestFunction
        # if F_form is non-linear form, we need to convert it to linear form first
        if self.is_linear:
            F_bilinear_form = ufl.replace(
                self.F_form, {
                    self.state: self.state_trial,  # type: ignore
                    self.adjoint: self.adjoint_test # type: ignore
                }
            )
            self.lhs_form: ufl.Form = ufl.lhs(F_bilinear_form)  # type: ignore
            self.rhs_form: ufl.Form = ufl.rhs(F_bilinear_form)  # type: ignore
            
            # NOTE: be careful, adjoint isn't dependent or output, state is output
            for c in self.lhs_form.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            for c in self.rhs_form.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            
        else:
            self.lhs_form: ufl.Form = ufl.replace(
                self.F_form, {self.adjoint: self.adjoint_test}  # type: ignore
            )
            self.rhs_form: ufl.Form = 0.0  # type: ignore
            
            # NOTE: be careful, adjoint isn't dependent or output, state is output and dependent
            for c in self.lhs_form.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
        
        self.bcs = bcs
        self.tlm_ksp_option = tlm_ksp_option
        self.adj_ksp_option = adj_ksp_option
        self.forward_ksp_option = forward_ksp_option
        self.tlm_kwargs = tlm_kwargs
        self.adj_kwargs = adj_kwargs
        self.forward_kwargs = forward_kwargs
        
        self.add_dependency(domain, no_duplicates=True)
        for bc in self.bcs:
            self.add_dependency(bc, no_duplicates=True)
    
    def create_F_linear_form(self):
        if self.is_linear:
            _uh = DolfinxFunction(self.state.function_space, name='_uh')
            # ufl.action will replace ufl.TrialFunction with _uh
            F_linear_form = ufl.action(self.lhs_form, _uh) - self.rhs_form
        else:
            _uh = self.state
            F_linear_form = self.lhs_form
        
        replace_map = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in F_linear_form.coefficients():
                replace_map[coeff] = block_variable.saved_output

        # replace _uh with latest result
        replace_map[_uh] = self.get_outputs()[0].saved_output
        return ufl.replace(F_linear_form, replace_map)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        F_linear_form = self.create_F_linear_form()
        output_var = self.get_outputs()[0]
        u: dolfinx.fem.Function = output_var.output
        
        # since ufl.action from create_F_form, we add trail function here
        # and dFdu will certainly be linear form: dF/du * variation_u, variation_u is ufl.TrialFunction and what we want
        dFdu = ufl.derivative(F_linear_form, output_var.saved_output, ufl.TrialFunction(u.function_space))
        return {'F_linear_form': F_linear_form, 'dFdu': dFdu}

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared: dict) -> np.ndarray:
        '''
        if F(u, v) = 0, then
            => dF/du * variation_u + dF/dv * variation_v = 0
            => dF/du * variation_u = -dF/dv * variation_v
            => dF/du is bilinear matrix, dF/dv is linear vector, variation_u is TrialFunction
        '''
        # F_linear_form contains ufl.TestFunction
        F_linear_form: ufl.Form = prepared["F_linear_form"]
        
        bcs = []
        dFdm: ufl.Form = 0.0  # type: ignore
        dFdm_shape:  ufl.Form = 0.0  # type: ignore
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output
            
            if isinstance(c, DolfinxDirichletBC):
                if tlm_value is None:
                    new_bc = BoundaryUtils.create_homogenize_bc(
                        bc_func_space=c.bc_func_space,
                        value_func_space=c.value_func_space,
                        dofs=c.dofs,  # type: ignore
                        is_mixed_element=c.is_mixed_element,
                        value_type=c.bc_value
                    )
                    bcs.append(new_bc)
                else:
                    raise NotImplementedError("No Method")
            
            if tlm_value is None:
                continue
            
            if isinstance(c, dolfinx.mesh.Mesh):
                coordinate_element = coord.ufl_domain().ufl_coordinate_element()  # type: ignore
                coordinate_func_space = dolfinx.fem.functionspace(self.domain, coordinate_element)
                
                tlm_value_fun = dolfinx.fem.Function(coordinate_func_space)
                if isinstance(tlm_value, np.ndarray):
                    tlm_value_fun.x.array[:] = tlm_value  # type: ignore
                elif isinstance(tlm_value, PETSc.Vec):    # type: ignore
                    tlm_value_fun.x.array[:] = tlm_value.array    # type: ignore
                elif isinstance(tlm_value, (dolfinx.fem.Function)):
                    tlm_value_fun = tlm_value
                else:
                    raise NotImplementedError("No Method")
                    
                coord = MeshUtils.define_coordinate(c)
                dFdm_shape += ufl.derivative(-F_linear_form, coord, tlm_value_fun)

            elif isinstance(c, dolfinx.fem.Function):
                tlm_value_fun = dolfinx.fem.Function(c.function_space)
                if isinstance(tlm_value, np.ndarray):
                    tlm_value_fun.x.array[:] = tlm_value  # type: ignore
                elif isinstance(tlm_value, PETSc.Vec):    # type: ignore
                    tlm_value_fun.x.array[:] = tlm_value.array    # type: ignore
                elif isinstance(tlm_value, (dolfinx.fem.Function)):
                    tlm_value_fun = tlm_value
                else:
                    raise NotImplementedError("No Method")
                
                dFdm += ufl.derivative(-F_linear_form, c_rep, tlm_value_fun)
            
            else:
                raise ValueError(f"Unsupported dependency type: {type(c)}")
        
        dFdu: ufl.Form = prepared["dFdu"]
        
        du = dolfinx.fem.Function(self.state.function_space)
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.domain.comm,
            uh=du,  # type: ignore
            a_form=dFdu,
            l_form=dFdm + dFdm_shape,
            bcs=bcs,
            mat_assemble_method=self.tlm_kwargs['mat_assemble_method'],
            ksp_option=self.tlm_ksp_option,
            with_residual=self.tlm_kwargs['with_residual'],
        )
        if not res_dict['is_solved']:
            raise ValueError(f"Failed to solve TLM problem.")
        
        return res_dict['res'].array

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):        
        '''
        reference: https://www.firedrakeproject.org/adjoint.html
        change from commit id: 7c836c546d2d21f05d5234fe7f7ea1538e8d866d
        '''
        output_var = self.get_outputs()[0]
        u = output_var.output
        u_rep = output_var.saved_output
        
        F_linear_form = self.create_F_linear_form()
        dFdu = ufl.derivative(F_linear_form, u_rep, ufl.TrialFunction(u.function_space))
        
        # --------------------------------
        '''
        # the same as below code, but use dolfinx.fem.Function instead of PETSc.Vec
        dJdu_vec: dolfinx.fem.Function = dolfinx.fem.Function(u.function_space, name="dJdu")  # type: ignore
        if isinstance(adj_inputs[0], PETSc.Vec):  # type: ignore
            dJdu_vec.x.array[:] = adj_inputs[0].array  # type: ignore
        elif isinstance(adj_inputs[0], np.ndarray):
            dJdu_vec.x.array[:] = adj_inputs[0]
        elif isinstance(adj_inputs[0], dolfinx.fem.Function):
            dJdu_vec = adj_inputs[0]
        else:
            raise ValueError(f"Unsupported adjoint input type: {type(adj_inputs[0])}")
        
        v_test = ufl.TestFunction(u.function_space)
        rhs = ufl.inner(dJdu_vec, v_test) * ufl.dx
        lhs = dFdu_T
        '''
        # --------------------------------
        
        # ------ assemble dJdu vector ------        
        dJdu_vec: PETSc.Vec = PetscUtils.create_vec(size=u.x.array.shape[0])  # type: ignore        
        if isinstance(adj_inputs[0], PETSc.Vec):  # type: ignore
            dJdu_vec = adj_inputs[0].copy()
        elif isinstance(adj_inputs[0], np.ndarray):
            dJdu_vec.array[:] = adj_inputs[0].copy()
        elif isinstance(adj_inputs[0], dolfinx.fem.Function):
            dJdu_vec.array[:] = adj_inputs[0].x.array.copy()
        else:
            raise ValueError(f"Unsupported adjoint input type: {type(adj_inputs[0])}")
        dJdu_vec = dJdu_vec * -1.0
        
        bcs = []
        for c in self.bcs:
            new_bc = BoundaryUtils.create_homogenize_bc(
                bc_func_space=c.bc_func_space,
                value_func_space=c.value_func_space,
                dofs=c.dofs,  # type: ignore
                is_mixed_element=c.is_mixed_element,
                value_type=c.bc_value
            )
            bcs.append(new_bc)
                
        # can't clean vec because dJdu_vec is adjoint input
        BoundaryUtils.apply_boundary_to_vec(
            dJdu_vec, bcs, DolfinUtils.ufl_to_dolfin(dFdu), 
            clean_vec=False, method=self.adj_kwargs['mat_assemble_method']
        )
        
        # ------ assemble dFdu_T matrix ------
        # ufl.adjoint mean swap ufl.TrialFunction and ufl.TestFunction, the same as transpose of dFdu matrix
        dFdu_T: ufl.Form = ufl.adjoint(dFdu)  # type: ignore
        ajoint_mat: PETSc.Mat = AssembleUtils.assemble_mat(  # type: ignore
            DolfinUtils.ufl_to_dolfin(dFdu_T), bcs,
            method=self.adj_kwargs['mat_assemble_method'],
            diagonal=1.0,
        )
        
        # ------ solve adjoint variable ------
        solver = LinearProblemSolver.create_petsc_solver(self.domain.comm, self.adj_ksp_option, ajoint_mat)
        
        if 'critical_method' in self.adj_kwargs.keys():
            with_residual = self.adj_kwargs['with_residual']
            if self.adj_kwargs['critical_method'] == 'residual':
                with_residual = True
        else:
            with_residual = self.adj_kwargs['with_residual']
        
        res_dict = LinearProblemSolver.solve_by_petsc(
            dJdu_vec, ajoint_mat, solver, 
            setOperators=False, 
            with_residual=with_residual
        )
        if not res_dict['is_solved']:
            # TODO: improve future
            if self.adj_kwargs['critical_method'] == 'residual':
                residual_error = res_dict['max_residual']
                if residual_error > 1e-8:
                    raise ValueError(f"Failed to solve adjoint problem with residual error: {residual_error}")
            else:
                raise ValueError(f"Failed to solve adjoint problem.")
        
        return {'F_linear_form': F_linear_form, 'adj_solution': res_dict['res']}

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared: dict) -> np.ndarray | None:
        '''
        in fact, we want to compute dJ/dm = dJ/dm + dF/dm
        here only consider dF/dm, since dJ/dm is computed from other block in the computation graph
        '''
        if not self.is_linear and self.state == block_variable.output:
            # NOTE: be careful, we are not able to calculate derivatives wrt itself.
            return None
        
        F_linear_form: ufl.Form = prepared["F_linear_form"]
        adj_solution: PETSc.Vec = prepared["adj_solution"]  # type: ignore
        adj_fun = dolfinx.fem.Function(self.state.function_space, name="adj_fun")  # type: ignore
        adj_fun.x.array[:] = adj_solution.array  # type: ignore
        
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, dolfinx.fem.Function):
            dFdm = ufl.derivative(F_linear_form, c_rep, ufl.TrialFunction(c.function_space))                   
            dFdm_T = ufl.adjoint(dFdm)  # transpose of dFdm
            # TODO: is * operator always correct?
            dFdm_adj = dFdm_T * adj_fun  # type: ignore
            dFdm_vec = AssembleUtils.assemble_vec(dFdm_adj)
        
        elif isinstance(c, dolfinx.mesh.Mesh):
            coord = MeshUtils.define_coordinate(c_rep)
            coordinate_space = coord.ufl_domain().ufl_coordinate_element()  # type: ignore
            function_space = dolfinx.fem.functionspace(c, coordinate_space)
            
            '''
            # if form is bilinear form, ufl.action will replace ufl.TrialFunction with adj_fun
            # if form is linear form, ufl.action will replace ufl.TestFunction with adj_fun
            F_normal_form = ufl.action(F_linear_form, adj_fun)
            dFdm = ufl.derivative(F_normal_form, coord, ufl.TestFunction(function_space))
            dFdm_vec = AssembleUtils.assemble_vec(dFdm)
            '''

            # the same as above code, but use ufl.derivative directly
            dFdm_tmp = ufl.derivative(F_linear_form, coord, ufl.TrialFunction(function_space))
            dFdm_mat_tmp = AssembleUtils.assemble_mat(dFdm_tmp, [])
            dFdm_mat_tmp_T = dFdm_mat_tmp.transpose()
            # print(dFdm_mat_T.getSize(), adj_solution.getSize())
            dFdm_vec = dFdm_mat_tmp_T * adj_solution
            # print(np.isclose(dFdm_vec.array, dFdm_vec_tmp.array).all())
        
        else:
            raise ValueError(f"Unsupported dependency type: {type(c)}")
        
        return dFdm_vec.array
    
    def prepare_recompute_component(self, inputs, relevant_outputs):
        uh = dolfinx.fem.Function(self.state.function_space, name="recompute_uh")
        
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output
            if isinstance(c, (dolfinx.fem.DirichletBC, DolfinxDirichletBC)):
                bcs.append(c_rep)
        
        replace_map = {}
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output
                        
            if isinstance(c, (dolfinx.fem.DirichletBC, DolfinxDirichletBC)):
                continue

            if c in self.F_form.coefficients():
                c_rep = block_variable.saved_output     
                replace_map[c] = c_rep
                if c == self.state:
                    replace_map[c] = uh

        F_recompute_form = ufl.replace(self.F_form, replace_map)
        return {'F_recompute_form': F_recompute_form, 'uh': uh, 'bcs': bcs}
    
    def recompute_component(self, inputs, block_variable, idx, prepared: dict) -> dolfinx.fem.Function:
        F_recompute_form: ufl.Form = prepared['F_recompute_form']
        bcs: list[dolfinx.fem.DirichletBC] = prepared['bcs']
        uh: dolfinx.fem.Function = prepared['uh']
        
        if self.is_linear:
            F_recompute_bilinear_form = ufl.replace(
                F_recompute_form, {
                    self.state: self.state_trial,  # type: ignore
                    self.adjoint: self.adjoint_test # type: ignore
                }
            )
            lhs: ufl.Form  = ufl.lhs(F_recompute_bilinear_form)  # type: ignore
            rhs: ufl.Form  = ufl.rhs(F_recompute_bilinear_form)
                        
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.domain.comm, 
                uh=uh, 
                a_form=lhs,
                l_form=rhs, 
                bcs=bcs, 
                ksp_option=self.forward_ksp_option['ksp_setting'],
                mat_assemble_method=self.forward_kwargs['mat_assemble_method'],
                with_residual=self.forward_kwargs['with_residual']
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve recompute problem.")

        else:
            F_recompute_linear_form: ufl.Form = ufl.replace(
                F_recompute_form, {
                    self.adjoint: self.adjoint_test # type: ignore
                }
            )
            
            jacobi_form = ufl.derivative(
                F_recompute_linear_form,
                uh,
                ufl.TrialFunction(uh.function_space),
            )
            jacobi_form_dolfin = DolfinUtils.ufl_to_dolfin(jacobi_form)
            
            res_dict = NonLinearProblemSolver.solve_fea_petsc(
                comm=self.domain.comm,
                f_form=F_recompute_linear_form,
                uh=uh,
                jacobi_form=jacobi_form_dolfin,  # type: ignore
                bcs=bcs,
                mat_assemble_method=self.forward_kwargs['mat_assemble_method'],
                snes_setting=self.forward_ksp_option['snes_setting'],
                ksp_setting=self.forward_ksp_option['ksp_setting'],
                criterion={},
                with_residual=self.forward_kwargs['with_residual'],
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve recompute problem.")

        return res_dict['res']

    