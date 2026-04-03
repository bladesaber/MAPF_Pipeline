import shutil
from typing import Literal, Union, Callable
import os

import dolfinx
import ufl
from ufl import inner, grad, div

import numpy as np
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pyvista
import torch

from scripts_py.mesh_utils import MeshUtils, MeshObj
from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, compute_gradient
from thirdparty.pyadjoint.pyadjoint.adjfloat import max, min  # NOTE: please use max and min from pyadjoint
from thirdparty.pyadjoint.pyadjoint.adjfloat import AdjFloat
from scripts_py.vtu_recorder import TensorBoardRecorder

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation
from scripts_py.auto_dolfinx.blocks.block_fipy_cfd_solve import equation_cfd_topoWeight_solve

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_weight import TopoFluidOptimizer
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_weight import AlphaSigmoidFunc, AlphaHeavisideFunc, AlphaExponentialFunc


class TopoFluidWeightOptimizerFipy(TopoFluidOptimizer):
    def __init__(
        self, 
        name: str, 
        msh_file: str,
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        weight_filter_radius: float,
        result_dir: str,
        alpha_fun: dict = {
            'name': 'exponential',
            'args': {
                'alpha_min': 1e-12,
                'alpha_max': 100.0,
                'p': 3.0,
            }
        },
        velocity_degree = 1,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
    ):
        super().__init__(
            name=name,
            mesh=mesh,
            fluid_type=fluid_type,
            viscosity_init=viscosity_init,
            weight_filter_radius=weight_filter_radius,
            result_dir=result_dir,
            alpha_fun=alpha_fun,
            velocity_degree=velocity_degree,
            pressure_degree=pressure_degree,
            velocity_space=velocity_space,
            pressure_space=pressure_space,
        )
        
        self.msh_file = msh_file
        self.simulation_dir = os.path.join(self.result_dir, 'simulation')
        
        self.fipy_log_dir = os.path.join(self.result_dir, 'fipy_log')
        if os.path.exists(self.fipy_log_dir):
            shutil.rmtree(self.fipy_log_dir)
        os.makedirs(self.fipy_log_dir)
    
    def get_bcs(self):
        """set wall boundary at last"""        
        return [bry_info for bry_info in sorted(self.boundaries.values(), key=lambda x: x.priority)]
    
    def define_fluid_problem(
        self,
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        simulation_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        velocityRelaxation: float = 0.6,
        pressureRelaxation: float = 0.8,
        tlm_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        adj_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        forward_ksp_option: dict = {
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
            'critical_method': 'residual',
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        forward_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        log_step: int = 2,
        hot_steps: int = 10,
        parallel_num: int = 1,
        use_tqdm: bool = True,
        use_log: bool = True,
    ):
        self.define_fluid_boundary()
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            self.viscosity = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_viscosity")
            self.viscosity.x.array[:] = self.viscosity_init  # type: ignore
                    
            govern_form = self.define_fluid_govern_equation(u, p, v, q, f, self.viscosity)  # type: ignore
            
            if self.is_topo_cell:
                self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            else:
                self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.x.array[:] = self.topo_weight_init  # type: ignore
            self.init_topo_weight(self.topo_weight)
            
            alpha = self.alpha_func(self.topo_weight)  # type: ignore
            govern_form += alpha * inner(u, v) * ufl.dx  # type: ignore
                        
            uh = equation_cfd_topoWeight_solve(
                fluid_type=self.fluid_type,  # type: ignore
                boundaries=self.get_bcs(),  # type: ignore
                msh_file=self.msh_file,
                viscosity=self.viscosity_init,
                velocityRelaxation=velocityRelaxation,
                pressureRelaxation=pressureRelaxation,
                structure_mesh=self.mesh,  # type: ignore
                structure_obs_weight=None,
                log_step=log_step,
                hot_steps=hot_steps,
                simulation_iterations=simulation_iterations,
                convergence_threshold=convergence_threshold,
                save_dir=self.simulation_dir,
                parallel_num=parallel_num,
                use_tqdm=use_tqdm,
                use_log=use_log,
                with_cache=True,
                
                # ------ result convert related ------
                V_to_W_dofs=self.V_to_W_dofs,
                Q_to_W_dofs=self.Q_to_W_dofs,
                V_func_space = self.V,
                Q_func_space = self.Q,
                
                # ------ alpha related ------
                topo_weight_name=self.topo_weight.name,  # type: ignore
                alpha_func=self.alpha_func,  # type: ignore
                
                # ------ autodolfinx related ------
                F_form=govern_form,
                state=uh,  # type: ignore
                adjoint=vq,  # type: ignore
                tlm_ksp_option=tlm_ksp_option,
                adj_ksp_option=adj_ksp_option,
                tlm_kwargs=tlm_kwargs,
                adj_kwargs=adj_kwargs,
                tape=self.tape,
            )

            self.uh = uh  # type: ignore
            self.f = f  # type: ignore
            u, p = ufl.split(self.uh)  # type: ignore
            
            # ------ define object cost ------
            self.obj_cost_args_list = obj_cost_args_list
            for obj_idx, obj_cost_args in enumerate(self.obj_cost_args_list):
                obj_cost_args = self.prepare_cost_args(
                    u=u, p=p, obj_cost_args=obj_cost_args,  # type: ignore
                    viscosity=self.viscosity, alpha=alpha, rho=self.topo_weight,
                )
                if obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                    obj_cost = assemble_scalar(
                        form=(
                            obj_cost_args['dissipation_weight'] * obj_cost_args['min_dissipation_form'] 
                            + obj_cost_args['penalized_weight'] * obj_cost_args['penalized_term']
                        ),
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'FixedFlux':
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    if with_max_dif_cost:
                        max_dif_cost, max_dif_weight = 0.0, 0.0
                        
                    for i, (marker, outlet_item) in enumerate(obj_cost_args['outlet_cost_forms'].items()):
                        target_flux = outlet_item['target_flux']
                        outlet_flux = assemble_scalar(form=outlet_item['flux_form'], domain=self.domain, tape=self.tape)
                        print(f"{marker} flux: {outlet_flux}, target: {target_flux}, inlet_flux: {obj_cost_args['inlet_flux']}")
                                                
                        flux_dif = outlet_flux - target_flux
                        if with_rescale:
                            flux_dif = flux_dif / target_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                            max_dif_weight = outlet_item['weight']
                        
                        if i == 0:
                            obj_cost = outlet_item['weight'] * flux_dif_cost
                        else:
                            obj_cost += outlet_item['weight'] * flux_dif_cost
                            
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                else:
                    raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            
                if obj_idx == 0:
                    all_obj_cost = obj_cost
                else:
                    all_obj_cost += obj_cost  # type: ignore

            # ------ define regularization cost ------
            self.reg_cost_args = self.prepare_reg_args(reg_cost_args)
            topo_volume = assemble_scalar(self.reg_cost_args['vol_form'], domain=self.domain, tape=self.tape)
            
            align_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
            if self.reg_cost_args['target_volume'] < align_volume:
                align_volume = align_volume * 0.99
            else:
                align_volume = align_volume * 1.01
            align_volume = AdjFloat(align_volume)
            
            if self.reg_cost_args['reg_rescale']:
                volume_dif = topo_volume / align_volume - 1.0  # type: ignore
                reg_cost = self.reg_cost_args['weight'] * (volume_dif ** 2)
            else:
                reg_cost = self.reg_cost_args['weight'] * ((topo_volume - align_volume) ** 2)  # type: ignore
            
            loss_all = all_obj_cost + reg_cost
            
            # ------ define optimization problem ------
            self.control_phi = Control(self.topo_weight)  # type: ignore
            self.control_align_volume = Control(align_volume)  # type: ignore
            self.opt_problem = ReducedFunctional(loss_all, [self.control_phi, self.control_align_volume])
        
        self.record_Upw(0)
        loss_dict, log_dict = self.eval()
        print('[Log]: Initial loss:', loss_dict)
    
    def eval(self):
        eval_dict = {}
        log_dict = {'cost': {}}
        
        # ------ compute object cost ------
        for obj_cost_args in self.obj_cost_args_list:
            if obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                min_dissipation_cost = obj_cost_args['dissipation_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['min_dissipation_form'])
                penalized_cost = obj_cost_args['penalized_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['penalized_term'])
                eval_dict['MinDissCost'] = min_dissipation_cost
                eval_dict['PenalizedCost'] = penalized_cost
                log_dict['cost']['MinDissCost'] = eval_dict['MinDissCost']
                log_dict['cost']['PenalizedCost'] = eval_dict['PenalizedCost']
            
            elif obj_cost_args['style'] == 'FixedFlux':
                if 'flux' not in log_dict.keys():
                    log_dict['flux'] = {}
                if 'flux_cost' not in log_dict.keys():
                    log_dict['flux_cost'] = {}
                
                outlet_cost_list = []
                if obj_cost_args['with_max_dif_cost']:
                    max_dif_cost, max_dif_weight = 0.0, 0.0
                
                for (marker, outlet_item) in zip(obj_cost_args['outlet_markers'], obj_cost_args['outlet_cost_forms'].values()):
                    outlet_flux = AssembleUtils.assemble_scalar(form=outlet_item['flux_form'])
                    log_dict['flux'][f'{marker}_flux'] = outlet_flux
                    
                    flux_dif = outlet_flux - outlet_item['target_flux']
                    if obj_cost_args['with_rescale']:
                        flux_dif = flux_dif / outlet_item['target_flux']
                    flux_cost = flux_dif ** 2
                    outlet_cost_list.append(outlet_item['weight'] * flux_cost)
                    log_dict['flux_cost'][f'{marker}_flux_cost'] = outlet_item['weight'] * flux_cost
                    
                    if obj_cost_args['with_max_dif_cost']:
                        max_dif_cost = max(max_dif_cost, flux_cost)
                        max_dif_weight = outlet_item['weight']
                
                outlet_cost_list = np.array(outlet_cost_list)
                eval_dict['FixedFluxCost'] = float(np.mean(outlet_cost_list))
                log_dict['cost']['FixedFluxCost'] = eval_dict['FixedFluxCost']
                if obj_cost_args['with_max_dif_cost']:
                    eval_dict[f'FixedFluxCost'] += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                    log_dict['cost'][f'FixedFluxCost_max_dif'] = max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)      
            
            else:
                raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            
        # ------ compute regularization cost ------
        align_volume = self.control_align_volume.data()  # type: ignore
        topo_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
        if self.reg_cost_args['reg_rescale']:
            volume_dif = topo_volume / align_volume - 1.0
            eval_dict['RegCost'] = self.reg_cost_args['weight'] * float(volume_dif ** 2)
        else:
            eval_dict['RegCost'] = self.reg_cost_args['weight'] * (float(align_volume - topo_volume) ** 2)
        log_dict['cost']['RegCost'] = eval_dict['RegCost']
        log_dict['volume'] = topo_volume
        
        # ------ aggregate cost ------        
        log_dict['loss_all'] = np.sum(list(eval_dict.values()))
        
        return eval_dict, log_dict
    
    def record_Upw(self, step: int, gradient: np.ndarray|None=None):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
        
        self.recorder.clear(clear_point_data=True, clear_cell_data=True)
        self.recorder.update_point_data('vel', vel_ref_np)
        self.recorder.update_point_data('vel_mag', np.linalg.norm(vel_ref_np, axis=1))
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        
        if self.is_topo_cell:
            self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array)    # type: ignore
            point_scalar_fun = dolfinx.fem.Function(self.point_scalar_func_space)
            point_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_point_data('topo_weight', point_scalar_fun.x.array)  # type: ignore
            
            self.recorder.update_cell_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            self.recorder.update_point_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(point_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            
        else:
            self.recorder.update_point_data('topo_weight', self.topo_weight.x.array)  # type: ignore
            cell_scalar_fun = dolfinx.fem.Function(self.cell_scalar_func_space)
            cell_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_cell_data('topo_weight', cell_scalar_fun.x.array)  # type: ignore
            
            self.recorder.update_point_data('topo_binary', (self.topo_weight.x.array >= 0.5).astype(np.float32))
            self.recorder.update_cell_data('topo_binary', (cell_scalar_fun.x.array >= 0.5).astype(np.float32))  # type: ignore
            
            if self.use_alpha_sigmoid:
                self.recorder.update_point_data(
                    'alpha', AlphaSigmoidFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
                )
                self.recorder.update_cell_data(
                    'alpha', AlphaSigmoidFunc.alpha_fun(cell_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
                )
            else:
                self.recorder.update_point_data(
                    'alpha', AlphaHeavisideFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False)  # type: ignore
                )
                self.recorder.update_cell_data(
                    'alpha', AlphaHeavisideFunc.alpha_fun(cell_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False)  # type: ignore
                )
            
        # ------ record obstacle source ------
        point_obs = np.zeros(shape=(self.mesh.num_points))
        cell_obs = np.zeros(shape=(self.mesh.num_cells))
        for bry_info in self.boundaries.values():
            if bry_info.boundary_type == 'wall':
                if bry_info.boundary_dim == 'cell':
                    point_insides =MeshUtils.extract_entity_dofs(
                        DolfinUtils.point_scalar_function_space(self.mesh.domain), 
                        self.mesh.tdim, self.mesh.cell_tags.find(bry_info.marker)
                    )
                    point_obs[point_insides] = 1
                    
                    cell_insides =MeshUtils.extract_entity_dofs(
                        DolfinUtils.cell_scalar_function_space(self.mesh.domain), 
                        self.mesh.tdim, self.mesh.cell_tags.find(bry_info.marker)
                    )
                    cell_obs[cell_insides] = 1
        
        if point_obs.sum() > 0:
            self.recorder.update_point_data('obstacle', point_obs)
            self.recorder.update_cell_data('obstacle', cell_obs)
        
        if gradient is not None:
            self.recorder.update_point_data('gradient', gradient)
        
        self.recorder.save(step)
    
    def run_along_volume(
        self,
        min_lr: float = 1e-2,
        max_lr: float = 1e-1,
        clip_lr=False,
        max_iteration=200,
        recorder: TensorBoardRecorder|None = None,
        result_vtu_file: str|None = None,
        feedback_func: Callable|None = None,
    ) -> dict:
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(0)
        
        topo_phi_tensor = torch.from_numpy(topo_phi.x.array.copy())  # type: ignore
        topo_phi_tensor.requires_grad = True
        optimizer = torch.optim.Adam([topo_phi_tensor], lr=max_lr)  # type: ignore
        
        scheduler = ReduceLROnPlateau(optimizer, min_lr=min_lr, factor=0.5, patience=10)  # type: ignore
        last_lr = scheduler.get_last_lr()[0]
        volume_stages = np.linspace(self.control_align_volume.data(), self.reg_cost_args['target_volume'], max_iteration)  # type: ignore
        
        stop_process = False
        for step, stage_vol in zip(range(max_iteration), volume_stages):
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.6f} lr:{last_lr:.3f} vol:{stage_vol:.3f}]"
                
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.6f}"
                print(log)
                
                if recorder is not None:
                    for name, value in log_dict.items():
                        if isinstance(value, dict):
                            recorder.write_scalars(tag=name, scalar_dict=value, step=step)
                        else:
                            recorder.write_scalar(tag=name, scalar_value=value, step=step)
                
            direction_np = self.compute_gradient(
                adj_input=np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2), 
                with_filter=True, with_norm=True
            )
            
            # ------ update parameters ------
            if clip_lr:
                old_topo_np = topo_phi.x.array.copy()  # type: ignore
                topo_phi_tensor.grad = torch.from_numpy(direction_np)
                optimizer.step()
                
                new_topo_np = topo_phi_tensor.detach().numpy()
                max_update_grad = np.max(np.abs(new_topo_np - old_topo_np))
                lr_limit = last_lr * 2.0
                if max_update_grad > lr_limit:
                    # print(f'[DEBUG]: Reset gradient to {lr_limit}')
                    update_grad = new_topo_np - old_topo_np
                    update_grad = update_grad / np.max(np.abs(update_grad)) * lr_limit
                    new_topo_np = old_topo_np + update_grad
            else:
                topo_phi_tensor.grad = torch.from_numpy(direction_np)
                optimizer.step()
                new_topo_np = topo_phi_tensor.detach().numpy()
            
            new_topo_np = np.maximum(np.minimum(new_topo_np, 1.0 - 1e-16), 1e-16) 
            topo_phi_tensor.data = torch.from_numpy(new_topo_np).to(topo_phi_tensor.dtype)
            topo_phi.x.array[:] = new_topo_np  # type: ignore
            
            # ------ update loss weight ------
            self.control_align_volume.update(stage_vol)
            
            # ------ recompute loss
            loss_current = self.opt_problem(values=[topo_phi, self.control_align_volume.data()])    # type: ignore
            self.record_Upw(step, gradient=direction_np)
            
            scheduler.step(loss_current)
            last_lr = scheduler.get_last_lr()[0]
            
            if feedback_func is not None:
                stop_process = feedback_func(self)
            
            if stop_process:
                break
                    
        if result_vtu_file is not None:
            last_step_record_file = self.recorder.get_step_file_path(step)
            if not os.path.exists(last_step_record_file):
                last_step_record_file = self.recorder.get_step_file_path(step - 1)
            print(f'[DEBUG]: copy {last_step_record_file} to {result_vtu_file}')
            shutil.copy(last_step_record_file, result_vtu_file)

        return {'is_success': True if not stop_process else False}
