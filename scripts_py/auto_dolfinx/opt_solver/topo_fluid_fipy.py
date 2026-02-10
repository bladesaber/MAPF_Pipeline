from typing import Literal, Callable
import os
import shutil

import dolfinx
import ufl
from ufl import inner, grad, div

import numpy as np

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, compute_gradient
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape
from thirdparty.pyadjoint.pyadjoint.adjfloat import max, min

from scripts_py.vtu_recorder import TensorBoardRecorder
from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils, BoundaryUtils
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import (
    TopoFluidOptimizer, 
    AutoFluidBoundary, 
    TopoWeightConstraintTerm
)
from scripts_py.auto_dolfinx.blocks.block_fipy_cfd_solve import equation_cfd_topoWeight_solve


class TopoFluidOptimizerFipy(TopoFluidOptimizer):
    def __init__(
        self, 
        name: str,
        msh_file: str,
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        topo_weight_init: float|np.ndarray,
        init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'],
        weight_filter_radius: float,
        result_dir: str,
        alpha_fun_max: float = 100.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
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
            topo_weight_init=topo_weight_init,
            init_topo_weight_method=init_topo_weight_method,
            weight_filter_radius=weight_filter_radius,
            result_dir=result_dir,
            alpha_fun_max=alpha_fun_max,
            alpha_fun_min=alpha_fun_min,
            alpha_fun_p=alpha_fun_p,
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
    
    def copy_fipy_result_log(self, step: int):
        result_file = os.path.join(self.simulation_dir, 'cache', 'result.vtu')
        shutil.copy(result_file, os.path.join(self.fipy_log_dir, f'fipy_iter_{step}.vtu'))
        
    def get_bcs(self):
        """set wall boundary at last"""        
        return [bry_info for bry_info in sorted(self.boundaries.values(), key=lambda x: x.priority)]

    def define_problem(
        self,
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        obstacle_cost_args: dict|None = None,
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
        # adj_ksp_option: dict = {  # TODO: should I use richardson here?
        #     'ksp_type': 'richardson',
        #     'ksp_max_it': 1, # only one iteration = equivalent to preonly
        #     'pc_type': 'lu',
        #     'pc_factor_mat_solver_type': 'mumps',
        #     'mat_mumps_icntl_24': 1,
        #     'mat_mumps_icntl_25': 0,
        # },
        adj_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
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
        log_step: int = 2,
        hot_steps: int = 10,
        parallel_num: int = 1,
        use_tqdm: bool = True,
        use_log: bool = True,
        alpha_filter_ratio=-1.0,
    ):
        # ------ init topo weight boundary constraint terms ------
        if self.init_topo_weight_method != 'none':
            for name, bry_info in self.boundaries.items():
                assert bry_info.boundary_type in ['velocity', 'pressure', 'wall']
                if bry_info.boundary_type in ['velocity', 'pressure']:
                    term = TopoWeightConstraintTerm(
                        name=name, mesh=self.mesh, bry_info=bry_info,
                        init_method='boundary_radius' if self.init_topo_weight_method == 'filter' else 'boundary_raw'
                    )
                elif bry_info.boundary_type == 'wall':
                    term = TopoWeightConstraintTerm(name=name, mesh=self.mesh, init_method='boundary_raw', bry_info=bry_info)
                self.topo_weight_constraint_terms[name] = term
        
        # ------ obstacle part ------
        if obstacle_cost_args is not None:
            obstacle_weight = np.zeros(self.mesh.num_points)
            obstacle_dofs = obstacle_cost_args['obstacle_func'](DolfinUtils.tabulate_dof_coordinates(self.point_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                obstacle_weight[obstacle_dofs] = 1.0
                
            obs_point_mask = DolfinxFunction(self.point_scalar_func_space, name="obstacle_source")
            obs_point_mask.x.array[:] = obstacle_weight  # type: ignore
            obstacle_cost_args['obs_point_mask'] = obs_point_mask
            obstacle_cost_args['is_cell_field'] = False
            
            if obstacle_dofs.shape[0] > 0 and self.init_topo_weight_method != 'none':
                obs_term = TopoWeightConstraintTerm(
                    name='obstacle', mesh=self.mesh, 
                    init_method='obstacle', 
                    obstacle_func=obstacle_cost_args['obstacle_func']
                )
                self.topo_weight_constraint_terms[obs_term.name] = obs_term 
        self.obstacle_cost_args = obstacle_cost_args
        
        self.define_boundary()
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            self.viscosity = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_viscosity")
            self.viscosity.x.array[:] = self.viscosity_init  # type: ignore
                    
            govern_form = self.define_govern_equation(u, p, v, q, f, self.viscosity)  # type: ignore
            
            if self.is_topo_cell:
                self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            else:
                self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.x.array[:] = self.topo_weight_init  # type: ignore
            if self.init_topo_weight_method != 'none':
                self.init_topo_weight(self.topo_weight, method=self.init_topo_weight_method)  # type: ignore
            
            alpha = self.alpha_fun(
                rho=self.topo_weight,  # type: ignore
                alpha_min=self.alpha_fun_min, 
                alpha_max=self.alpha_fun_max, 
                p=self.alpha_fun_p, 
                pow_operator=np.power
            )
            govern_form += alpha * inner(u, v) * ufl.dx
            
            structure_obs_weight = None
            if self.obstacle_cost_args is not None:
                if self.obstacle_cost_args['obs_cell_mask'].sum() > 0:
                    structure_obs_weight = self.obstacle_cost_args['obs_cell_mask']
            
            uh = equation_cfd_topoWeight_solve(
                fluid_type=self.fluid_type,  # type: ignore
                boundaries=self.get_bcs(),
                msh_file=self.msh_file,
                viscosity=self.viscosity_init,
                velocityRelaxation=velocityRelaxation,
                pressureRelaxation=pressureRelaxation,
                structure_mesh=self.mesh,  # type: ignore
                structure_obs_weight=structure_obs_weight,
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
                alpha_fun=self.alpha_fun,
                alpha_min=self.alpha_fun_min,
                alpha_max=self.alpha_fun_max,
                alpha_p=self.alpha_fun_p,
                alpha_filter_ratio=alpha_filter_ratio,
                
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
            self.record_Upw(0)
            u, p = ufl.split(self.uh)  # type: ignore
            
            # ------ define object cost ------
            for obj_idx, obj_cost_args in enumerate(obj_cost_args_list):
                obj_cost_args = self.prepare_cost_args(
                    u=u, p=p, obj_cost_args=obj_cost_args,  # type: ignore
                    viscosity=self.viscosity, alpha=alpha,
                )
                if obj_cost_args['style'] == 'MinDissipation':
                    obj_cost = assemble_scalar(
                        form=obj_cost_args['weight'] * obj_cost_args['min_dissipation_form'], 
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                    obj_cost = assemble_scalar(
                        form=(
                            obj_cost_args['dissipation_weight'] * obj_cost_args['min_dissipation_form'] 
                            + obj_cost_args['penalized_weight'] * obj_cost_args['penalized_term']
                        ),
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'MinPressure':
                    obj_cost = assemble_scalar(
                        form=obj_cost_args['weight'] * obj_cost_args['pressure_form'],
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'FixedFlux':
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    if with_max_dif_cost:
                        max_dif_cost, max_dif_weight = 0.0, 0.0
                        
                    for i, (marker, outlet_item) in enumerate(obj_cost_args['outlet_cost_forms'].items()):
                        outlet_flux = assemble_scalar(
                            form=outlet_item['flux_form'],
                            domain=self.domain, tape=self.tape
                        )
                        target_flux = outlet_item['target_flux']
                        print(f"{marker} flux: {outlet_flux}, target: {target_flux}, inlet_flux: {obj_cost_args['inlet_flux']}")
                                                
                        flux_dif = outlet_flux - target_flux
                        if with_rescale:
                            flux_dif = flux_dif / target_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                            max_dif_weight = outlet_item['weight']
                        
                        weight = outlet_item['weight']
                        if i == 0:
                            obj_cost = weight * flux_dif_cost
                        else:
                            obj_cost += weight * flux_dif_cost
                            
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)

                elif obj_cost_args['style'] == 'BalanceFlux':
                    weight = obj_cost_args['weight']
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    outlet_flux_list = []
                    for marker, outlet_flux_form in obj_cost_args['outlet_forms'].items():
                        outlet_flux = assemble_scalar(form=outlet_flux_form, domain=self.domain, tape=self.tape)
                        outlet_flux_list.append(outlet_flux)
                    
                    mean_flux = 0.0
                    for outlet_flux in outlet_flux_list:
                        mean_flux += outlet_flux
                    mean_flux = mean_flux / obj_cost_args['num_outlets']
                    
                    if with_max_dif_cost:
                        max_dif_cost = 0.0
                        
                    for i, outlet_flux in enumerate(outlet_flux_list):
                        flux_dif = outlet_flux - mean_flux
                        if with_rescale:
                            flux_dif = flux_dif / mean_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                        
                        if i == 0:
                            obj_cost = weight * flux_dif_cost
                        else:
                            obj_cost += weight * flux_dif_cost
                        
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', weight)
                else:
                    raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
                                
                if obj_idx == 0:
                    all_obj_cost = obj_cost
                else:
                    all_obj_cost += obj_cost  # type: ignore
                # print(f'{obj_cost_args["style"]}: {obj_cost}')
                
            self.obj_cost_args_list = obj_cost_args_list
            
            # ------ define regularization cost ------
            reg_cost_args = self.prepare_reg_args(reg_cost_args)  # type: ignore
            if reg_cost_args['style'] == 'Volume':
                target_volume = reg_cost_args['target_volume']
                topo_volume = assemble_scalar(reg_cost_args['vol_form'], domain=self.domain, tape=self.tape)
                volume_dif = target_volume - topo_volume
                if reg_cost_args['reg_rescale']:
                    volume_dif = volume_dif / target_volume
                reg_cost = reg_cost_args['weight'] * (volume_dif ** 2)
            else:
                raise ValueError(f'Invalid regularization style: {reg_cost_args['style']}')
            self.reg_cost_args = reg_cost_args
            
            loss_all = all_obj_cost + reg_cost
            
            # ------ define obstacle cost ------
            if self.obstacle_cost_args is not None:
                if self.obstacle_cost_args['is_cell_field']:
                    obs_cost_form = inner(self.obstacle_cost_args['obs_cell_mask'], self.topo_weight) * ufl.dx
                else:
                    obs_cost_form = inner(self.obstacle_cost_args['obs_point_mask'], self.topo_weight) * ufl.dx
                self.obstacle_cost_args['cost_form'] = obs_cost_form
                
                obs_cost = assemble_scalar(obs_cost_form, domain=self.domain, tape=self.tape)
                obs_cost = obs_cost * self.obstacle_cost_args['weight']
                
                loss_all += obs_cost
            
            # ------ define optimization problem ------
            self.control_phi = Control(self.topo_weight)  # type: ignore
            self.opt_problem = ReducedFunctional(loss_all, [self.control_phi])

        loss_dict, log_dict = self.eval()
        print('[Log]: Initial loss:', loss_dict)
        
        # for block in self.tape.get_blocks():
        #     print(block, ' type: ', type(block))

    def record_Upw(self, step: int, **kwargs):
        super().record_Upw(step, **kwargs)
        # self.copy_fipy_result_log(step)  # for debug


def debug_1():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t38'
    msh_file = os.path.join(base_dir, 't38_model.msh')

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    optim = TopoFluidOptimizerFipy(
        name='t38',
        msh_file=msh_file,
        mesh=mesh,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        topo_weight_init=0.1,
        init_topo_weight_method='filter',
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'fipy_topo'),
        alpha_fun_max=100.0,
        alpha_fun_min=1e-12,
        alpha_fun_p=3.0,
    )    
    optim.setup_tape(Tape())
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=22,
        face_center=np.array([-0.05, 0.2]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=23,
        face_center=np.array([1.05, 0.8]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=24
    )
    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    
    # 似乎topo weight的约束cost应该占据主要地位
    optim.define_problem(
        obj_cost_args_list=[{
            'style': 'MinDissipation+BrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 1.0,
            'penalized_weight': 1.0,
        }],
        reg_cost_args={
            'style': 'Volume',
            'weight': 1.0,
            'target_ratio': 0.1,
            'reg_rescale': True,
        },
        simulation_iterations=200,
        convergence_threshold=1e-4,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=1,
        use_tqdm=True,
        use_log=True,
    )
    
    # recorder = TensorBoardRecorder(os.path.join(base_dir, 'fipy_topo', 'log'))
    # result = optim.run(
    #     max_iteration=300,
    #     line_search_steps=[0.05],
    #     recorder=recorder,
    #     full_run=False,
    # )


def debug_2():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/compare_2d'
    msh_file = os.path.join(base_dir, 'model.msh')

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    optim = TopoFluidOptimizerFipy(
        name='compare_2d',
        msh_file=msh_file,
        mesh=mesh,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        init_topo_weight_method='filter',
        topo_weight_init=1.0,
        weight_filter_radius=0.075,
        result_dir=os.path.join(base_dir, 'fipy_topo'),
        alpha_fun_max=100.0,
        alpha_fun_min=1e-12,
        alpha_fun_p=3.0,
    )    
    optim.setup_tape(Tape())
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=1,
        face_center=np.array([-0.05, 0.5]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=2,
        face_center=np.array([1.05, 0.5]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=3
    )
    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    
    # 似乎topo weight的约束cost应该占据主要地位
    optim.define_problem(
        obj_cost_args_list=[{
            'style': 'MinDissipation+BrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 1.0,
            'penalized_weight': 1.0,
        }],
        reg_cost_args={
            'style': 'Volume',
            'weight': 1.0,
            'target_ratio': 0.1,
            'reg_rescale': True,
        },
        simulation_iterations=200,
        convergence_threshold=1e-5,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=1,
        use_tqdm=True,
        use_log=True,
    )
    
    recorder = TensorBoardRecorder(os.path.join(base_dir, 'fipy_topo', 'log'))
    result = optim.run(
        max_iteration=300,
        line_search_steps=[0.05],
        recorder=recorder,
        full_run=True,
    )


def debug_3():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/compare'
    msh_file = os.path.join(base_dir, 'model.msh')

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    optim = TopoFluidOptimizerFipy(
        name='compare',
        msh_file=msh_file,
        mesh=mesh,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        init_topo_weight_method='filter',
        topo_weight_init=0.1,
        weight_filter_radius=0.075,
        result_dir=os.path.join(base_dir, 'fipy_topo'),
        alpha_fun_max=100.0,
        alpha_fun_min=1e-12,
        alpha_fun_p=3.0,
    )    
    optim.setup_tape(Tape())
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=1,
        face_center=np.array([-0.05, 0.5, 0.5]),
        face_norm_vector=np.array([1.0, 0.0, 0.0]),
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=2,
        face_center=np.array([1.05, 0.5, 0.5]),
        face_norm_vector=np.array([1.0, 0.0, 0.0]),
        face_radius=0.1
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=3
    )
    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    
    optim.define_problem(
        obj_cost_args_list=[{
            'style': 'MinDissipation+BrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 1.0,
            'penalized_weight': 1.0,
        }],
        reg_cost_args={
            'style': 'Volume',
            'weight': 0.1,
            'target_ratio': 0.02,
            'reg_rescale': True,
        },
        simulation_iterations=200,
        convergence_threshold=1e-4,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=8,
        use_tqdm=True,
        use_log=True,
    )
    
    recorder = TensorBoardRecorder(os.path.join(base_dir, 'fipy_topo', 'log'))
    result = optim.run(
        max_iteration=500,
        line_search_steps=[0.05],
        recorder=recorder,
        full_run=True,
    )


if __name__ == '__main__':
    # debug_1()
    # debug_2()
    debug_3()