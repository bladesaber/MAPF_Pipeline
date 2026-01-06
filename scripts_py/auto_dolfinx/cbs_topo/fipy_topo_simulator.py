import os, shutil
import argparse

import dolfinx
import ufl

from meshio import xdmf
import numpy as np
import pyvista
from mpi4py import MPI
import json

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable
from fipy import DiffusionTerm
from fipy import ImplicitSourceTerm

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.simulator.fipy_simulator import FipyStokeSimulator
from scripts_py.openmpi_utils import OpenMpiUtils


class FipyStokeTopoSimulator(FipyStokeSimulator):
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        parallel_num: int,
        structure_mesh: MeshObj|None,
        structure_alpha: np.ndarray|None=None,
        structure_init_x_velocity: np.ndarray|None = None,
        structure_init_y_velocity: np.ndarray|None = None,
        structure_init_z_velocity: np.ndarray|None = None,
        structure_init_pressure: np.ndarray|None = None,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            parallel_num=parallel_num,
            structure_init_x_velocity=structure_init_x_velocity,
            structure_init_y_velocity=structure_init_y_velocity,
            structure_init_z_velocity=structure_init_z_velocity,
            structure_init_pressure=structure_init_pressure,
        )    
        if self.using_parallel:
            if OpenMpiUtils.is_master_rank():
                if structure_alpha is not None:
                    data_dict = {'with_alpha': True, 'alpha': structure_alpha[self.structure_mesh_to_mesh_idxs]}
                    OpenMpiUtils.send_to_all(data_dict, tag=100)
                else:
                    data_dict = {'with_alpha': False}
                    OpenMpiUtils.send_to_all(data_dict, tag=100)
            else:
                data_dict = OpenMpiUtils.recv(source=0, tag=100)
        else:
            if structure_alpha is not None:
                data_dict = {'with_alpha': True, 'alpha': structure_alpha[self.structure_mesh_to_mesh_idxs]}
            else:
                data_dict = {'with_alpha': False}
        
        if data_dict['with_alpha']:
            # Fipy will handle the global value automatically            
            self.alpha_cell = CellVariable(mesh=self.mesh, name='alpha', value=data_dict['alpha'])
            
            self.x_source = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore
            self.y_source = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore
            if self.tdim == 3:
                self.z_source = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore

            if self.tdim == 2:
                self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0.]) - self.x_source
                self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1.]) - self.y_source
            elif self.tdim == 3:
                self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.]) - self.x_source
                self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.]) - self.y_source
                self.zVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 0., 1.]) - self.z_source
        else:
            self.alpha_cell = None
        
        if self.using_parallel:
            OpenMpiUtils.synchronize_all_ranks()

    def get_alpha(self, map_to_structure: bool) -> np.ndarray|None:
        if self.alpha_cell is None:
            return None
        
        if self.using_parallel:
            alpha = self.alpha_cell.globalValue
        else:
            alpha = self.alpha_cell.value.copy()
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            if map_to_structure:
                alpha = alpha[self.mesh_to_structure_mesh_idxs]
            return alpha
        return None

    def record(self, save_file: str):
        velocity = self.get_velocity(map_to_structure=True)
        pressure = self.get_pressure(map_to_structure=True)
        alpha = self.get_alpha(map_to_structure=True)
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            assert self.structure_mesh is not None, "structure_mesh is not initialized"
            assert velocity is not None, "velocity is not initialized"
            assert pressure is not None, "pressure is not initialized"
            
            self.structure_mesh.grid.point_data.clear()
            self.structure_mesh.grid.cell_data.clear()
            
            self.structure_mesh.grid.cell_data['pressure'] = pressure
            if velocity.shape[1] == 2:
                velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=1)
            self.structure_mesh.grid.cell_data['velocity'] = velocity
            self.structure_mesh.grid.cell_data['velocity_mag'] = np.linalg.norm(velocity, axis=1)
            if alpha is not None:
                self.structure_mesh.grid.cell_data['alpha'] = alpha
            
            # ------ debug record ------
            cell_vector_space = DolfinUtils.cell_vector_function_space(self.structure_mesh.domain)
            cell_vector_fun = dolfinx.fem.Function(cell_vector_space)        
            cell_vector_fun.x.array[:] = velocity[:, :self.tdim].reshape(-1)  # type: ignore
            
            cell_scalar_space = DolfinUtils.cell_scalar_function_space(self.structure_mesh.domain)
            cell_scalar_fun = dolfinx.fem.Function(cell_scalar_space)
            cell_scalar_fun.x.array[:] = pressure.reshape(-1)  # type: ignore
            
            point_vector_space = DolfinUtils.point_vector_function_space(self.structure_mesh.domain)
            point_vector_fun = dolfinx.fem.Function(point_vector_space)
            point_vector_fun.interpolate(cell_vector_fun)  # type: ignore
            
            point_scalar_space = DolfinUtils.point_scalar_function_space(self.structure_mesh.domain)
            point_scalar_fun = dolfinx.fem.Function(point_scalar_space)
            point_scalar_fun.interpolate(cell_scalar_fun)  # type: ignore
            
            point_velocity = point_vector_fun.x.array.copy().reshape((-1, self.tdim))  # type: ignore
            if point_velocity.shape[1] == 2:
                point_velocity = np.concatenate([point_velocity, np.zeros((point_velocity.shape[0], 1))], axis=1)
            
            self.structure_mesh.grid.point_data['velocity'] = point_velocity
            self.structure_mesh.grid.point_data['pressure'] = point_scalar_fun.x.array.copy()  # type: ignore
            self.structure_mesh.grid.point_data['velocity_mag'] = np.linalg.norm(point_velocity, axis=1)
            
            # ------ debug record ------
            self.structure_mesh.grid.save(save_file)

    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_alpha: np.ndarray|None = None,
        structure_init_x_velocity: np.ndarray|None = None,
        structure_init_y_velocity: np.ndarray|None = None,
        structure_init_z_velocity: np.ndarray|None = None,
        structure_init_pressure: np.ndarray|None = None,
        force_msh_init: bool = False
    ):
        num_partitions = OpenMpiUtils.get_mpi_size()
        base_dir = os.path.dirname(source_msh_file)
        base_name = os.path.basename(source_msh_file).replace('.msh', '')
        if num_partitions == 1:
            fipy_msh_file = os.path.join(base_dir, f"{base_name}_single.msh")
        else:
            fipy_msh_file = os.path.join(base_dir, f"{base_name}_parallel_{num_partitions}.msh")
        
        if (num_partitions == 1) or (num_partitions > 1 and OpenMpiUtils.is_master_rank()):
            if (not os.path.exists(fipy_msh_file)) or force_msh_init:
                FipyStokeSimulator.msh_version_convert(
                    source_msh_file=source_msh_file, 
                    target_msh_file=fipy_msh_file, 
                    num_partitions=num_partitions, 
                    target_version='msh2'
                )
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim, comm=MPI.COMM_SELF, verbose=False)
            structure_mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        else:
            structure_mesh = None
        
        # # ------ visual structure alpha ------
        # if OpenMpiUtils.is_master_rank():
        #     assert structure_mesh is not None, "structure_mesh is not initialized"
        #     assert structure_alpha is not None, "structure_alpha is not initialized"
        #     cell_centers = structure_mesh.grid.cell_centers().points
        #     plotter = pyvista.Plotter()
        #     plotter.add_mesh(structure_mesh.grid, color='black', style='wireframe')
        #     plotter.add_mesh(pyvista.PointSet(cell_centers[structure_alpha > 1.0, :]), color='red', point_size=2.0)
        #     plotter.add_mesh(pyvista.PointSet(cell_centers[structure_alpha < 1.0, :]), color='blue', point_size=2.0)
        #     plotter.show()
        # # ------ visual structure alpha ------
        
        if dim == 3:
            mesh = Gmsh3D(fipy_msh_file)
        elif dim == 2:
            mesh = Gmsh2D(fipy_msh_file)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
                
        return FipyStokeTopoSimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            parallel_num=num_partitions,
            structure_alpha=structure_alpha,
            structure_init_x_velocity=structure_init_x_velocity,
            structure_init_y_velocity=structure_init_y_velocity,
            structure_init_z_velocity=structure_init_z_velocity,
            structure_init_pressure=structure_init_pressure,
        )
    
    @staticmethod
    def create_simulation_config(
        cfg_file: str, 
        source_msh_file: str,
        dim: int,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        boundary_list: list[FluidBoundary],
        iterations: int,
        convergence_threshold: float,
        hot_steps: int,
        log_step: int,
        save_dir: str,
        use_tqdm: bool,
        use_log: bool,
        structure_init_vtu_file: str|None = None,
        vtu_velocity_field: str = 'velocity',
        vtu_pressure_field: str = 'pressure',
        vtu_alpha_field: str = 'alpha',
    ):
        assert cfg_file.endswith('.json'), "cfg_file must be a json file"
        
        cfg_dict = {
            'source_msh_file': source_msh_file,
            'dim': dim,
            'viscosity': viscosity,
            'velocityRelaxation': velocityRelaxation,
            'pressureRelaxation': pressureRelaxation,
            'iterations': iterations,
            'convergence_threshold': convergence_threshold,
            'hot_steps': hot_steps,
            'log_step': log_step,
            'save_dir': save_dir,
            'use_tqdm': use_tqdm,
            'use_log': use_log,
            'boundary_list': [],
            'structure_init_vtu_file': structure_init_vtu_file,
            'vtu_velocity_field': vtu_velocity_field,
            'vtu_pressure_field': vtu_pressure_field,
            'vtu_alpha_field': vtu_alpha_field,
        }
        
        for bry in boundary_list:
            if isinstance(bry.value, np.ndarray):
                bry_value = bry.value.tolist()
            else:
                bry_value = bry.value
            bry_dict = {
                'name': bry.name,
                'boundary_type': bry.boundary_type,
                'value': bry_value,
                'marker': bry.marker,
            }
            if bry.boundary_type == 'velocity':
                if isinstance(bry.face_center, np.ndarray):
                    bry_face_center = bry.face_center.tolist()
                else:
                    bry_face_center = bry.face_center
                if isinstance(bry.face_norm_vector, np.ndarray):
                    bry_face_norm_vector = bry.face_norm_vector.tolist()
                else:
                    bry_face_norm_vector = bry.face_norm_vector
                
                bry_dict['face_center'] = bry_face_center
                bry_dict['face_norm_vector'] = bry_face_norm_vector
                bry_dict['face_radius'] = bry.face_radius
            cfg_dict['boundary_list'].append(bry_dict)
        
        with open(cfg_file, 'w') as f:
            json.dump(cfg_dict, f, indent=4)

    @staticmethod
    def run_from_config(cfg_file: str, force_msh_init: bool = False):
        assert cfg_file.endswith('.json'), "cfg_file must be a json file"
        with open(cfg_file, 'r') as f:
            cfg_dict = json.load(f)
        
        init_x_velocity = None
        init_y_velocity = None
        init_z_velocity = None
        init_pressure = None
        init_alpha = None
        if cfg_dict['structure_init_vtu_file'] is not None:
            init_vtu = pyvista.read(cfg_dict['structure_init_vtu_file'])
            
            if cfg_dict['vtu_velocity_field'] in init_vtu.cell_data.keys():
                init_velocity = init_vtu.cell_data[cfg_dict['vtu_velocity_field']]
                init_x_velocity = init_velocity[:, 0]
                init_y_velocity = init_velocity[:, 1]
                if cfg_dict['dim'] == 3:
                    init_z_velocity = init_velocity[:, 2]
            
            if cfg_dict['vtu_pressure_field'] in init_vtu.cell_data.keys():
                init_pressure = init_vtu.cell_data[cfg_dict['vtu_pressure_field']]
            
            if cfg_dict['vtu_alpha_field'] in init_vtu.cell_data.keys():
                init_alpha = init_vtu.cell_data[cfg_dict['vtu_alpha_field']]
        
        simulator = FipyStokeTopoSimulator.from_msh(
            source_msh_file=cfg_dict['source_msh_file'],
            dim=cfg_dict['dim'],
            viscosity=cfg_dict['viscosity'],
            velocityRelaxation=cfg_dict['velocityRelaxation'],
            pressureRelaxation=cfg_dict['pressureRelaxation'],
            structure_alpha=init_alpha,
            structure_init_x_velocity=init_x_velocity,
            structure_init_y_velocity=init_y_velocity,
            structure_init_z_velocity=init_z_velocity,
            structure_init_pressure=init_pressure,
            force_msh_init=force_msh_init,
        )
        
        for bry_dict in cfg_dict['boundary_list']:
            if bry_dict['boundary_type'] == 'velocity':
                bry = FluidBoundary(
                    name=bry_dict['name'],
                    boundary_type='velocity',
                    value=bry_dict['value'],
                    marker=bry_dict['marker'],
                    face_center=bry_dict['face_center'],
                    face_norm_vector=bry_dict['face_norm_vector'],
                    face_radius=bry_dict['face_radius'],
                )
            elif bry_dict['boundary_type'] == 'pressure':
                bry = FluidBoundary(
                    name=bry_dict['name'],
                    boundary_type='pressure',
                    value=bry_dict['value'],
                    marker=bry_dict['marker'],
                )
            elif bry_dict['boundary_type'] == 'wall':
                bry = FluidBoundary(
                    name=bry_dict['name'],
                    boundary_type='wall',
                    value=bry_dict['value'],
                    marker=bry_dict['marker'],
                )
            else:
                raise ValueError(f"Invalid boundary type: {bry_dict['boundary_type']}")
            simulator.add_boundary(bry)
        
        simulator.define_boundary()
        
        simulator.run(
            iterations=cfg_dict['iterations'],
            convergence_threshold=cfg_dict['convergence_threshold'],
            hot_steps=cfg_dict['hot_steps'],
            log_step=cfg_dict['log_step'],
            save_dir=cfg_dict['save_dir'],
            use_tqdm=cfg_dict['use_tqdm'],
            use_log=cfg_dict['use_log'],
        )
    
    @staticmethod
    def alpha_fun(
        rho: np.ndarray|dolfinx.fem.Function, alpha_min=0.01, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow
    ):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore


def debug_2D():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t30'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    topo_center = np.array([0.5, 0.5])
    topo_radius = 0.1
    
    cell_centers = mesh.grid.cell_centers().points
    topo_weight = np.ones(cell_centers.shape[0]) - 1e-12
    topo_bool = np.linalg.norm(cell_centers[:, :mesh.tdim] - topo_center.reshape((1, -1))[:, :2], axis=1) < topo_radius
    topo_weight[topo_bool] = 1e-12
    structure_alpha = FipyStokeTopoSimulator.alpha_fun(
        rho=topo_weight,
        alpha_min=0.01,
        alpha_max=200.0,
        p=3.0,
        pow_operator=np.power
    )
    
    mesh.grid.cell_data.clear()
    mesh.grid.cell_data['alpha'] = structure_alpha
    init_vtu_file = os.path.join(base_dir, 'init.vtu')
    mesh.grid.save(init_vtu_file)

    FipyStokeTopoSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=msh_file,
        dim=mesh.tdim,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        boundary_list=[
            FluidBoundary(
                name='inlet',
                boundary_type='velocity',
                value=[1.0, 0.0],
                marker=22,
                face_center=[-0.05, 0.2],
                face_norm_vector=[1.0, 0.0],
                face_radius=0.1
            ),
            FluidBoundary(
                name='outlet',
                boundary_type='pressure',
                value=0.0,
                marker=23
            ),
            FluidBoundary(
                name='wall',
                boundary_type='wall',
                value=0.0,
                marker=24
            )
        ],
        iterations=60,
        convergence_threshold=1e-5,
        hot_steps=10,
        log_step=2,
        save_dir=os.path.join(base_dir, 'cache'),
        use_tqdm=False,
        use_log=True,
        structure_init_vtu_file=init_vtu_file,
        vtu_velocity_field='velocity',
        vtu_pressure_field='pressure',
        vtu_alpha_field='alpha',
    )
    

def debug_3D():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t28'
    msh_file = os.path.join(base_dir, 't28_model.msh')
    
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    topo_center = np.array([0.5, 0.5, 0.25])
    topo_radius = 0.1
    
    cell_centers = mesh.grid.cell_centers().points
    topo_weight = np.ones(cell_centers.shape[0]) - 1e-12
    topo_bool = np.linalg.norm(cell_centers[:, :2] - topo_center.reshape((1, -1))[:, :2], axis=1) < topo_radius
    topo_weight[topo_bool] = 1e-12
    structure_alpha = FipyStokeTopoSimulator.alpha_fun(
        rho=topo_weight,
        alpha_min=0.01,
        alpha_max=200.0,
        p=3.0,
        pow_operator=np.power
    )
    
    mesh.grid.cell_data.clear()
    mesh.grid.cell_data['alpha'] = structure_alpha
    init_vtu_file = os.path.join(base_dir, 'init.vtu')
    mesh.grid.save(init_vtu_file)
    
    FipyStokeTopoSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=msh_file,
        dim=mesh.tdim,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        boundary_list=[
            FluidBoundary(
                name='inlet',
                boundary_type='velocity',
                value=[1.0, 0.0, 0.0],
                marker=20,
                face_center=[-0.05, 0.2, 0.25],
                face_norm_vector=[1.0, 0.0, 0.0],
                face_radius=0.1
            ),
            FluidBoundary(
                name='outlet',
                boundary_type='pressure',
                value=0.0,
                marker=21
            ),
            FluidBoundary(
                name='wall',
                boundary_type='wall',
                value=0.0,
                marker=22
            )
        ],
        iterations=60,
        convergence_threshold=1e-5,
        hot_steps=10,
        log_step=2,
        save_dir=os.path.join(base_dir, 'cache'),
        use_tqdm=False,
        use_log=True,
        structure_init_vtu_file=init_vtu_file,
        vtu_velocity_field='velocity',
        vtu_pressure_field='pressure',
        vtu_alpha_field='alpha',
    )


def run_main():
    """
    config_path = '*/config.json'
    cmd = f'mpirun -n 4 python scripts_py/auto_dolfinx/cbs_topo/fipy_topo_simulator.py --cfg_file {config_path} --force_msh_init True'
    subprocess.run(
        cmd, shell=True,
        cwd='/home/admin123456/Desktop/work/MAPF_Pipeline'
    )
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--force_msh_init', type=bool, default=False)
    args = parser.parse_args()
    FipyStokeTopoSimulator.run_from_config(args.cfg_file, args.force_msh_init)


if __name__ == '__main__':
    # debug_2D()
    # debug_3D()
    run_main()
    