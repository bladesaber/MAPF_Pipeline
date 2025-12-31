import os, shutil
from typing import Literal
import subprocess
import argparse
from tqdm import tqdm

import dolfinx

import json
import numpy as np
from sklearn.neighbors import KDTree
from mpi4py import MPI

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.openmpi_utils import OpenMpiUtils
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.simulator.eary_stoper import EarlyStopper
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator


class FipyStokeParallelSimulator(FipyStokeSimulator):
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: MeshObj|None,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh, # type: ignore
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
        )
        
        # ------ wait for all processes to finish initialization ------
        OpenMpiUtils.synchronize_all_ranks()
    
    def run(
        self, 
        iterations: int, 
        convergence_threshold: float = 1e-5,
        hot_steps: int = 10,
        log_step: int = 10, 
        save_dir: str|None = None, 
        use_tqdm: bool = False, 
        use_log: bool = True
    ):
        if OpenMpiUtils.is_master_rank():
            if save_dir is not None:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            
            if use_tqdm and use_log:
                tqdm_bar = tqdm(total=iterations, desc='Fipy Parallel Simulator')
            
            stoper = EarlyStopper(hot_step=10, window_size=30, shift_step=5, atol=1e-6, rtol=1e-3, atol_inv=1e-3)
            stoper.define_var('x_residual', convergence_threshold)
            stoper.define_var('y_residual', convergence_threshold)
            if self.tdim == 3:
                stoper.define_var('z_residual', convergence_threshold)
            stoper.define_var('p_residual', convergence_threshold)
            stoper.define_var('continuity_residual', convergence_threshold)
        
        # ------ wait for all processes to finish initialization ------
        OpenMpiUtils.synchronize_all_ranks()
        
        # ------ start the simulation ------
        for iter_idx in range(iterations):
            # ------ solve the momentum equation ------
            self.xVelocityEq.cacheMatrix()
            x_residual = self.xVelocityEq.sweep(var=self.xVelocity, underRelaxation=self.velocityRelaxation)
            
            self.yVelocityEq.cacheMatrix()
            y_residual = self.yVelocityEq.sweep(var=self.yVelocity, underRelaxation=self.velocityRelaxation)
            
            if self.tdim == 3:
                self.zVelocityEq.cacheMatrix()
                z_residual = self.zVelocityEq.sweep(var=self.zVelocity, underRelaxation=self.velocityRelaxation)
            # ------ solve the momentum equation ------

            # ------ update the ap coefficient from the matrix diagonal ------
            xmat = self.xVelocityEq.matrix
            ap_x = -numerix.asarray(xmat.takeDiagonal())

            ymat = self.yVelocityEq.matrix
            ap_y = -numerix.asarray(ymat.takeDiagonal())

            if self.tdim == 3:
                zmat = self.zVelocityEq.matrix
                ap_z = -numerix.asarray(zmat.takeDiagonal())

            if self.tdim == 2:
                self.ap[:] = (ap_x + ap_y) * 0.5
            else:
                self.ap[:] = (ap_x + ap_y + ap_z) / 3.0
            # ------ update the ap coefficient from the matrix diagonal ------
            
            # interpolate the pressure gradient to the face based on linear pressure gradient difference
            presgrad = self.pressure.grad  # cell pressure gradient
            facepresgrad = _FaceGradVariable(self.pressure)  # face pressure gradient

            self.face_velocity[0] = (
                self.xVelocity.arithmeticFaceValue 
                + self.control_volume / self.ap.arithmeticFaceValue * (presgrad[0].arithmeticFaceValue - facepresgrad[0])
                )
            self.face_velocity[1] = (
                self.yVelocity.arithmeticFaceValue 
                + self.control_volume / self.ap.arithmeticFaceValue * (presgrad[1].arithmeticFaceValue - facepresgrad[1])
            )
            if self.tdim == 3:
                self.face_velocity[2] = (
                    self.zVelocity.arithmeticFaceValue 
                    + self.control_volume / self.ap.arithmeticFaceValue * (presgrad[2].arithmeticFaceValue - facepresgrad[2])
                )
            self.apply_boundary_to_correction_velocity()

            # ------ solve the pressure correction equation ------
            self.pressureCorrectionEq.cacheRHSvector()
            ## left bottom point must remain at pressure 0, so no correction
            p_residual = self.pressureCorrectionEq.sweep(var=self.pressureCorrection, solver=self.pressureCorrectionSolver)
            continuity_residual = self.pressureCorrectionEq.RHSvector

            # ------ update the pressure and velocity ------
            self.pressure.setValue(self.pressure + self.pressureRelaxation * self.pressureCorrection)
            self.xVelocity.setValue(self.xVelocity - self.pressureCorrection.grad[0] / self.ap * self.mesh.cellVolumes)
            self.yVelocity.setValue(self.yVelocity - self.pressureCorrection.grad[1] / self.ap * self.mesh.cellVolumes)
            if self.tdim == 3:
                self.zVelocity.setValue(self.zVelocity - self.pressureCorrection.grad[2] / self.ap * self.mesh.cellVolumes)
            
            # ------ wait for all processes to finish calculating ------
            OpenMpiUtils.synchronize_all_ranks()
            
            # ------ reduce the residual to the master process ------
            global_x_residual: float = OpenMpiUtils.reduce(x_residual, mode='max', root=0)  # type: ignore
            global_y_residual: float = OpenMpiUtils.reduce(y_residual, mode='max', root=0)  # type: ignore
            if self.tdim == 3:
                global_z_residual: float = OpenMpiUtils.reduce(z_residual, mode='max', root=0)  # type: ignore
            global_p_residual: float = OpenMpiUtils.reduce(p_residual, mode='max', root=0)  # type: ignore
            global_continuity_residual: float = OpenMpiUtils.reduce(np.max(np.abs(continuity_residual)), mode='max', root=0)  # type: ignore
            
            if log_step > 0 and use_log:
                if iter_idx % log_step == 0:
                    if OpenMpiUtils.is_master_rank():
                        if use_tqdm:
                            tqdm_bar.update(log_step)
                        else:
                            log = f'[Log: {iter_idx}]: x residual:{global_x_residual:.8f}, y residual:{global_y_residual:.8f}, '
                            if self.tdim == 3:
                                log += f'z residual:{global_z_residual:.8f},'
                            log += f' p residual:{global_p_residual:.8f}, continuity_residual:{global_continuity_residual:.8f}'
                            OpenMpiUtils.print(log)
                    
                    if save_dir is not None:
                        self.record(save_file=os.path.join(save_dir, f'iter_{iter_idx}.vtu'))
            
            if iter_idx > hot_steps and convergence_threshold > 0.0:
                if OpenMpiUtils.is_master_rank():
                    
                    var_dict = {
                        'x_residual': global_x_residual,
                        'y_residual': global_y_residual,
                        'p_residual': global_p_residual,
                        'continuity_residual': global_continuity_residual,
                    }
                    if self.tdim == 3:
                        var_dict['z_residual'] = global_z_residual
                    stoper.add_var(var_dict)
                    convergence_flag = stoper.is_converged()                                        
                    OpenMpiUtils.send_to_all(data={'is_solve': convergence_flag}, tag=100)
                    if convergence_flag:
                        break
                
                else:
                    msg = OpenMpiUtils.recv(source=0, tag=100)
                    if msg['is_solve']:
                        break
        
        if save_dir is not None:
            self.record(save_file=os.path.join(save_dir, 'result.vtu'))
            
            velocity = self.get_velocity(map_to_structure=False)
            pressure = self.get_pressure(map_to_structure=False)
            if OpenMpiUtils.is_master_rank():
                np.save(os.path.join(save_dir, 'fipy_velocity.npy'), velocity)  # type: ignore
                np.save(os.path.join(save_dir, 'fipy_pressure.npy'), pressure)  # type: ignore
            
        if OpenMpiUtils.is_master_rank() and save_dir is not None:
            residual_dict = {
                'x_residual': global_x_residual,
                'y_residual': global_y_residual,
                'p_residual': global_p_residual,
                'continuity_residual': global_continuity_residual,
            }
            if self.tdim == 3:
                residual_dict['z_residual'] = global_z_residual
            with open(os.path.join(save_dir, 'residual.json'), 'w') as f:
                json.dump(residual_dict, f, indent=4)
        
    def get_mesh_to_structure_mesh_mapping(self):
        global_fipy_cell_centers = self.mesh.cellCenters.globalValue
        if OpenMpiUtils.is_master_rank():
            assert self.structure_mesh is not None, "structure_mesh is not initialized"
            cell_centers = self.structure_mesh.grid.cell_centers().points[: , :self.tdim]
            
            global_fipy_cell_centers = np.copy(global_fipy_cell_centers).T
            tree = KDTree(global_fipy_cell_centers)

            dists_list, idxs_list = tree.query(cell_centers, k=1, return_distance=True)
            dists_list = np.concatenate(dists_list, axis=-1)
            assert np.max(dists_list) < 1e-8, "mesh point is center of fipy cell"
            
            mesh_to_structure_mesh_idxs = np.concatenate(idxs_list, axis=-1)
            structure_mesh_to_mesh_idxs = np.argsort(mesh_to_structure_mesh_idxs)
            
            return mesh_to_structure_mesh_idxs, structure_mesh_to_mesh_idxs
        return None, None

    def record(self, save_file: str):
        velocity = self.get_velocity(map_to_structure=True)
        pressure = self.get_pressure(map_to_structure=True)
        
        if OpenMpiUtils.is_master_rank():
            assert self.structure_mesh is not None, "structure_mesh is not initialized"
            assert velocity is not None, "velocity is not initialized"
            assert pressure is not None, "pressure is not initialized"
            
            self.structure_mesh.grid.point_data.clear()
            self.structure_mesh.grid.cell_data.clear()
            
            self.structure_mesh.grid.cell_data['pressure_sim'] = pressure
            if velocity.shape[1] == 2:
                velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=1)
            self.structure_mesh.grid.cell_data['velocity_sim'] = velocity
            self.structure_mesh.grid.cell_data['velocity_mag_sim'] = np.linalg.norm(velocity, axis=1)
            
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
            
            self.structure_mesh.grid.point_data['velocity_sim'] = point_velocity
            self.structure_mesh.grid.point_data['pressure_sim'] = point_scalar_fun.x.array.copy()  # type: ignore
            self.structure_mesh.grid.point_data['velocity_mag_sim'] = np.linalg.norm(point_velocity, axis=1)
            
            # ------ debug record ------
            self.structure_mesh.grid.save(save_file)
    
    @staticmethod
    def msh_version_convert(
        source_msh_file: str, 
        target_msh_file: str, 
        num_partitions: int,
        target_version: Literal['msh2', 'msh4']
    ):
        """
        -part_topo: Create the partition topology, optional
        -part_ghosts: Create ghost cells, critical for parallel simulation, must be enabled
        -part_[no_]physicals: Create physical groups for partitions, must be disabled
        """
        base_dir = os.path.dirname(target_msh_file)
        log_file = os.path.join(base_dir, 'msh_version_convert.log')
        run_bash = f"gmsh {source_msh_file} -format {target_version} -part {num_partitions} -part_ghosts -o {target_msh_file} -save > {log_file}"
        subprocess.run(run_bash, shell=True, check=True)
        
    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_mesh: MeshObj|None = None,
    ):
        # ------ initialize the partition msh file ------
        num_partitions = OpenMpiUtils.get_mpi_size()
        base_dir = os.path.dirname(source_msh_file)
        base_name = os.path.basename(source_msh_file).replace('.msh', '')
        partition_msh_file = os.path.join(base_dir, f"{base_name}_partition_{num_partitions}.msh")
        
        if OpenMpiUtils.is_master_rank():
            FipyStokeParallelSimulator.msh_version_convert(source_msh_file, partition_msh_file, num_partitions, 'msh2')
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim, comm=MPI.COMM_SELF)
            structure_mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        else:
            structure_mesh = None
            
        OpenMpiUtils.synchronize_all_ranks()
        
        if dim == 3:
            mesh = Gmsh3D(partition_msh_file)
        elif dim == 2:
            mesh = Gmsh2D(partition_msh_file)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
                
        return FipyStokeParallelSimulator(
            mesh=mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_mesh=structure_mesh,
        )

    def eval_point(self, points: np.ndarray):
        raise NotImplementedError
    
    def get_velocity(self, map_to_structure: bool) -> np.ndarray|None:
        velocity_list = [
            self.xVelocity.globalValue.reshape((-1, 1)),
            self.yVelocity.globalValue.reshape((-1, 1))
        ]
        if self.tdim == 3:
            velocity_list.append(self.zVelocity.globalValue.reshape((-1, 1)))
        
        if OpenMpiUtils.is_master_rank():
            velocity = np.concatenate(velocity_list, axis=1)
            if map_to_structure:
                velocity = velocity[self.mesh_to_structure_mesh_idxs, :]
            return velocity
        return None
    
    def get_pressure(self, map_to_structure: bool) -> np.ndarray|None:
        pressure = self.pressure.globalValue
        if OpenMpiUtils.is_master_rank():
            if map_to_structure:
                pressure = pressure[self.mesh_to_structure_mesh_idxs]
            return pressure
        return None
    
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
    def run_from_config(cfg_file: str,):
        assert cfg_file.endswith('.json'), "cfg_file must be a json file"
        with open(cfg_file, 'r') as f:
            cfg_dict = json.load(f)
        
        simulator = FipyStokeParallelSimulator.from_msh(
            source_msh_file=cfg_dict['source_msh_file'],
            dim=cfg_dict['dim'],
            viscosity=cfg_dict['viscosity'],
            velocityRelaxation=cfg_dict['velocityRelaxation'],
            pressureRelaxation=cfg_dict['pressureRelaxation'],
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


def debug_mpi():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t38'
    simulator = FipyStokeParallelSimulator.from_msh(
        os.path.join(base_dir, 't38_model.msh'),
        dim=2,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
    )        
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=22,
        face_center=[-0.05, 0.2],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=23
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=24
    )
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    simulator.define_boundary()
    
    if OpenMpiUtils.is_master_rank():
        start_time = OpenMpiUtils.get_time()
    simulator.run(
        iterations=60,
        log_step=2,
        save_dir=os.path.join(base_dir, 'cache'),
        use_tqdm=True,
        use_log=True,
    )
    if OpenMpiUtils.is_master_rank():
        end_time = OpenMpiUtils.get_time()
        OpenMpiUtils.print(f"Time taken: {end_time - start_time} seconds")


def debug_mpi_config():
    boundary_list = [
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
    ]
    
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t38'
    FipyStokeParallelSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=os.path.join(base_dir, 't38_model.msh'),
        dim=2,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
        boundary_list=boundary_list,
        iterations=60,
        convergence_threshold=1e-5,
        hot_steps=10,
        log_step=2,
        save_dir=os.path.join(base_dir, 'cache'),
        use_tqdm=True,
        use_log=True,
    )
    

def run_main():
    """
    config_path = '/home/admin123456/Desktop/work/gmsh_test/t38/config.json'
    cmd = f'mpirun -n 4 python scripts_py/simulator/fipy_simulator_parallel.py --cfg_file {config_path}'
    subprocess.run(
    cmd, shell=True,
    cwd='/home/admin123456/Desktop/work/MAPF_Pipeline'
    )
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    args = parser.parse_args()
    FipyStokeParallelSimulator.run_from_config(args.cfg_file)


if __name__ == '__main__':
    # debug_mpi()
    # debug_mpi_config()
    
    run_main()