import os, shutil
from tempfile import TemporaryDirectory
import argparse

import dolfinx

from meshio import xdmf
import numpy as np
import pyvista
from mpi4py import MPI
import json

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix
from fipy import ImplicitSourceTerm

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.simulator.fipy_simulator import FipyStokeSimulator
from scripts_py.simulator.fipy_simulator_parallel import FipyStokeParallelSimulator
from scripts_py.openmpi_utils import OpenMpiUtils


class FipyStokeTopoSimulator(FipyStokeSimulator):
    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_mesh: SeqMeshObj|None = None,
        structure_alpha: np.ndarray|None = None,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        base_name = os.path.basename(source_msh_file).replace('.msh', '')        
        if structure_mesh is None:
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim)
            structure_mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        with TemporaryDirectory() as temp_dir:
            temp_msh_file = os.path.join(temp_dir, f"{base_name}.msh")
            FipyStokeSimulator.msh_version_convert(source_msh_file, temp_msh_file, 'msh2')
            if structure_mesh.tdim == 3:
                mesh = Gmsh3D(temp_msh_file)
            elif structure_mesh.tdim == 2:
                mesh = Gmsh2D(temp_msh_file)
            else:
                raise ValueError(f"Invalid dimension: {structure_mesh.tdim}")
            
        return FipyStokeTopoSimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_alpha=structure_alpha,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
        )
    
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: SeqMeshObj,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_alpha: np.ndarray|None=None,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
        )
        
        self.structure_alpha = structure_alpha
        if self.structure_alpha is not None:
            self.alpha_cell = CellVariable(mesh=self.mesh, name='alpha')
            self.alpha_cell.setValue(self.structure_alpha[self.structure_mesh_to_mesh_idxs])
            
            # # ------ visualize topology weight ------
            # plotter = pyvista.Plotter()
            # cell_center = pyvista.PointSet(self.cell_centers)
            # cell_center.point_data['alpha'] = self.alpha_cell.value
            # cell_center.warp_by_scalar(scalars='alpha')
            # plotter.add_mesh(cell_center, opacity=1.0)
            # plotter.show()
            # # ------ visualize topology weight ------
        
            # TODO: i don't understand why the source term is negative?
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
            
            '''
            # Remove: unstable
            self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.]) - self.alpha_cell * self.xVelocity
            self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.]) - self.alpha_cell * self.yVelocity
            if self.tdim == 3:
                self.zVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 0., 1.]) - self.alpha_cell * self.zVelocity
            '''


class FipyStokeParallelTopoSimulator(FipyStokeParallelSimulator):
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_mesh: SeqMeshObj|None,
        structure_alpha: np.ndarray|None=None,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
        )        
        if OpenMpiUtils.is_master_rank():
            if structure_alpha is not None:
                structure_alpha = structure_alpha[self.structure_mesh_to_mesh_idxs]
                data_dict = {'structure_alpha': structure_alpha, 'state': True}
                OpenMpiUtils.send_to_all(data_dict, tag=100)
            else:
                data_dict = {'state': False}
                OpenMpiUtils.send_to_all(data_dict, tag=100)
        else:
            data_dict = OpenMpiUtils.recv(source=0, tag=100)
        
        if data_dict['state']:
            # self.structure_alpha = data_dict['structure_alpha'][mesh.cellGlobalIDs]  # type: ignore
            # self.alpha_cell = CellVariable(mesh=self.mesh, name='alpha')
            # self.alpha_cell.setValue(self.structure_alpha)
            
            # Fipy will handle the global value automatically            
            self.alpha_cell = CellVariable(mesh=self.mesh, name='alpha', value=data_dict['structure_alpha'])

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
        
        OpenMpiUtils.synchronize_all_ranks()
        

    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_alpha: np.ndarray|None = None,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        # ------ initialize the partition msh file ------
        num_partitions = OpenMpiUtils.get_mpi_size()
        base_dir = os.path.dirname(source_msh_file)
        base_name = os.path.basename(source_msh_file).replace('.msh', '')
        partition_msh_file = os.path.join(base_dir, f"{base_name}_partition_{num_partitions}.msh")
        
        if OpenMpiUtils.is_master_rank():
            FipyStokeParallelSimulator.msh_version_convert(source_msh_file, partition_msh_file, num_partitions, 'msh2')
            # TODO: why use gmshio will cause alpha shift?
            # domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim, comm=MPI.COMM_SELF)
            # structure_mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
            
            # TODO: why use xdmf is correct?
            xdmf_file = source_msh_file.replace('.msh', '.xdmf')
            MeshUtils.msh_to_XDMF(source_msh_file, xdmf_file, dim=dim, comm=MPI.COMM_SELF)
            structure_mesh = SeqMeshObj(xdmf_file=xdmf_file, comm=MPI.COMM_SELF)
        else:
            structure_mesh = None
        OpenMpiUtils.synchronize_all_ranks()
        
        # # ------ visual structure alpha ------
        # if OpenMpiUtils.is_master_rank():
        #     assert structure_mesh is not None, "structure_mesh is not initialized"
        #     assert structure_alpha is not None, "structure_alpha is not initialized"
        #     plotter = pyvista.Plotter()
        #     plotter.add_mesh(structure_mesh.grid, color='black', style='wireframe')
        #     plotter.add_mesh(pyvista.PointSet(structure_mesh.cell_centers[structure_alpha > 1.0, :]), color='red', point_size=2.0)
        #     plotter.add_mesh(pyvista.PointSet(structure_mesh.cell_centers[structure_alpha < 1.0, :]), color='blue', point_size=2.0)
        #     plotter.show()
        # # ------ visual structure alpha ------
        
        if dim == 3:
            mesh = Gmsh3D(partition_msh_file)
        elif dim == 2:
            mesh = Gmsh2D(partition_msh_file)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
                
        return FipyStokeParallelTopoSimulator(
            mesh=mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_mesh=structure_mesh,
            structure_alpha=structure_alpha,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
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
        structure_alpha_file: str|None = None,
        init_x_velocity_file: str|None = None,
        init_y_velocity_file: str|None = None,
        init_z_velocity_file: str|None = None,
        init_pressure_file: str|None = None,
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
            'structure_alpha_file': structure_alpha_file,
            'boundary_list': [],
            'init_x_velocity_file': init_x_velocity_file,
            'init_y_velocity_file': init_y_velocity_file,
            'init_z_velocity_file': init_z_velocity_file,
            'init_pressure_file': init_pressure_file,
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
        
        if cfg_dict['structure_alpha_file'] is not None and OpenMpiUtils.is_master_rank():
            if cfg_dict['structure_alpha_file'].endswith('.npy'):
                structure_alpha = np.load(cfg_dict['structure_alpha_file'], allow_pickle=True)
            else:
                raise ValueError(f"Invalid structure alpha file: {cfg_dict['structure_alpha_file']}")
        else:
            structure_alpha = None
        
        if cfg_dict['init_x_velocity_file'] is not None:
            if cfg_dict['init_x_velocity_file'].endswith('.npy'):
                init_x_velocity = np.load(cfg_dict['init_x_velocity_file'], allow_pickle=True)
            else:
                raise ValueError(f"Invalid init x velocity file: {cfg_dict['init_x_velocity_file']}")
        else:
            init_x_velocity = None
        
        if cfg_dict['init_y_velocity_file'] is not None:
            if cfg_dict['init_y_velocity_file'].endswith('.npy'):
                init_y_velocity = np.load(cfg_dict['init_y_velocity_file'], allow_pickle=True)
            else:
                raise ValueError(f"Invalid init y velocity file: {cfg_dict['init_y_velocity_file']}")
        else:
            init_y_velocity = None
        
        if cfg_dict['init_z_velocity_file'] is not None:
            if cfg_dict['init_z_velocity_file'].endswith('.npy'):
                init_z_velocity = np.load(cfg_dict['init_z_velocity_file'], allow_pickle=True)
            else:
                raise ValueError(f"Invalid init z velocity file: {cfg_dict['init_z_velocity_file']}")
        else:
            init_z_velocity = None
        
        if cfg_dict['init_pressure_file'] is not None:
            if cfg_dict['init_pressure_file'].endswith('.npy'):
                init_pressure = np.load(cfg_dict['init_pressure_file'], allow_pickle=True)
            else:
                raise ValueError(f"Invalid init pressure file: {cfg_dict['init_pressure_file']}")
        else:
            init_pressure = None
        
        simulator = FipyStokeParallelTopoSimulator.from_msh(
            source_msh_file=cfg_dict['source_msh_file'],
            dim=cfg_dict['dim'],
            viscosity=cfg_dict['viscosity'],
            velocityRelaxation=cfg_dict['velocityRelaxation'],
            pressureRelaxation=cfg_dict['pressureRelaxation'],
            structure_alpha=structure_alpha,
            init_x_velocity=init_x_velocity,
            init_y_velocity=init_y_velocity,
            init_z_velocity=init_z_velocity,
            init_pressure=init_pressure,
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
        
    def record(self, save_file: str):
        velocity = self.get_velocity(map_to_structure=True)
        pressure = self.get_pressure(map_to_structure=True)
        if self.alpha_cell is not None:
            alpha = self.alpha_cell.globalValue
        else:
            alpha = None
        
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
            if alpha is not None:
                self.structure_mesh.grid.cell_data['alpha_sim'] = alpha
            
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


def debug_3D():
    from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
    
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t28'
    MeshUtils.msh_to_XDMF(
        msh_file=os.path.join(base_dir, 't28_model.msh'),
        xdmf_file=os.path.join(base_dir, 't28_model.xdmf'),
        dim=3,
    )
    structure_mesh = SeqMeshObj(xdmf_file=os.path.join(base_dir, 't28_model.xdmf'))
    topotacle_center = np.array([0.5, 0.5, 0.25])
    topotacle_radius = 0.1
    
    structure_topo_position = structure_mesh.grid.cell_centers().points
    
    # test when all is 0.5
    # structure_topo_weight = np.ones(structure_topo_position.shape[0]) * 0.2
    
    # test when alpha has a hole
    structure_topo_weight = np.ones(structure_topo_position.shape[0]) - 1e-8
    topotacle_bool = np.linalg.norm(structure_topo_position[:, :2] - topotacle_center.reshape((1, -1))[:, :2], axis=1) < topotacle_radius
    structure_topo_weight[topotacle_bool] = 1e-8
    
    structure_alpha = TopoFluidOptimizer.alpha_fun(
        rho=structure_topo_weight,
        alpha_min=0.01,
        alpha_max=200.0,
        p=3.0,
        pow_operator=np.power
    )
    
    structure_mesh.grid.cell_data['topology_weight'] = structure_topo_weight
    structure_mesh.grid.cell_data['alpha'] = structure_alpha
    # structure_mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t28/test.vtu')
    
    simulator = FipyStokeTopoSimulator.from_msh(
        os.path.join(base_dir, 't28_model.msh'),
        dim=3,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
        structure_mesh=structure_mesh,
        structure_alpha=structure_alpha,
    )
        
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.2, 0.25],
        face_norm_vector=[1.0, 0.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=21
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=22
    )
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    simulator.define_boundary()
    
    simulator.run(
        iterations=60, 
        log_step=2, 
        convergence_threshold=1e-16,
        save_dir=os.path.join(base_dir, 'topo_cache')
    )


def debug_2D():
    from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
    
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t30'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    xdmf_file = os.path.join(base_dir, 't30_model.xdmf')
    MeshUtils.msh_to_XDMF(msh_file, xdmf_file, dim=2)
    
    structure_mesh = SeqMeshObj(xdmf_file=xdmf_file)
    topotacle_center = np.array([0.5, 0.5])
    topotacle_radius = 0.1
    
    structure_topo_position = structure_mesh.grid.cell_centers().points
    
    # test when all is 0.5
    # structure_topo_weight = np.ones(structure_topo_position.shape[0]) * 0.2
    
    # test when alpha has a hole
    structure_topo_weight = np.ones(structure_topo_position.shape[0]) - 1e-8
    topotacle_bool = np.linalg.norm(structure_topo_position[:, :2] - topotacle_center.reshape((1, -1))[:, :2], axis=1) < topotacle_radius
    structure_topo_weight[topotacle_bool] = 1e-8
    
    structure_alpha = TopoFluidOptimizer.alpha_fun(
        rho=structure_topo_weight,
        alpha_min=0.01,
        alpha_max=200.0,
        p=3.0,
        pow_operator=np.power
    )
    
    structure_mesh.grid.cell_data['topology_weight'] = structure_topo_weight
    structure_mesh.grid.cell_data['alpha'] = structure_alpha
    
    simulator = FipyStokeTopoSimulator.from_msh(
        os.path.join(base_dir, 't30_model.msh'),
        dim=2,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
        structure_mesh=structure_mesh,
        structure_alpha=structure_alpha,
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
    
    simulator.run(
        iterations=60, 
        convergence_threshold=1e-6,
        log_step=2, 
        save_dir=os.path.join(base_dir, 'topo_cache')
    )


def create_parallel_config():
    from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
    
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t39'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    xdmf_file = os.path.join(base_dir, 't30_model.xdmf')
    MeshUtils.msh_to_XDMF(msh_file, xdmf_file, dim=2)
    
    structure_mesh = SeqMeshObj(xdmf_file=xdmf_file)
    obstacle_center = np.array([0.5, 0.5])
    obstacle_radius = 0.1
    
    structure_topo_position = structure_mesh.grid.cell_centers().points
    
    # test when all is 0.5
    # structure_topo_weight = np.ones(structure_topo_position.shape[0]) * 0.2
    
    # test when alpha has a hole
    structure_topo_weight = np.ones(structure_topo_position.shape[0]) - 1e-8
    obstacle_bool = np.linalg.norm(structure_topo_position[:, :2] - obstacle_center.reshape((1, -1))[:, :2], axis=1) < obstacle_radius
    structure_topo_weight[obstacle_bool] = 1e-8

    structure_alpha = TopoFluidOptimizer.alpha_fun(
        rho=structure_topo_weight,
        alpha_min=0.01,
        alpha_max=200.0,
        p=3.0,
        pow_operator=np.power
    )
    structure_alpha_file = os.path.join(base_dir, 'structure_alpha.npy')
    np.save(structure_alpha_file, structure_alpha)
    
    # # ------ visual alpha ------
    plotter = pyvista.Plotter()
    plotter.add_mesh(structure_mesh.grid, color='black', style='wireframe')
    plotter.add_mesh(pyvista.PointSet(structure_mesh.cell_centers[structure_alpha > 1.0, :]), color='red', point_size=2.0)
    plotter.add_mesh(pyvista.PointSet(structure_mesh.cell_centers[structure_alpha < 1.0, :]), color='blue', point_size=2.0)
    plotter.show()
    # # ------ visual alpha ------
    
    FipyStokeParallelTopoSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=msh_file,
        dim=2,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
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
        use_tqdm=True,
        use_log=True,
        structure_alpha_file=structure_alpha_file,
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
    FipyStokeParallelTopoSimulator.run_from_config(args.cfg_file)


if __name__ == '__main__':
    # debug_3D()
    # debug_2D()
    
    # create_parallel_config()
    
    run_main()
    