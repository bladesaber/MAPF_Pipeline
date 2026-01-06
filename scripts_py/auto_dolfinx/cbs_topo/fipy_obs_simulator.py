import os, shutil
from tempfile import TemporaryDirectory
import argparse

import dolfinx

import numpy as np
from sklearn.neighbors import KDTree
import pyvista
import json
from mpi4py import MPI

from fipy import Gmsh3D, Gmsh2D

from scripts_py.simulator.fipy_simulator import FipyStokeSimulator
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.auto_dolfinx.cbs_topo.fipy_topo_simulator import FipyStokeTopoSimulator
from scripts_py.openmpi_utils import OpenMpiUtils


class FipyStokeObsTopoSimulator(FipyStokeTopoSimulator):
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        parallel_num: int,
        structure_mesh: MeshObj|None,
        structure_alpha: np.ndarray|None=None,
        structure_obs_weight: np.ndarray|None = None,
        structure_init_x_velocity: np.ndarray|None = None,
        structure_init_y_velocity: np.ndarray|None = None,
        structure_init_z_velocity: np.ndarray|None = None,
        structure_init_pressure: np.ndarray|None = None,
    ):
        super().__init__(
            mesh=mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_mesh=structure_mesh,
            parallel_num=parallel_num,
            structure_alpha=structure_alpha,
            structure_init_x_velocity=structure_init_x_velocity,
            structure_init_y_velocity=structure_init_y_velocity,
            structure_init_z_velocity=structure_init_z_velocity,
            structure_init_pressure=structure_init_pressure,
        )
        
        # TODO: do we need this?
        # if OpenMpiUtils.is_master_rank():
        #     assert MeshUtils.get_cell_type(self.structure_mesh.domain) in [
        #         dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.tetrahedron], \
        #         "[Error]: Only triangle and tetrahedron are supported."
        
        self.structure_obs_weight = structure_obs_weight
        self.obs_cell_mask = np.zeros(self.mesh.numberOfCells, dtype=bool)
        self.obs_face_mask = np.zeros(self.mesh.numberOfFaces, dtype=bool)        
        self.define_obs_mask(structure_obs_weight)
        
        if self.using_parallel:
            OpenMpiUtils.synchronize_all_ranks()
    
    def define_boundary(self):
        super().define_boundary()
        if self.obs_cell_mask.sum() > 0:
            self.xVelocity.constrain(0., where=self.obs_cell_mask)
            self.yVelocity.constrain(0., where=self.obs_cell_mask)
            if self.tdim == 3:
                self.zVelocity.constrain(0., where=self.obs_cell_mask)

    def apply_boundary_to_correction_velocity(self):
        super().apply_boundary_to_correction_velocity()
        if self.obs_face_mask.sum() > 0:
            self.face_velocity[0, self.obs_face_mask] = 0.
            self.face_velocity[1, self.obs_face_mask] = 0.
            if self.tdim == 3:
                self.face_velocity[2, self.obs_face_mask] = 0.
    
    def define_obs_mask(self, structure_cell_obs_weight: np.ndarray|None):
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            assert self.structure_mesh is not None, "structure_mesh is not initialized"
                
            data_dict = {'with_obs': False}
            if structure_cell_obs_weight is not None:
                structure_cell_mask_idxs = np.arange(0, self.structure_mesh.grid.n_cells)[structure_cell_obs_weight > 1e-2]
                obs_cell_positions = self.structure_mesh.cell_centers[structure_cell_mask_idxs]  # type: ignore
                                    
                obs_face_positions = []
                for cell_idx in structure_cell_mask_idxs:
                    cell = self.structure_mesh.grid.get_cell(cell_idx)
                    if self.structure_mesh.tdim == 2:
                        for edge in cell.edges:
                            face_center = np.array(edge.center)
                            obs_face_positions.append(face_center)
                    elif self.structure_mesh.tdim == 3:
                        for face in cell.faces:
                            face_center = np.array(face.center)
                            obs_face_positions.append(face_center)
                    else:
                        raise ValueError(f'Invalid cell type: {cell.GetCellType()}')
                    
                data_dict = {
                    'with_obs': len(obs_face_positions) > 0, 
                    'obs_face_positions': np.array(obs_face_positions)[:, :self.structure_mesh.tdim],
                    'obs_cell_positions': obs_cell_positions[:, :self.structure_mesh.tdim],
                }
        
        if self.using_parallel:
            if OpenMpiUtils.is_master_rank():
                OpenMpiUtils.send_to_all(data_dict, tag=100)
            else:
                data_dict = OpenMpiUtils.recv(source=0, tag=100)
                
        if data_dict['with_obs']:
            obs_face_positions = data_dict['obs_face_positions']           
            tree = KDTree(self.face_centers)
            dists_list, idxs_list = tree.query(obs_face_positions, k=1, return_distance=True)
            dists_list = np.concatenate(dists_list, axis=-1)
            idxs_list = np.concatenate(idxs_list, axis=-1)            
            idxs_list = idxs_list[dists_list < 1e-6]
            idxs_list = np.unique(idxs_list)
            self.obs_face_mask[idxs_list] = True
        
            obs_cell_positions = data_dict['obs_cell_positions']
            tree = KDTree(self.cell_centers)
            dists_list, idxs_list = tree.query(obs_cell_positions, k=1, return_distance=True)
            dists_list = np.concatenate(dists_list, axis=-1)
            idxs_list = np.concatenate(idxs_list, axis=-1)
            idxs_list = idxs_list[dists_list < 1e-6]
            idxs_list = np.unique(idxs_list)
            self.obs_cell_mask[idxs_list] = True

    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_alpha: np.ndarray|None = None,
        structure_obs_weight: np.ndarray|None = None,
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
            structure_mesh = MeshObj(
                xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags
            )
        else:
            structure_mesh = None
                
        if dim == 3:
            mesh = Gmsh3D(fipy_msh_file)
        elif dim == 2:
            mesh = Gmsh2D(fipy_msh_file)
        else:
            raise ValueError(f"Invalid dimension: {dim}")

        return FipyStokeObsTopoSimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            parallel_num=num_partitions,
            structure_alpha=structure_alpha,
            structure_obs_weight=structure_obs_weight,
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
        vtu_obs_weight_field: str = 'obstacle',
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
            'vtu_obs_weight_field': vtu_obs_weight_field,
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
        init_obs_weight = None
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
            
            if cfg_dict['vtu_obs_weight_field'] in init_vtu.cell_data.keys():
                init_obs_weight = init_vtu.cell_data[cfg_dict['vtu_obs_weight_field']]
                
        simulator = FipyStokeObsTopoSimulator.from_msh(
            source_msh_file=cfg_dict['source_msh_file'],
            dim=cfg_dict['dim'],
            viscosity=cfg_dict['viscosity'],
            velocityRelaxation=cfg_dict['velocityRelaxation'],
            pressureRelaxation=cfg_dict['pressureRelaxation'],
            structure_alpha=init_alpha,
            structure_obs_weight=init_obs_weight,
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
            if self.structure_obs_weight is not None:
                self.structure_mesh.grid.cell_data['obstacle'] = self.structure_obs_weight
            
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
  

def debug_2D():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t40'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    obs_center = np.array([0.5, 0.5])
    obs_radius = 0.1
    
    obs_position = mesh.grid.cell_centers().points
    obs_weight = np.zeros(obs_position.shape[0])
    obstacle_bool = np.linalg.norm(obs_position[:, :2] - obs_center.reshape((1, -1))[:, :2], axis=1) < obs_radius
    obs_weight[obstacle_bool] = 1.0
    
    mesh.grid.cell_data.clear()
    mesh.grid.cell_data['obstacle'] = obs_weight
    init_vtu_file = os.path.join(base_dir, 'init.vtu')
    mesh.grid.save(init_vtu_file)
    
    FipyStokeObsTopoSimulator.create_simulation_config(
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
        use_tqdm=False,
        use_log=True,
        structure_init_vtu_file=init_vtu_file,
        vtu_velocity_field='velocity',
        vtu_pressure_field='pressure',
        vtu_alpha_field='alpha',
        vtu_obs_weight_field='obstacle',
    )


def debug_3D():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t33'
    msh_file = os.path.join(base_dir, 't33_model.msh')
    
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    obs_center = np.array([0.5, 0.5, 0.15])
    obs_radius = 0.1
    
    obs_position = mesh.grid.cell_centers().points
    obs_weight = np.zeros(obs_position.shape[0])
    obs_bool = np.linalg.norm(obs_position - obs_center.reshape((1, -1)), axis=1) < obs_radius
    obs_weight[obs_bool] = 1.0
    # obs_weight = ShapeUtils.convert_weight_to_ConvexHull(mesh, obs_weight)
    
    mesh.grid.cell_data.clear()
    mesh.grid.cell_data['obstacle'] = obs_weight
    init_vtu_file = os.path.join(base_dir, 'init.vtu')
    mesh.grid.save(init_vtu_file)
    
    FipyStokeObsTopoSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=msh_file,
        dim=3,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        boundary_list=[
            FluidBoundary(
                name='inlet',
                boundary_type='velocity',
                value=[1.0, 0.0, 0.0],
                marker=20,
                face_center=[-0.05, 0.15, 0.15],
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
        vtu_obs_weight_field='obstacle',
    )


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--force_msh_init', type=bool, default=False)
    args = parser.parse_args()
    FipyStokeObsTopoSimulator.run_from_config(args.cfg_file, args.force_msh_init)


if __name__ == '__main__':
    # debug_2D()
    # debug_3D()
    
    run_main()