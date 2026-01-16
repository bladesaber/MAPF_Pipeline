import os, shutil
from typing import Literal, Union
import argparse

import dolfinx
from mpi4py import MPI

import subprocess
import numpy as np
import pyvista
from sklearn.neighbors import KDTree
from tqdm import tqdm
import json
import time
from functools import partial

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix
from fipy import ImplicitSourceTerm

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, BoundaryUtils
from scripts_py.simulator.eary_stoper import EarlyStopper
from scripts_py.openmpi_utils import OpenMpiUtils
from scripts_py.functionspae_utils import FunctionSpaceUtils


class FipyStokeSimulator(object):
    '''Gmsh2D/Gmsh3D only accept .msh version 2'''
    def __init__(
        self,
        velocityRelaxation: float,
        pressureRelaxation: float,
        parallel_num: int,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: MeshObj|None,
        structure_viscosity: float|np.ndarray|list[float],
        structure_source_term: float|np.ndarray|list[float]|None = None,
        structure_obs_weight: np.ndarray|None = None,
        structure_init_x_velocity: np.ndarray|None = None,
        structure_init_y_velocity: np.ndarray|None = None,
        structure_init_z_velocity: np.ndarray|None = None,
        structure_init_pressure: np.ndarray|None = None,
        interpolation_ksp_option = {
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        },
        log_level: int = 2
    ):
        """
        log_level: int
            1: save redsidual message and last result
            2: save all progress report
        interpolation_ksp_option must use {
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        }, if not, will raise "petsc4py.PETSc.Error: error code 83"
        """
        self.log_level = log_level
        
        self.dolfinx_interpolation_ksp_option = interpolation_ksp_option
        self.dolfinx_velocity_bcs = []
        self.dolfinx_pressure_bcs = []
        
        self.parallel_num = parallel_num
        self.using_parallel = parallel_num > 1
        
        self.velocityRelaxation = velocityRelaxation
        self.pressureRelaxation = pressureRelaxation
        
        self.structure_mesh = structure_mesh
        self.mesh = mesh
        if isinstance(self.mesh, Gmsh2D):
            self.tdim = 2
        elif isinstance(self.mesh, Gmsh3D):
            self.tdim = 3
        else:
            raise ValueError('[Error]: Invalid Mesh Type.')
        self.fdim = self.tdim - 1
        
        self.mesh_to_structure_mesh_idxs, self.structure_mesh_to_mesh_idxs = self.get_mesh_to_structure_mesh_mapping()
        if self.using_parallel:
            if OpenMpiUtils.is_master_rank():
                data_dict = {
                    'mesh_to_structure_mesh_idxs': self.mesh_to_structure_mesh_idxs,
                    'structure_mesh_to_mesh_idxs': self.structure_mesh_to_mesh_idxs,
                }
                OpenMpiUtils.send_to_all(data_dict, tag=100)
            else:
                data_dict = OpenMpiUtils.recv(source=0, tag=100)
                self.mesh_to_structure_mesh_idxs = data_dict['mesh_to_structure_mesh_idxs']
                self.structure_mesh_to_mesh_idxs = data_dict['structure_mesh_to_mesh_idxs']
                
        self.boundaries: dict[str, FluidBoundary] = {}
        
        # ------ define momentum equation ------
        init_pressure = 0.0
        if structure_init_pressure is not None:
            init_pressure = structure_init_pressure[self.structure_mesh_to_mesh_idxs].reshape(-1)
        self.pressure = CellVariable(mesh=self.mesh, name='pressure', value=init_pressure) # type: ignore
        
        init_x_velocity = 0.0
        if structure_init_x_velocity is not None:
            init_x_velocity = structure_init_x_velocity[self.structure_mesh_to_mesh_idxs].reshape(-1)
        self.xVelocity = CellVariable(mesh=self.mesh, name='X velocity', value=init_x_velocity) # type: ignore
        
        init_y_velocity = 0.0
        if structure_init_y_velocity is not None:
            init_y_velocity = structure_init_y_velocity[self.structure_mesh_to_mesh_idxs].reshape(-1)
        self.yVelocity = CellVariable(mesh=self.mesh, name='Y velocity', value=init_y_velocity) # type: ignore
        
        if self.tdim == 3:
            init_z_velocity = 0.0
            if structure_init_z_velocity is not None:
                init_z_velocity = structure_init_z_velocity[self.structure_mesh_to_mesh_idxs].reshape(-1)
            self.zVelocity = CellVariable(mesh=self.mesh, name='Z velocity', value=init_z_velocity) # type: ignore
        
        self.face_velocity = FaceVariable(mesh=self.mesh, rank=1)
        
        if isinstance(structure_viscosity, (np.ndarray, list)):
            if isinstance(structure_viscosity, list):
                structure_viscosity = np.array(structure_viscosity)
            structure_viscosity = structure_viscosity[self.structure_mesh_to_mesh_idxs].reshape(-1)
            init_viscosity = CellVariable(mesh=self.mesh, name='viscosity', value=structure_viscosity) # type: ignore
        elif isinstance(structure_viscosity, (float, int)):
            init_viscosity = float(structure_viscosity)
        else:
            raise ValueError(f"Invalid viscosity type: {type(structure_viscosity)}")
        
        self.source_cell = None
        if structure_source_term is not None:
            if isinstance(structure_source_term, (np.ndarray, list)):
                if isinstance(structure_source_term, list):
                    structure_source_term = np.array(structure_source_term)
                init_source = structure_source_term[self.structure_mesh_to_mesh_idxs].reshape(-1)
            elif isinstance(structure_source_term, (float, int)):
                init_source = float(structure_source_term)
            else:
                raise ValueError(f"Invalid source term type: {type(structure_source_term)}")
            
            self.source_cell = CellVariable(mesh=self.mesh, name='source', value=init_source)  # type: ignore
            self.x_source = ImplicitSourceTerm(coeff=self.source_cell)  # type: ignore
            self.y_source = ImplicitSourceTerm(coeff=self.source_cell)  # type: ignore
            if self.tdim == 3:
                self.z_source = ImplicitSourceTerm(coeff=self.source_cell)  # type: ignore
            
            if self.tdim == 2:
                self.xVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([1., 0.]) - self.x_source
                self.yVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 1.]) - self.y_source
            elif self.tdim == 3:
                self.xVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([1., 0., 0.]) - self.x_source
                self.yVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 1., 0.]) - self.y_source
                self.zVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 0., 1.]) - self.z_source
        else:
            if self.tdim == 2:
                self.xVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([1., 0.])
                self.yVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 1.])
            elif self.tdim == 3:
                self.xVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([1., 0., 0.])
                self.yVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 1., 0.])
                self.zVelocityEq = DiffusionTerm(coeff=init_viscosity) - self.pressure.grad.dot([0., 0., 1.])
        
        # ------ define pressure correction equation ------
        self.ap = CellVariable(mesh=self.mesh, value=1.)
        self.coeff = 1. / self.ap.arithmeticFaceValue * self.mesh._faceAreas * self.mesh._cellDistances
        self.pressureCorrectionEq = DiffusionTerm(coeff=self.coeff) - self.face_velocity.divergence
        
        self.pressureCorrection = CellVariable(mesh=self.mesh)
        self.pressureCorrectionSolver = self.pressureCorrectionEq.getDefaultSolver(tolerance=1e-10, iterations=2000)
        
        # ------ define other variables ------
        self.volume = CellVariable(mesh=self.mesh, value=self.mesh.cellVolumes, name='Volume')
        self.control_volume = self.volume.arithmeticFaceValue
        
        self.cell_centers = self.mesh.cellCenters.value.T
        self.face_centers = self.mesh.faceCenters.value.T
        
        # ------ define obstacle related ------
        self.structure_obs_weight = structure_obs_weight
        self.obs_cell_mask = np.zeros(self.mesh.numberOfCells, dtype=bool)
        self.obs_face_mask = np.zeros(self.mesh.numberOfFaces, dtype=bool)
        self.define_obs_mask(structure_obs_weight)
        
        # ------ wait for all processes to finish initialization ------
        if self.using_parallel:
            OpenMpiUtils.synchronize_all_ranks()
    
    # ------ boundary related ------
    def add_boundary(self, boundary: FluidBoundary):
        self.boundaries[boundary.name] = boundary
    
    def define_boundary(self):
        for _, bry_info in self.boundaries.items():
            assert bry_info.boundary_type in ['velocity', 'pressure', 'wall'], 'Invalid boundary type' 
            if bry_info.boundary_type == 'velocity':
                velocity_values = FluidBoundary.inflow_velocity_exp(
                    pos=self.mesh.faceCenters.value,
                    velocity=np.array(bry_info.value),
                    tdim=self.tdim,
                    face_center=bry_info.face_center, # type: ignore
                    face_norm_vector=bry_info.face_norm_vector, # type: ignore
                    face_radius=bry_info.face_radius, # type: ignore
                )
                                                
                face_mask = self.mesh.physicalFaceMap == bry_info.marker
                self.xVelocity.constrain(velocity_values[0, :], where=face_mask)
                self.yVelocity.constrain(velocity_values[1, :], where=face_mask)
                if self.tdim == 3:
                    self.zVelocity.constrain(velocity_values[2, :], where=face_mask)

            elif bry_info.boundary_type == 'pressure':
                self.pressure.constrain(bry_info.value, where=self.mesh.physicalFaceMap == bry_info.marker)
                self.pressureCorrection.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
        
        # ------ set wall boundary at last ------
        for _, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'wall':
                self.xVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
                self.yVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
                if self.tdim == 3:
                    self.zVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
        
        # ------ apply obstacle boundary to velocity ------
        if self.obs_cell_mask.sum() > 0:
            self.xVelocity.constrain(0., where=self.obs_cell_mask)
            self.yVelocity.constrain(0., where=self.obs_cell_mask)
            if self.tdim == 3:
                self.zVelocity.constrain(0., where=self.obs_cell_mask)

        """ FIXME: can not be use in mpirun parallel mode
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            self.define_dolfinx_boundary()
        """

    def define_dolfinx_boundary(self):
        """
        function space transform is not consistent with strict boundary condition.
        so i just fix the wall and pressure boundary.
        """
        # ------ define dolfinx boundary for interpolation ------
        assert self.structure_mesh is not None, "structure_mesh is not initialized"

        self.dolfinx_velocity_bcs.clear()
        self.dolfinx_pressure_bcs.clear()
        point_vector_space = DolfinUtils.point_vector_function_space(self.structure_mesh.domain)
        point_scalar_space = DolfinUtils.point_scalar_function_space(self.structure_mesh.domain)

        for _, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                # inflow_velocity_exp = partial(
                #     FluidBoundary.inflow_velocity_exp,
                #     velocity=np.array(bry_info.value),
                #     tdim=self.structure_mesh.tdim,
                #     face_center=bry_info.face_center, # type: ignore
                #     face_norm_vector=bry_info.face_norm_vector, # type: ignore
                #     face_radius=bry_info.face_radius, # type: ignore
                #     method=bry_info.interpolate_method,
                # )
                # self.dolfinx_velocity_bcs.append(
                #     BoundaryUtils.define_dirichlet_bc(
                #         bc_func_space=point_vector_space, 
                #         value_func_space=point_vector_space, 
                #         dofs=MeshUtils.extract_entity_dofs(
                #             point_vector_space, self.structure_mesh.fdim,
                #             MeshUtils.extract_entity_indices(
                #                 self.structure_mesh.domain, style='facet', mesh_tags=self.structure_mesh.facet_tags, marker=bry_info.marker
                #             )
                #         ), 
                #         bc_value=inflow_velocity_exp, 
                #         is_mixed_element=False)
                # )
                pass
                
            elif bry_info.boundary_type == 'pressure':
                self.dolfinx_pressure_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=point_scalar_space, 
                        value_func_space=point_scalar_space, 
                        dofs=MeshUtils.extract_entity_dofs(
                            point_scalar_space, self.structure_mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.structure_mesh.domain, style='facet', mesh_tags=self.structure_mesh.facet_tags, marker=bry_info.marker
                            )
                        ), 
                        bc_value=bry_info.value, 
                        is_mixed_element=False)
                )
        
            elif bry_info.boundary_type == 'wall':
                self.dolfinx_velocity_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=point_vector_space, 
                        value_func_space=point_vector_space, 
                        dofs=MeshUtils.extract_entity_dofs(
                            point_vector_space, self.structure_mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.structure_mesh.domain, style='facet', mesh_tags=self.structure_mesh.facet_tags, marker=bry_info.marker
                            )
                        ), 
                        bc_value=dolfinx.fem.Function(point_vector_space), 
                        is_mixed_element=False)
                )
     
    def apply_boundary_to_correction_velocity(self):
        for _, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                velocity_values = FluidBoundary.inflow_velocity_exp(
                    pos=self.mesh.faceCenters.value,
                    velocity=np.array(bry_info.value),
                    tdim=self.tdim,
                    face_center=bry_info.face_center, # type: ignore
                    face_norm_vector=bry_info.face_norm_vector, # type: ignore
                    face_radius=bry_info.face_radius, # type: ignore
                )
                
                face_mask = self.mesh.physicalFaceMap == bry_info.marker
                self.face_velocity[0, face_mask] = velocity_values[0, face_mask]
                self.face_velocity[1, face_mask] = velocity_values[1, face_mask]
                if self.tdim == 3:
                    self.face_velocity[2, face_mask] = velocity_values[2, face_mask]
        
        # ------ set wall boundary at last ------
        for _, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'wall':
                face_mask = self.mesh.physicalFaceMap == bry_info.marker
                self.face_velocity[0, face_mask] = 0.
                self.face_velocity[1, face_mask] = 0.
                if self.tdim == 3:
                    self.face_velocity[2, face_mask] = 0.
        
        # ------ apply obstacle boundary to correction velocity ------
        if self.obs_face_mask.sum() > 0:
            self.face_velocity[0, self.obs_face_mask] = 0.
            self.face_velocity[1, self.obs_face_mask] = 0.
            if self.tdim == 3:
                self.face_velocity[2, self.obs_face_mask] = 0.

    # ------ obstacle related ------
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

    # ------ get related ------
    def get_mesh_to_structure_mesh_mapping(self):
        if self.using_parallel:
            global_fipy_cell_centers = self.mesh.cellCenters.globalValue
            global_fipy_cell_centers = np.copy(global_fipy_cell_centers).T
        else:
            global_fipy_cell_centers = self.mesh.cellCenters.value.T
            
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            assert self.structure_mesh is not None, "structure_mesh is not initialized"
            
            cell_centers = self.structure_mesh.grid.cell_centers().points[: , :self.tdim]
            tree = KDTree(global_fipy_cell_centers)
                
            dists_list, idxs_list = tree.query(cell_centers, k=1, return_distance=True)
            dists_list = np.concatenate(dists_list, axis=-1)
            assert np.max(dists_list) < 1e-8, "mesh point is center of fipy cell"
                
            mesh_to_structure_mesh_idxs = np.concatenate(idxs_list, axis=-1)
            structure_mesh_to_mesh_idxs = np.argsort(mesh_to_structure_mesh_idxs)
                
            return mesh_to_structure_mesh_idxs, structure_mesh_to_mesh_idxs
        return None, None

    def get_velocity(self, map_to_structure: bool) -> np.ndarray|None:
        if self.using_parallel:
            velocity_list = [
                self.xVelocity.globalValue.reshape((-1, 1)),
                self.yVelocity.globalValue.reshape((-1, 1))
            ]
            if self.tdim == 3:
                velocity_list.append(self.zVelocity.globalValue.reshape((-1, 1)))
        else:
            velocity_list = [
                self.xVelocity.value.copy().reshape((-1, 1)),
                self.yVelocity.value.copy().reshape((-1, 1))
            ]
            if self.tdim == 3:
                velocity_list.append(self.zVelocity.value.copy().reshape((-1, 1)))
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            velocity = np.concatenate(velocity_list, axis=1)
            if map_to_structure:
                velocity = velocity[self.mesh_to_structure_mesh_idxs, :]
            return velocity
        return None
    
    def get_pressure(self, map_to_structure: bool) -> np.ndarray|None:
        if self.using_parallel:
            pressure = self.pressure.globalValue
        else:
            pressure = self.pressure.value.copy()
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            if map_to_structure:
                pressure = pressure[self.mesh_to_structure_mesh_idxs]
            return pressure
        return None

    def get_source_term(self, map_to_structure: bool) -> np.ndarray|None:
        if self.source_cell is None:
            return None
        if self.using_parallel:
            source_value = self.source_cell.globalValue
        else:
            source_value = self.source_cell.value.copy()
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            if map_to_structure:
                source_value = source_value[self.mesh_to_structure_mesh_idxs]
            return source_value
        return None

    def eval_points(self, points: np.ndarray):
        """
        points: should be (dim, n_points)
            if is 2D, points should be (2, n_points)
            if is 3D, points should be (3, n_points)
        """
        # FIXME: non-finished
        if not self.using_parallel:
            pressure = self.pressure(points.T)
            vel_list = []
            vel_list.append(self.xVelocity(points).reshape((-1, 1)))
            vel_list.append(self.yVelocity(points).reshape((-1, 1)))
            if self.tdim == 3:
                vel_list.append(self.zVelocity(points).reshape((-1, 1)))
            velocity = np.concatenate(vel_list, axis=1)
            
            print(pressure)
            print(vel_list)
            
            return pressure, velocity
        else:
            raise NotImplementedError("Parallel evaluation is not implemented")

    # ------ record related ------
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
        if num_partitions == 1:
            run_bash = f"gmsh {source_msh_file} -format {target_version} -o {target_msh_file} -save > {log_file}"
        else:
            run_bash = f"gmsh {source_msh_file} -format {target_version} -part {num_partitions} -part_ghosts -o {target_msh_file} -save > {log_file}"
        subprocess.run(run_bash, shell=True, check=True)

    def record(self, save_file: str):
        velocity = self.get_velocity(map_to_structure=True)
        pressure = self.get_pressure(map_to_structure=True)
        source_value = self.get_source_term(map_to_structure=True)
        
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
            
            if source_value is not None:
                self.structure_mesh.grid.cell_data['source_term'] = source_value
            if self.structure_obs_weight is not None:
                self.structure_mesh.grid.cell_data['obstacle'] = self.structure_obs_weight
            
            """ FIXME: can not be use in mpirun parallel mode
            cell_vector_space = DolfinUtils.cell_vector_function_space(self.structure_mesh.domain)
            cell_velocity_fun = dolfinx.fem.Function(cell_vector_space)        
            cell_velocity_fun.x.array[:] = velocity[:, :self.tdim].reshape(-1)  # type: ignore
            point_velocity_fun = FunctionSpaceUtils.convert_cell_to_point(
                mesh=self.structure_mesh,
                func_data = cell_velocity_fun,  # type: ignore
                is_vector = True,
                method = 'L2',
                bcs = self.dolfinx_velocity_bcs,  # TODO: sholud i use here?
                with_dolfinx_function = True,
                ksp_option=self.dolfinx_interpolation_ksp_option
            )
            
            cell_scalar_space = DolfinUtils.cell_scalar_function_space(self.structure_mesh.domain)
            cell_pressure_fun = dolfinx.fem.Function(cell_scalar_space)
            cell_pressure_fun.x.array[:] = pressure.reshape(-1)  # type: ignore
            point_pressure_fun = FunctionSpaceUtils.convert_cell_to_point(
                mesh=self.structure_mesh,
                func_data = cell_pressure_fun,  # type: ignore
                is_vector = False,
                method = 'L2',
                bcs = self.dolfinx_pressure_bcs,  # TODO: sholud i use here?
                with_dolfinx_function = True,
                ksp_option=self.dolfinx_interpolation_ksp_option
            )
            
            point_velocity = point_velocity_fun.x.array.copy().reshape((-1, self.tdim))  # type: ignore
            if point_velocity.shape[1] == 2:
                point_velocity = np.concatenate([point_velocity, np.zeros((point_velocity.shape[0], 1))], axis=1)
            
            self.structure_mesh.grid.point_data['velocity'] = point_velocity
            self.structure_mesh.grid.point_data['pressure'] = point_pressure_fun.x.array.copy()  # type: ignore
            self.structure_mesh.grid.point_data['velocity_mag'] = np.linalg.norm(point_velocity, axis=1)
            """
            
            # ------ debug record ------
            self.structure_mesh.grid.save(save_file)

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
        start_time = OpenMpiUtils.get_time()
        
        if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
            if save_dir is not None:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            
            if use_tqdm and use_log:
                tqdm_bar = tqdm(total=iterations, desc='Fipy Simulator')
            
            stoper = EarlyStopper(hot_step=hot_steps, window_size=30, shift_step=5)
            stoper.define_var('x_residual', convergence_threshold)
            stoper.define_var('y_residual', convergence_threshold)
            if self.tdim == 3:
                stoper.define_var('z_residual', convergence_threshold)
            stoper.define_var('p_residual', convergence_threshold)
            stoper.define_var('continuity_residual', convergence_threshold)
        
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
            p_residual = self.pressureCorrectionEq.sweep(var=self.pressureCorrection, solver=self.pressureCorrectionSolver)
            continuity_residual = self.pressureCorrectionEq.RHSvector

            # ------ update the pressure and velocity ------
            self.pressure.setValue(self.pressure + self.pressureRelaxation * self.pressureCorrection)
            self.xVelocity.setValue(self.xVelocity - self.pressureCorrection.grad[0] / self.ap * self.mesh.cellVolumes)
            self.yVelocity.setValue(self.yVelocity - self.pressureCorrection.grad[1] / self.ap * self.mesh.cellVolumes)
            if self.tdim == 3:
                self.zVelocity.setValue(self.zVelocity - self.pressureCorrection.grad[2] / self.ap * self.mesh.cellVolumes)
            
            # ------ wait for all processes to finish calculating ------
            if self.using_parallel:
                OpenMpiUtils.synchronize_all_ranks()

            # ------ compute the global residual ------
            if self.using_parallel:
                global_x_residual: float = OpenMpiUtils.reduce(x_residual, mode='max', root=0)  # type: ignore
                global_y_residual: float = OpenMpiUtils.reduce(y_residual, mode='max', root=0)  # type: ignore
                if self.tdim == 3:
                    global_z_residual: float = OpenMpiUtils.reduce(z_residual, mode='max', root=0)  # type: ignore
                global_p_residual: float = OpenMpiUtils.reduce(p_residual, mode='max', root=0)  # type: ignore
                global_continuity_residual: float = OpenMpiUtils.reduce(np.max(np.abs(continuity_residual)), mode='max', root=0)  # type: ignore
            else:
                global_x_residual = x_residual
                global_y_residual = y_residual
                if self.tdim == 3:
                    global_z_residual = z_residual
                global_p_residual = p_residual
                global_continuity_residual = np.max(np.abs(continuity_residual))

            # ------ log the result ------
            if log_step > 0 and use_log:
                if iter_idx % log_step == 0:
                    if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
                        log = f'[Log: {iter_idx}]: x residual:{global_x_residual:.8f}, y residual:{global_y_residual:.8f}, '
                        if self.tdim == 3:
                            log += f'z residual:{global_z_residual:.8f},'
                        log += f' p residual:{global_p_residual:.8f}, continuity_residual:{global_continuity_residual:.8f}'
                        if use_tqdm:
                            tqdm_bar.update(log_step)
                            # tqdm_bar.set_description(log)
                        else:
                            print(log, flush=True)
                        
                    if save_dir is not None and self.log_level > 1:
                        self.record(save_file=os.path.join(save_dir, f'iter_{iter_idx}.vtu'))
            
            if iter_idx > hot_steps and convergence_threshold > 0.0:
                if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
                    residual_dict = {
                        'x_residual': global_x_residual,
                        'y_residual': global_y_residual,
                        'p_residual': global_p_residual,
                        'continuity_residual': global_continuity_residual,
                    }
                    if self.tdim == 3:
                        residual_dict['z_residual'] = global_z_residual
                    stoper.add_var(residual_dict)
                    convergence_flag = stoper.is_converged()
                    if self.using_parallel:
                        OpenMpiUtils.send_to_all(data={'is_solve': convergence_flag}, tag=100)
                    if convergence_flag:
                        break
                
                else:
                    if self.using_parallel:
                        msg = OpenMpiUtils.recv(source=0, tag=100)
                        if msg['is_solve']:
                            break
        
        if save_dir is not None:
            self.record(save_file=os.path.join(save_dir, 'result.vtu'))
            if (not self.using_parallel) or (self.using_parallel and OpenMpiUtils.is_master_rank()):
                residual_dict = {
                    'x_residual': global_x_residual,
                    'y_residual': global_y_residual,
                    'p_residual': global_p_residual,
                    'continuity_residual': global_continuity_residual,
                    'time_cost': OpenMpiUtils.get_time() - start_time,
                }
                if self.tdim == 3:
                    residual_dict['z_residual'] = global_z_residual
                with open(os.path.join(save_dir, 'residual.json'), 'w') as f:
                    json.dump(residual_dict, f, indent=4)

    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        structure_viscosity: float|np.ndarray|list[float],
        structure_source_term: float|np.ndarray|list[float]|None,
        structure_obs_weight: np.ndarray|None = None,
        structure_init_x_velocity: np.ndarray|None = None,
        structure_init_y_velocity: np.ndarray|None = None,
        structure_init_z_velocity: np.ndarray|None = None,
        structure_init_pressure: np.ndarray|None = None,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        force_msh_init: bool = False,
        log_level=2
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
        
        if dim == 3:
            mesh = Gmsh3D(fipy_msh_file)
        elif dim == 2:
            mesh = Gmsh2D(fipy_msh_file)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        
        return FipyStokeSimulator(
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            parallel_num=num_partitions,
            mesh=mesh,
            structure_mesh=structure_mesh,
            structure_viscosity=structure_viscosity,
            structure_source_term=structure_source_term,
            structure_obs_weight=structure_obs_weight,
            structure_init_x_velocity=structure_init_x_velocity,
            structure_init_y_velocity=structure_init_y_velocity,
            structure_init_z_velocity=structure_init_z_velocity,
            structure_init_pressure=structure_init_pressure,
            log_level=log_level
        )

    @staticmethod
    def create_simulation_config(
        cfg_file: str, 
        source_msh_file: str,
        dim: int,
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
        viscosity: float|None = None,
        structure_init_vtu_file: str|None = None,
        vtu_velocity_field: str = 'velocity',
        vtu_pressure_field: str = 'pressure',
        vtu_viscosity_field: str = 'viscosity',
        vtu_source_term_field: str = 'source_term',
        vtu_obs_weight_field: str = 'obstacle',
        log_level: int = 2,
    ):
        assert cfg_file.endswith('.json'), "cfg_file must be a json file"
        assert (viscosity is not None) or (structure_init_vtu_file is not None), "viscosity or structure_init_vtu_file is required"
        
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
            'vtu_viscosity_field': vtu_viscosity_field,
            'vtu_source_term_field': vtu_source_term_field,
            'vtu_obs_weight_field': vtu_obs_weight_field,
            'log_level': log_level,
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
        
        assert (cfg_dict['viscosity'] is not None) or (cfg_dict['structure_init_vtu_file'] is not None), "viscosity or structure_init_vtu_file is required"
        
        init_x_velocity = None
        init_y_velocity = None
        init_z_velocity = None
        init_pressure = None
        init_viscosity = cfg_dict['viscosity']
        init_source_term = None
        init_obs_weight = None
        
        if cfg_dict['structure_init_vtu_file'] is not None:
            init_vtu = pyvista.read(cfg_dict['structure_init_vtu_file'])
            
            if cfg_dict['viscosity'] is None:
                init_viscosity = init_vtu.cell_data[cfg_dict['vtu_viscosity_field']]
            
            if cfg_dict['vtu_velocity_field'] in init_vtu.cell_data.keys():
                init_velocity = init_vtu.cell_data[cfg_dict['vtu_velocity_field']]
                init_x_velocity = init_velocity[:, 0]
                init_y_velocity = init_velocity[:, 1]
                if cfg_dict['dim'] == 3:
                    init_z_velocity = init_velocity[:, 2]
            
            if cfg_dict['vtu_pressure_field'] in init_vtu.cell_data.keys():
                init_pressure = init_vtu.cell_data[cfg_dict['vtu_pressure_field']]
            
            if cfg_dict['vtu_source_term_field'] in init_vtu.cell_data.keys():
                init_source_term = init_vtu.cell_data[cfg_dict['vtu_source_term_field']]
            
            if cfg_dict['vtu_obs_weight_field'] in init_vtu.cell_data.keys():
                init_obs_weight = init_vtu.cell_data[cfg_dict['vtu_obs_weight_field']]
        
        assert init_viscosity is not None, "viscosity is not initialized"
        
        simulator = FipyStokeSimulator.from_msh(
            source_msh_file=cfg_dict['source_msh_file'],
            dim=cfg_dict['dim'],
            structure_viscosity=init_viscosity,
            structure_source_term=init_source_term,
            structure_obs_weight=init_obs_weight,
            structure_init_x_velocity=init_x_velocity,
            structure_init_y_velocity=init_y_velocity,
            structure_init_z_velocity=init_z_velocity,
            structure_init_pressure=init_pressure,
            velocityRelaxation=cfg_dict['velocityRelaxation'],
            pressureRelaxation=cfg_dict['pressureRelaxation'],
            force_msh_init=force_msh_init,
            log_level=cfg_dict['log_level'],
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


def run_main():
    """
    config_path = '/home/admin123456/Desktop/work/gmsh_test/t38/config.json'
    cmd = f'mpirun -n 4 python scripts_py/simulator/fipy_simulator.py --cfg_file {config_path} --force_msh_init True'
    subprocess.run(
        cmd, shell=True,
        cwd='/home/admin123456/Desktop/work/MAPF_Pipeline'
    )
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--force_msh_init', type=bool, default=False)
    args = parser.parse_args()
    FipyStokeSimulator.run_from_config(cfg_file=args.cfg_file, force_msh_init=args.force_msh_init)


def create_2D_config():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t38'
    msh_file = os.path.join(base_dir, 't38_model.msh')
    
    # ------ create original config ------
    FipyStokeSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'original_config.json'),
        source_msh_file=msh_file,
        dim=2,
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
        save_dir=os.path.join(base_dir, 'original_cache'),
        use_tqdm=False,
        use_log=True,
        viscosity=1.0 / 400.0,
    )
    
    # ------ generate init vtu ------
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-8, alpha_max=100.0, p=3.0, pow_operator=np.power):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    cell_centers = mesh.grid.cell_centers().points
    
    init_grid = mesh.grid.copy()
    init_grid.cell_data.clear()
    init_grid.point_data.clear()
    init_vtu_file = os.path.join(base_dir, 'init.vtu')
    
    source_center = np.array([0.5, 0.5])
    source_radius = 0.1
    source_weight = np.ones(cell_centers.shape[0]) - 1e-12
    source_weight[
        np.linalg.norm(cell_centers[:, :mesh.tdim] - source_center.reshape((1, -1))[:, :2], axis=1) < source_radius
    ] = 1e-12
    structure_source_term = alpha_fun(rho=source_weight)
    init_grid.cell_data['source_term'] = structure_source_term
    
    obs_center = np.array([0.5, 0.5])
    obs_radius = 0.1
    
    obs_weight = np.zeros(cell_centers.shape[0])
    obs_weight[
        np.linalg.norm(cell_centers[:, :mesh.tdim] - obs_center.reshape((1, -1))[:, :2], axis=1) < obs_radius
    ] = 1.0
    init_grid.cell_data['obstacle'] = obs_weight
    
    init_grid.save(init_vtu_file)
    
    # ------ create source term config ------
    FipyStokeSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'source_term_config.json'),
        source_msh_file=msh_file,
        dim=mesh.tdim,
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
        save_dir=os.path.join(base_dir, 'source_term_cache'),
        use_tqdm=False,
        use_log=True,
        viscosity=1.0 / 400.0,
        structure_init_vtu_file=init_vtu_file,
        vtu_source_term_field='source_term',
        vtu_obs_weight_field='obstacle_test',
    )
    
    # ------ create obstacle config ------
    FipyStokeSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'obstacle_config.json'),
        source_msh_file=msh_file,
        dim=mesh.tdim,
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
        save_dir=os.path.join(base_dir, 'obstacle_cache'),
        use_tqdm=False,
        use_log=True,
        viscosity=1.0 / 400.0,
        structure_init_vtu_file=init_vtu_file,
        vtu_source_term_field='source_term_test',
        vtu_obs_weight_field='obstacle',
    )


def create_3D_config():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t28'
    FipyStokeSimulator.create_simulation_config(
        cfg_file=os.path.join(base_dir, 'config.json'),
        source_msh_file=os.path.join(base_dir, 't28_model.msh'),
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
    )


def debug_global_processor():
    """
    # TODO: still wire problem need to fixed
    from mpi4py import MPI
    from petsc4py import PETSc
    
    def create_petsc_solver(
        comm=MPI.COMM_WORLD, 
        solver_setting: dict|None = None,
    ):
        solver: PETSc.KSP = PETSc.KSP().create(comm)  # type: ignore
        if solver_setting is None:
            return solver

        opts = PETSc.Options()  # type: ignore
        for key, value in solver_setting.items():
            opts.setValue(key, value)
        solver.setFromOptions()
        
        return solver

    ''' when initializing the solver, it throw error:
    Traceback (most recent call last):
  File "/home/admin123456/Desktop/work/MAPF_Pipeline/scripts_py/simulator/fipy_simulator.py", line 396, in <module>
    debug_global_processor()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/admin123456/Desktop/work/MAPF_Pipeline/scripts_py/simulator/fipy_simulator.py", line 391, in debug_global_processor
    debug()
    ~~~~~^^
  File "/home/admin123456/Desktop/work/MAPF_Pipeline/scripts_py/simulator/fipy_simulator.py", line 356, in debug
    simulator.run(iterations=30, log_step=2, save_dir='/home/admin123456/Desktop/work/gmsh_test/t28/cache')
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/admin123456/Desktop/work/MAPF_Pipeline/scripts_py/simulator/fipy_simulator.py", line 161, in run
    x_residual = self.xVelocityEq.sweep(var=self.xVelocity, underRelaxation=self.velocityRelaxation)
  File "/home/admin123456/.conda/envs/dolfinx-env/lib/python3.13/site-packages/fipy/terms/term.py", line 261, in sweep
    solver._solve()
    ~~~~~~~~~~~~~^^
  File "/home/admin123456/.conda/envs/dolfinx-env/lib/python3.13/site-packages/fipy/solvers/solver.py", line 196, in _solve
    x = self._solve_(L, x, b)
  File "/home/admin123456/.conda/envs/dolfinx-env/lib/python3.13/site-packages/fipy/solvers/petsc/petscKrylovSolver.py", line 217, in _solve_
    ksp.solve(b, x)
    ~~~~~~~~~^^^^^^
  File "petsc4py/PETSc/KSP.pyx", line 1782, in petsc4py.PETSc.KSP.solve
petsc4py.PETSc.Error: error code 83
[0] KSPSolve() at /home/conda/feedstock_root/build_artifacts/bld/rattler-build_petsc_1754877033/work/src/ksp/ksp/interface/itfunc.c:1091
[0] KSPSolve_Private() at /home/conda/feedstock_root/build_artifacts/bld/rattler-build_petsc_1754877033/work/src/ksp/ksp/interface/itfunc.c:914
[0] KSPSolve_PREONLY() at /home/conda/feedstock_root/build_artifacts/bld/rattler-build_petsc_1754877033/work/src/ksp/ksp/impls/preonly/preonly.c:20
[0] KSP of type preonly (application of preconditioner only) doesn't make sense with nonzero initial guess you probably want a KSP of type Richardson
    '''
    solver = create_petsc_solver(
        solver_setting={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
    )
    solver.view()
    solver.destroy()
    del solver
    solver = None    
    """
    

if __name__ == '__main__':
    # create_2D_config()
    # create_3D_config()
    run_main()
    