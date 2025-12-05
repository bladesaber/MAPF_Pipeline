import os, shutil
from typing import Literal, Union

from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import pyvista
from sklearn.neighbors import KDTree

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj


class FipyStokeSimpleSimulator(object):
    '''
    Gmsh2D/Gmsh3D only accept .msh version 2
    '''
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: MeshObj,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
    ):
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
        
        self.boundaries: dict[str, FluidBoundary] = {}
        
        # ------ define momentum equation ------
        self.pressure = CellVariable(mesh=self.mesh, name='pressure')
        self.xVelocity = CellVariable(mesh=self.mesh, name='X velocity')
        self.yVelocity = CellVariable(mesh=self.mesh, name='Y velocity')
        if self.tdim == 3:
            self.zVelocity = CellVariable(mesh=self.mesh, name='Z velocity')
        self.face_velocity = FaceVariable(mesh=self.mesh, rank=1)
        
        self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.])
        self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.])
        if self.tdim == 3:
            self.zVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 0., 1.])
        
        # ------ define pressure correction equation ------
        self.ap = CellVariable(mesh=self.mesh, value=1.)
        self.coeff = 1./ self.ap.arithmeticFaceValue * self.mesh._faceAreas * self.mesh._cellDistances
        self.pressureCorrectionEq = DiffusionTerm(coeff=self.coeff) - self.face_velocity.divergence
        
        self.pressureCorrection = CellVariable(mesh=self.mesh)
        self.pressureCorrectionSolver = self.pressureCorrectionEq.getDefaultSolver(tolerance=1e-10, iterations=2000)
        
        # ------ define other variables ------
        self.volume = CellVariable(mesh=self.mesh, value=self.mesh.cellVolumes, name='Volume')
        self.control_volume = self.volume.arithmeticFaceValue
        
        self.cell_centers = self.mesh.cellCenters.value.T
        self.face_centers = self.mesh.faceCenters.value.T
    
    def add_boundary(self, boundary: FluidBoundary):
        self.boundaries[boundary.name] = boundary
    
    def define_boundary(self):
        for name, bry_info in self.boundaries.items():
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

            elif bry_info.boundary_type == 'wall':
                self.xVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
                self.yVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)
                if self.tdim == 3:
                    self.zVelocity.constrain(0., where=self.mesh.physicalFaceMap == bry_info.marker)

            else:
                raise ValueError('[Error]: Invalid Boundary Type.')

    def apply_boundary_to_correction_velocity(self):
        for name, bry_info in self.boundaries.items():
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
            
            elif bry_info.boundary_type == 'wall':
                face_mask = self.mesh.physicalFaceMap == bry_info.marker
                self.face_velocity[0, face_mask] = 0.
                self.face_velocity[1, face_mask] = 0.
                if self.tdim == 3:
                    self.face_velocity[2, face_mask] = 0.

    def run(self, iterations: int, log_step: int = 10, save_dir: str|None = None):
        if save_dir is not None:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
        for iter_idx in range(iterations):
            
            '''
            # ------ solve the momentum equation ------
            self.xVelocityEq.cacheMatrix()
            x_residual = self.xVelocityEq.sweep(var=self.xVelocity, underRelaxation=self.velocityRelaxation)
            y_residual = self.yVelocityEq.sweep(var=self.yVelocity, underRelaxation=self.velocityRelaxation)
            if self.tdim == 3:
                z_residual = self.zVelocityEq.sweep(var=self.zVelocity, underRelaxation=self.velocityRelaxation)
            # ------ solve the momentum equation ------

            # ------ update the ap coefficient from the matrix diagonal ------
            xmat = self.xVelocityEq.matrix
            self.ap[:] = -numerix.asarray(xmat.takeDiagonal())
            # ------ update the ap coefficient from the matrix diagonal ------
            '''
            
            # TODO I don't know which one is better
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
            
            if log_step > 0:
                if iter_idx % log_step == 0:
                    log = f'[Log: {iter_idx}]: x residual:{x_residual:.8f}, y residual:{y_residual:.8f}, '
                    if self.tdim == 3:
                        log += f'z residual:{z_residual:.8f},'
                    log += f' p residual:{p_residual:.8f}, continuity_residual:{np.max(np.abs(continuity_residual)):.8f}'
                    print(log)
                    
                    if save_dir is not None:
                        self.record(save_file=os.path.join(save_dir, f'res_{iter_idx}.vtu'))

    def eval_point(self, points: np.ndarray):
        pressure = self.pressure(points)
        
        vel_list = []
        vel_list.append(self.xVelocity(points).reshape((-1, 1)))
        vel_list.append(self.yVelocity(points).reshape((-1, 1)))
        if self.tdim == 3:
            vel_list.append(self.zVelocity(points).reshape((-1, 1)))
        velocity = np.concatenate(vel_list, axis=1)
        return pressure, velocity

    def get_mesh_to_structure_mesh_mapping(self) -> tuple[np.ndarray, np.ndarray]:
        cell_centers = self.structure_mesh.grid.cell_centers().points
        tree = KDTree(self.mesh.cellCenters.value.T)
        dists_list, idxs_list = tree.query(cell_centers, k=1, return_distance=True)
        dists_list = np.concatenate(dists_list, axis=-1)
        assert np.max(dists_list) < 1e-8, "mesh point is center of fipy cell"
        
        mesh_to_structure_mesh_idxs = np.concatenate(idxs_list, axis=-1)
        structure_mesh_to_mesh_idxs = np.argsort(mesh_to_structure_mesh_idxs)
        
        return mesh_to_structure_mesh_idxs, structure_mesh_to_mesh_idxs

    def get_velocity(self, map_to_structure: bool) -> np.ndarray:
        velocity_list = [
            self.xVelocity.value.copy().reshape((-1, 1)),
            self.yVelocity.value.copy().reshape((-1, 1))
        ]
        if self.tdim == 3:
            velocity_list.append(self.zVelocity.value.copy().reshape((-1, 1)))
        velocity = np.concatenate(velocity_list, axis=1)
        if map_to_structure:
            velocity = velocity[self.mesh_to_structure_mesh_idxs, :]
        return velocity
    
    def get_pressure(self, map_to_structure: bool) -> np.ndarray:
        pressure = self.pressure.value.copy()
        if map_to_structure:
            pressure = pressure[self.mesh_to_structure_mesh_idxs]
        return pressure
    
    def record(self, save_file: str):
        velocity = self.get_velocity(map_to_structure=True)
        pressure = self.get_pressure(map_to_structure=True)
        self.structure_mesh.grid.cell_data['pressure'] = pressure
        self.structure_mesh.grid.cell_data['velocity'] = velocity
        self.structure_mesh.grid.cell_data['velocity_mag'] = np.linalg.norm(velocity, axis=1)
        self.structure_mesh.grid.save(save_file)

    @staticmethod
    def msh_version_convert(source_msh_file: str, target_msh_file: str, target_version: Literal['msh2', 'msh4']):
        run_bash = f"gmsh {source_msh_file} -format {target_version} -o {target_msh_file} -save"
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
        base_dir = os.path.dirname(source_msh_file)
        base_name = os.path.basename(source_msh_file).replace('.msh', '')
        
        if structure_mesh is None:
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim)
            structure_mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        with TemporaryDirectory() as temp_dir:
            temp_msh_file = os.path.join(temp_dir, f"{base_name}.msh")
            FipyStokeSimpleSimulator.msh_version_convert(source_msh_file, temp_msh_file, 'msh2')
            if structure_mesh.tdim == 3:
                mesh = Gmsh3D(temp_msh_file)
            elif structure_mesh.tdim == 2:
                mesh = Gmsh2D(temp_msh_file)
            else:
                raise ValueError(f"Invalid dimension: {structure_mesh.tdim}")
            
        return FipyStokeSimpleSimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
        )


def debug():
    simulator = FipyStokeSimpleSimulator.from_msh(
        '/home/admin123456/Desktop/work/gmsh_test/t28/model.msh',
        dim=3,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
    )
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.3, 0.25],
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
    
    simulator.run(iterations=30, log_step=2, save_dir='/home/admin123456/Desktop/work/gmsh_test/t28/cache')


if __name__ == '__main__':
    debug()