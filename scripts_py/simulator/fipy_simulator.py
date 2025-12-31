import os, shutil
from typing import Literal, Union

import dolfinx

from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import pyvista
from sklearn.neighbors import KDTree
from tqdm import tqdm

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix

from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.simulator.eary_stoper import EarlyStopper

class FipyStokeSimulator(object):
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
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
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
        self.pressure = CellVariable(mesh=self.mesh, name='pressure', value=init_pressure if init_pressure is not None else 0.0) # type: ignore
        self.xVelocity = CellVariable(mesh=self.mesh, name='X velocity', value=init_x_velocity if init_x_velocity is not None else 0.0) # type: ignore
        self.yVelocity = CellVariable(mesh=self.mesh, name='Y velocity', value=init_y_velocity if init_y_velocity is not None else 0.0) # type: ignore
        if self.tdim == 3:
            self.zVelocity = CellVariable(mesh=self.mesh, name='Z velocity', value=init_z_velocity if init_z_velocity is not None else 0.0) # type: ignore
        self.face_velocity = FaceVariable(mesh=self.mesh, rank=1)
        
        if self.tdim == 2:
            self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0.])
            self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1.])
        elif self.tdim == 3:
            self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.])
            self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.])
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

    def run(
        self, 
        iterations: int, 
        convergence_threshold: float = 1e-5,
        hot_steps: int = 10,
        log_step: int = 10, 
        save_dir: str|None = None, 
        use_tqdm: bool = False, 
        use_log: bool = True
    ) -> dict:
        if save_dir is not None:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        
        if use_tqdm and use_log:
            tqdm_bar = tqdm(total=iterations, desc='Fipy Simulator')
        
        stoper = EarlyStopper(hot_step=hot_steps, window_size=30, shift_step=5, atol=1e-6, rtol=1e-3, atol_inv=1e-3)
        stoper.define_var('x_residual', convergence_threshold)
        stoper.define_var('y_residual', convergence_threshold)
        if self.tdim == 3:
            stoper.define_var('z_residual', convergence_threshold)
        stoper.define_var('p_residual', convergence_threshold)
        stoper.define_var('continuity_residual', convergence_threshold)
        
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
            
            if log_step > 0 and use_log:
                if iter_idx % log_step == 0:
                    log = f'[Log: {iter_idx}]: x residual:{x_residual:.8f}, y residual:{y_residual:.8f}, '
                    if self.tdim == 3:
                        log += f'z residual:{z_residual:.8f},'
                    log += f' p residual:{p_residual:.8f}, continuity_residual:{np.max(np.abs(continuity_residual)):.8f}'
                    if use_tqdm:
                        tqdm_bar.update(log_step)
                        # tqdm_bar.set_description(log)
                    else:
                        print(log)
                    
                    if save_dir is not None:
                        self.record(save_file=os.path.join(save_dir, f'res_{iter_idx}.vtu'))
            
            if iter_idx > hot_steps and convergence_threshold > 0.0:
                residual_dict = {
                    'x_residual': x_residual,
                    'y_residual': y_residual,
                    'p_residual': p_residual,
                    'continuity_residual': np.max(np.abs(continuity_residual)),
                }
                if self.tdim == 3:
                    residual_dict['z_residual'] = z_residual
                stoper.add_var(residual_dict)
                convergence_flag = stoper.is_converged()
                if convergence_flag:
                    break
            
        res_dict = {
            'is_solve': True,
            'x_residual': x_residual,
            'y_residual': y_residual,
            'p_residual': p_residual,
            'continuity_residual': continuity_residual
        }
        if self.tdim == 3:
            res_dict['z_residual'] = z_residual
        return res_dict

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
        cell_centers = self.structure_mesh.grid.cell_centers().points[: , :self.tdim]        
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
    def msh_version_convert(source_msh_file: str, target_msh_file: str, target_version: Literal['msh2', 'msh4']):
        base_dir = os.path.dirname(target_msh_file)
        log_file = os.path.join(base_dir, 'msh_version_convert.log')
        run_bash = f"gmsh {source_msh_file} -format {target_version} -o {target_msh_file} -save > {log_file}"
        subprocess.run(run_bash, shell=True, check=True)

    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_mesh: MeshObj|None = None,
        init_x_velocity: np.ndarray|None = None,
        init_y_velocity: np.ndarray|None = None,
        init_z_velocity: np.ndarray|None = None,
        init_pressure: np.ndarray|None = None,
    ):
        base_name = os.path.basename(source_msh_file).replace('.msh', '')
        
        if structure_mesh is None:
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim)
            structure_mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        with TemporaryDirectory() as temp_dir:
            temp_msh_file = os.path.join(temp_dir, f"{base_name}.msh")
            FipyStokeSimulator.msh_version_convert(source_msh_file, temp_msh_file, 'msh2')
            if structure_mesh.tdim == 3:
                mesh = Gmsh3D(temp_msh_file)
            elif structure_mesh.tdim == 2:
                mesh = Gmsh2D(temp_msh_file)
            else:
                raise ValueError(f"Invalid dimension: {structure_mesh.tdim}")
            
        return FipyStokeSimulator(
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


def debug_3D():
    simulator = FipyStokeSimulator.from_msh(
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
    
    simulator.run(iterations=60, log_step=2, save_dir='/home/admin123456/Desktop/work/gmsh_test/t28/cache')


def debug_global_processor():
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
    debug_3D()
    

def debug_2D():
    import time
    
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t38'
    simulator = FipyStokeSimulator.from_msh(
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
    
    print('start running')
    start_time = time.time()
    simulator.run(iterations=60, log_step=2, save_dir=os.path.join(base_dir, 'cache'), use_tqdm=True, use_log=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == '__main__':
    # debug_global_processor()
    # debug_3D()
    debug_2D()
    pass