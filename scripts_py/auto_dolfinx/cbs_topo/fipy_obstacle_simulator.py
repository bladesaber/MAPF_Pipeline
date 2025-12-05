import os, shutil
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.neighbors import KDTree
import pyvista

from fipy import Gmsh3D, Gmsh2D
from fipy import CellVariable, FaceVariable
from fipy.variables.faceGradVariable import _FaceGradVariable
from fipy import DiffusionTerm, ConvectionTerm
from fipy.tools import numerix
from fipy import ImplicitSourceTerm

from scripts_py.simulator.fipy_simulator import FipyStokeSimpleSimulator
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj


class FipyObstacleSimulator(FipyStokeSimpleSimulator):
    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        structure_obs_weight: np.ndarray,
        structure_obs_position: np.ndarray,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_mesh: SeqMeshObj|None = None,
    ):
        base_name = os.path.basename(source_msh_file).replace('.msh', '')        
        if structure_mesh is None:
            domain, cell_tags, facet_tags = MeshUtils.read_msh(source_msh_file, dim)
            structure_mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        with TemporaryDirectory() as temp_dir:
            temp_msh_file = os.path.join(temp_dir, f"{base_name}.msh")
            FipyStokeSimpleSimulator.msh_version_convert(source_msh_file, temp_msh_file, 'msh2')
            if structure_mesh.tdim == 3:
                mesh = Gmsh3D(temp_msh_file)
            elif structure_mesh.tdim == 2:
                mesh = Gmsh2D(temp_msh_file)
            else:
                raise ValueError(f"Invalid dimension: {structure_mesh.tdim}")
            
        return FipyObstacleSimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_obs_weight=structure_obs_weight,
            structure_obs_position=structure_obs_position,
        )
    
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: SeqMeshObj,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_obs_weight: np.ndarray,
        structure_obs_position: np.ndarray,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
        )
        
        self.structure_obs_weight = structure_obs_weight
        self.structure_obs_position = structure_obs_position
        
        '''
        # TODO: too unstable when source term is larger than 10.0, please adjust the source term to make it stable
        self.obs_weight_cell = CellVariable(mesh=self.mesh, name='obstacle_weight')
        self.x_drag = ImplicitSourceTerm(coeff=self.obs_weight_cell)  # type: ignore
        self.y_drag = ImplicitSourceTerm(coeff=self.obs_weight_cell)  # type: ignore
        if self.tdim == 3:
            self.z_drag = ImplicitSourceTerm(coeff=self.obs_weight_cell)  # type: ignore

        self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.]) + self.x_drag
        self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.]) + self.y_drag
        if self.tdim == 3:
            self.zVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 0., 1.]) + self.z_drag
        '''
        
        # ------ parameters may be changed in the future ------
        self.obs_cell_mask, self.obs_face_mask = self.get_obs_mask()
    
    def get_obs_mask(self) -> tuple[np.ndarray, np.ndarray]:
        obs_weight = self.structure_obs_weight[self.structure_mesh_to_mesh_idxs]
        obs_cell_mask = obs_weight > 1e-2
        
        obs_face_mask = np.zeros(self.face_centers.shape[0], dtype=bool)
        
        obs_face_positions = []
        structure_cell_mask_idxs = np.arange(0, self.structure_mesh.grid.n_cells)[self.structure_obs_weight > 1e-2]
        for cell_idx in structure_cell_mask_idxs:
            cell = self.structure_mesh.grid.get_cell(cell_idx)
            for face in cell.faces:
                face_center = np.array(face.center)
                obs_face_positions.append(face_center)
        
        if len(obs_face_positions) > 0:
            obs_face_positions = np.array(obs_face_positions)
            tree = KDTree(self.face_centers)
            dists_list, idxs_list = tree.query(obs_face_positions, k=1, return_distance=True)
            dists_list = np.concatenate(dists_list, axis=-1)
            assert np.max(dists_list) < 1e-8, "mesh point is face center of fipy cell"
            idxs_list = np.unique(np.concatenate(idxs_list, axis=-1))
            obs_face_mask[idxs_list] = True
        
        return obs_cell_mask, obs_face_mask
    
    def define_boundary(self):
        super().define_boundary()
        
        if self.obs_cell_mask.sum() > 0:
            self.xVelocity.constrain(0., where=self.obs_cell_mask)
            self.yVelocity.constrain(0., where=self.obs_cell_mask)
            if self.tdim == 3:
                self.zVelocity.constrain(0., where=self.obs_cell_mask)
        
        '''
        # TODO: Remove, it may cause some point in zone is not constrained.
        tree = KDTree(self.face_centers)
        idxs_list = tree.query_radius(
            self.structure_obs_position[self.structure_obs_weight > 1e-2, :], 
            r=self.obs_radius, 
            return_distance=False
        )
        idxs_list = np.concatenate(idxs_list, axis=-1)
        obs_face_mask = np.zeros(self.face_centers.shape[0], dtype=bool)
        obs_face_mask[idxs_list] = True
        self.xVelocity.constrain(0., where=obs_face_mask)
        self.yVelocity.constrain(0., where=obs_face_mask)
        if self.tdim == 3:
            self.zVelocity.constrain(0., where=obs_face_mask)
        '''
        
        '''
        # TODO: Will cause nan error
        self.xVelocity.constrain(0., where=self.obs_face_mask)
        self.yVelocity.constrain(0., where=self.obs_face_mask)
        if self.tdim == 3:
            self.zVelocity.constrain(0., where=self.obs_face_mask)
        '''
        
        # # ------ visualize obs weight ------
        # plotter = pyvista.Plotter()
        # # plotter.add_mesh(pyvista.PointSet(self.cell_centers), color='red', opacity=0.5)
        # # plotter.add_mesh(pyvista.PointSet(self.cell_centers[obs_cell_mask, :]), color='blue', opacity=0.5)
        # plotter.add_mesh(pyvista.PointSet(self.face_centers), color='red', opacity=0.5)
        # plotter.add_mesh(pyvista.PointSet(self.face_centers[self.obs_face_mask, :]), color='blue', opacity=0.5)
        # plotter.show()
        # # ------ visualize obs weight ------

    def apply_boundary_to_correction_velocity(self):
        super().apply_boundary_to_correction_velocity()
        
        if self.obs_face_mask.sum() > 0:
            self.face_velocity[0, self.obs_face_mask] = 0.
            self.face_velocity[1, self.obs_face_mask] = 0.
            if self.tdim == 3:
                self.face_velocity[2, self.obs_face_mask] = 0.


def debug():
    MeshUtils.msh_to_XDMF(
        msh_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.msh',
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.xdmf',
        dim=3,
    )
    structure_mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.xdmf')
    obstacle_center = np.array([0.5, 0.5, 0.25])
    obstacle_radius = 0.2
    
    structure_obs_position = structure_mesh.grid.cell_centers().points
    structure_obs_weight = np.zeros(structure_obs_position.shape[0])
    obstacle_bool = np.linalg.norm(structure_obs_position[:, :2] - obstacle_center.reshape((1, -1))[:, :2], axis=1) < obstacle_radius
    structure_obs_weight[obstacle_bool] = 1000.0
    structure_mesh.grid.cell_data['obstacle'] = structure_obs_weight
    
    simulator = FipyObstacleSimulator.from_msh(
        '/home/admin123456/Desktop/work/gmsh_test/t28/model.msh',
        dim=3,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
        structure_mesh=structure_mesh,
        structure_obs_weight=structure_obs_weight,
        structure_obs_position=structure_obs_position,
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


if __name__ == '__main__':
    debug()