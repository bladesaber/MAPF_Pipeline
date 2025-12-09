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


class FipyTopologySimulator(FipyStokeSimpleSimulator):
    @staticmethod
    def from_msh(
        source_msh_file: str,
        dim: int,
        viscosity: float = 1.0 / 400.0,
        velocityRelaxation: float = 0.5,
        pressureRelaxation: float = 0.8,
        structure_mesh: SeqMeshObj|None = None,
        structure_alpha: np.ndarray|None = None,
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
            
        return FipyTopologySimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_alpha=structure_alpha,
        )
    
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: SeqMeshObj,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_alpha: np.ndarray|None=None,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
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
    
    simulator = FipyTopologySimulator.from_msh(
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
    
    simulator = FipyTopologySimulator.from_msh(
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


if __name__ == '__main__':
    # debug_3D()
    debug_2D()
    