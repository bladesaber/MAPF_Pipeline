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
        structure_topo_weight: np.ndarray,
        structure_topo_position: np.ndarray,
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
            
        return FipyTopologySimulator(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_topo_weight=structure_topo_weight,
            structure_topo_position=structure_topo_position,
        )
    
    def __init__(
        self,
        mesh: Gmsh2D|Gmsh3D,
        structure_mesh: SeqMeshObj,
        viscosity: float,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_alpha: np.ndarray,
    ):
        super().__init__(
            mesh=mesh,
            structure_mesh=structure_mesh,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
        )
        
        self.structure_alpha = structure_alpha
        
        # TODO: too unstable when source term is larger than 10.0, please adjust the source term to make it stable
        self.alpha_cell = CellVariable(mesh=self.mesh, name='alpha')
        self.alpha_cell.setValue(structure_alpha[self.structure_mesh_to_mesh_idxs])
        
        # TODO: check
        # ------ visualize topology weight ------
        plotter = pyvista.Plotter()
        plotter.add_mesh(pyvista.PointSet(self.cell_centers), color='red', opacity=0.5)
        plotter.add_mesh(pyvista.PointSet(self.cell_centers[self.alpha_cell.value > 1.0]), color='blue', opacity=0.5)
        plotter.show()
        # ------ visualize topology weight ------
        
        # self.x_drag = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore
        # self.y_drag = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore
        # if self.tdim == 3:
        #     self.z_drag = ImplicitSourceTerm(coeff=self.alpha_cell)  # type: ignore

        # self.xVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([1., 0., 0.]) + self.x_drag
        # self.yVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 1., 0.]) + self.y_drag
        # if self.tdim == 3:
        #     self.zVelocityEq = DiffusionTerm(coeff=viscosity) - self.pressure.grad.dot([0., 0., 1.]) + self.z_drag
    

def debug():
    from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
    
    MeshUtils.msh_to_XDMF(
        msh_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.msh',
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.xdmf',
        dim=3,
    )
    structure_mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t28/model.xdmf')
    topotacle_center = np.array([0.5, 0.5, 0.25])
    topotacle_radius = 0.1
    
    structure_topo_position = structure_mesh.grid.cell_centers().points
    structure_topo_weight = np.ones(structure_topo_position.shape[0]) - 1e-8
    topotacle_bool = np.linalg.norm(structure_topo_position[:, :2] - topotacle_center.reshape((1, -1))[:, :2], axis=1) < topotacle_radius
    structure_topo_weight[topotacle_bool] = 1e-8
    structure_alpha = TopoFluidOptimizer.alpha_fun(
        rho=structure_topo_weight,
        alpha_min=0.01,
        alpha_max=1000.0,
        p=3.0,
        pow_operator=np.power
    )
    
    structure_mesh.grid.cell_data['topology_weight'] = structure_topo_weight
    structure_mesh.grid.cell_data['alpha'] = structure_alpha
    # structure_mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t28/test.vtu')
    
    # simulator = FipyTopologySimulator.from_msh(
    #     '/home/admin123456/Desktop/work/gmsh_test/t28/model.msh',
    #     dim=3,
    #     viscosity=1.0 / 400.0,
    #     velocityRelaxation=0.5,
    #     pressureRelaxation=0.8,
    #     structure_mesh=structure_mesh,
    #     structure_topo_weight=structure_topo_weight,
    #     structure_topo_position=structure_topo_position,
    # )
        
    # inlet_bry = FluidBoundary(
    #     name='inlet',
    #     boundary_type='velocity',
    #     value=[1.0, 0.0, 0.0],
    #     marker=20,
    #     face_center=[-0.05, 0.3, 0.25],
    #     face_norm_vector=[1.0, 0.0, 0.0],
    #     face_radius=0.1
    # )
    # outlet_bry = FluidBoundary(
    #     name='outlet',
    #     boundary_type='pressure',
    #     value=0.0,
    #     marker=21
    # )
    # wall_bry = FluidBoundary(
    #     name='wall',
    #     boundary_type='wall',
    #     value=0.0,
    #     marker=22
    # )
    
    # simulator.add_boundary(inlet_bry)
    # simulator.add_boundary(outlet_bry)
    # simulator.add_boundary(wall_bry)
    # simulator.define_boundary()
    
    # simulator.run(iterations=60, log_step=2, save_dir='/home/admin123456/Desktop/work/gmsh_test/t28/cache')


if __name__ == '__main__':
    debug()