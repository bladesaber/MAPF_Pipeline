import numpy as np
import dolfinx
from typing import Dict, List
import os
import pyvista

from ..simulator.openfoam_simulators import OpenFoamComSimulator, OpenFoamOrgSimulator
from ..dolfinx_Grad.dolfinx_utils import MeshUtils


class FoamSimulator(object):
    def __init__(
            self,
            cond_name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            openfoam_cfg: Dict,
            dolfinFun2openfoamFun_nameMap: Dict[str, str]
    ):
        self.cond_name = cond_name
        self.openfoam_cfg = openfoam_cfg
        self.simulator = OpenFoamOrgSimulator(f"{cond_name}_simulator", domain, cell_tags, facet_tags, openfoam_cfg)
        self.dolfinFun2openfoamFun_nameMap = dolfinFun2openfoamFun_nameMap

        self.vector_cell_functionSpace = dolfinx.fem.VectorFunctionSpace(domain, ("DG", 0))
        self.scalar_cell_functionSpace = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
        self.vector_point_functionSpace = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.scalar_point_functionSpace = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

        self.info_dict = {
            'inlet_condition': {},
            'boundary_condition': {},
            'outlet_condition': {}
        }

    def init_env(self):
        res_info = self.simulator.run(
            run_name=self.cond_name, source_msh_file=self.openfoam_cfg['source_msh_file'],
            with_preprocess=True, with_simulate=False, with_postprocess=False
        )
        self.info_dict.update({'run_dir': res_info['run_dir']})

    def add_inlet_condition(self, marker: int, velocity_vector: np.ndarray):
        self.info_dict['inlet_condition'].update({marker: velocity_vector})

    def add_boundary_condition(self, marker: int):
        self.info_dict['boundary_condition'].update({marker: 0.0})

    def add_outlet_condition(self, marker: int, pressure_scalar: float):
        self.info_dict['outlet_condition'].update({marker: pressure_scalar})

    def simulate(self, modify_data: Dict[str, np.ndarray], return_point_field=True, apply_boundary=False):
        for dolfin_name, data in modify_data.items():
            if (data.ndim == 2) and (data.shape[1] == 1):
                data = data.reshape(-1)
            OpenFoamComSimulator.modify_internal_field(
                os.path.join(self.info_dict['run_dir'], f"0/{self.dolfinFun2openfoamFun_nameMap[dolfin_name]}"), data
            )

        res = self.simulator.run(self.cond_name, with_preprocess=False, with_simulate=True, with_postprocess=True)
        res_vtk: pyvista.UnstructuredGrid = res['res_vtk']

        nut_np = res_vtk.cell_data['nut'][self.simulator.openfoamVtk2dolfinMesh_cell_center_idxs]
        nut_tmp = dolfinx.fem.Function(self.scalar_cell_functionSpace, name='nut_tmp')
        nut_tmp.x.array[:] = nut_np

        if return_point_field:
            U_np = res_vtk.point_data['U'][self.simulator.openfoamVtk2dolfinMesh_point_idxs, :]
            U_tmp = dolfinx.fem.Function(self.vector_point_functionSpace, name='U_tmp')
            U_tmp.x.array[:] = U_np.reshape(-1)

            p_np = res_vtk.point_data['p'][self.simulator.openfoamVtk2dolfinMesh_point_idxs]
            p_tmp = dolfinx.fem.Function(self.scalar_point_functionSpace, name='p_tmp')
            p_tmp.x.array[:] = p_np

            if apply_boundary:
                U_np = U_tmp.x.array.reshape((-1, 3))
                for marker, vector_np in self.info_dict['inlet_condition'].items():
                    idxs = MeshUtils.extract_entity_dofs(
                        self.vector_point_functionSpace, dim=self.simulator.fdim,
                        entities_idxs=MeshUtils.extract_facet_entities(
                            self.simulator.domain, self.simulator.facet_tags, marker
                        )
                    )
                    U_np[idxs, :] = vector_np.reshape(-1)
                for marker, _ in self.info_dict['boundary_condition'].items():
                    idxs = MeshUtils.extract_entity_dofs(
                        self.vector_point_functionSpace, dim=self.simulator.fdim,
                        entities_idxs=MeshUtils.extract_facet_entities(
                            self.simulator.domain, self.simulator.facet_tags, marker
                        )
                    )
                    U_np[idxs, :] = 0.0
                U_tmp.x.array[:] = U_np.reshape(-1)

                for marker, v in self.info_dict['outlet_condition'].items():
                    idxs = MeshUtils.extract_entity_dofs(
                        self.scalar_point_functionSpace, dim=self.simulator.fdim,
                        entities_idxs=MeshUtils.extract_facet_entities(
                            self.simulator.domain, self.simulator.facet_tags, marker
                        )
                    )
                    p_tmp.x.array[idxs] = v

        else:
            U_np = res_vtk.cell_data['U'][self.simulator.openfoamVtk2dolfinMesh_cell_center_idxs, :]
            U_tmp = dolfinx.fem.Function(self.vector_cell_functionSpace, name='U_tmp')
            U_tmp.x.array[:] = U_np.reshape(-1)

            p_np = res_vtk.cell_data['p'][self.simulator.openfoamVtk2dolfinMesh_cell_center_idxs]
            p_tmp = dolfinx.fem.Function(self.scalar_cell_functionSpace, name='p_tmp')
            p_tmp.x.array[:] = p_np

        return U_tmp, p_tmp, nut_tmp
