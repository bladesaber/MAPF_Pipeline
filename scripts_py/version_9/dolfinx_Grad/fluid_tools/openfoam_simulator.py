"""
Note:
referece: https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH

sudo apt-get install openfoam11
Add source /opt/openfoam11/etc/bashrc to ~/.bashrc and update

1. Please use Openfoam tool gmshToFoam *.msh create the geometry
2. Openfoam only support *.msh(version 2), So when you save the .msh file in GMSH,
   "File->Export->*.msh" please select (version 2 ASCII)
3. gmsh convert msh4 to msh2: gmsh *.msh -save -format msh2 -o *.msh
4. gmsh convert msh2 to msh4: gmsh *.msh -save -format msh4 -o *.msh
5. gmsh --help for help refo
6. please modify /constant/polyMesh/boundary:  wall boundary to wall type
7. in GMSH, please use Netgen to optimize the mesh, original mesh can usually be divergence
"""

import numpy as np
import pandas as pd
import os
import shutil
import pyvista
import dolfinx
import time

from ..simulator_convert import OpenFoamUtils
from ..remesh_helper import ReMesher


class OpenFoamSimulator(object):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            openfoam_cfg: dict,
            remove_conda_env=False,
            conda_sh='~/anaconda3/etc/profile.d/conda.sh',
            run_code='foamRun'
    ):
        self.re_init(domain, cell_tags, facet_tags)
        self.name = name
        self.openfoam_cfg: dict = openfoam_cfg
        self.cache_dir = self.openfoam_cfg['cache_dir']
        self.remove_conda_env = remove_conda_env
        self.conda_sh = conda_sh
        self.run_code = run_code

        if self.openfoam_cfg['configuration'].get('decomposeParDict'):
            self.run_parallel = True
            self.num_of_threats = int(
                self.openfoam_cfg['configuration']['decomposeParDict']['args']['numberOfSubdomains']
            )
        else:
            self.run_parallel = False

        self.k, self.epsilon, self.Re = self.compute_k_epsilon(
            U=self.openfoam_cfg['k-Epsilon_args']['abs_velocity_U'],
            L=self.openfoam_cfg['k-Epsilon_args']['characteristic_length_L'],
            viscosity=self.openfoam_cfg['k-Epsilon_args']['kinematic_viscosity']
        )
        print(f"[INFO]: guess Reynold Number: {self.Re}, k:{self.k}, epsilon:{self.epsilon}")

    def re_init(self, domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags):
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

    def run_simulate_process(self, tmp_dir, orig_msh_file, convert_msh2=False, with_debug=False):
        # ------ Step 1: process msh file
        msh_file = os.path.join(tmp_dir, 'model.msh')
        if orig_msh_file != msh_file:
            shutil.copy(orig_msh_file, msh_file)
        if convert_msh2:
            OpenFoamUtils.exec_cmd(OpenFoamUtils.get_msh_version_change_code(tmp_dir, msh_file, msh_file))

        # ------ Step 2: create simulation configuration
        for object_name in self.openfoam_cfg['configuration'].keys():
            object_info = self.openfoam_cfg['configuration'][object_name]
            OpenFoamUtils.create_foam_file(
                tmp_dir, object_info['location'], object_info['class_name'],
                object_name=object_name, arg_dict=object_info['args']
            )

        # ------ Step 3: gmshToFoam
        OpenFoamUtils.exec_cmd(OpenFoamUtils.get_gmsh2foam_code(tmp_dir, msh_file))

        # ------ Step 4: modify type dict
        OpenFoamUtils.modify_boundary_type(tmp_dir, modify_dict=self.openfoam_cfg['modify_type_dict'])

        # ------ Step 5: scale unit
        if self.openfoam_cfg['unit_scale'] is not None:
            OpenFoamUtils.exec_cmd(OpenFoamUtils.get_unit_scale_code(tmp_dir, scale=self.openfoam_cfg['unit_scale']))

        # ------ Step 6: run simulation
        if with_debug:
            tick_0 = time.time()

        if self.run_parallel:
            res = OpenFoamUtils.exec_cmd(OpenFoamUtils.get_simulation_parallel_code(
                tmp_dir, num_of_process=self.num_of_threats,
                remove_conda_env=self.remove_conda_env, conda_sh=self.conda_sh
            ))
            assert res.returncode == 0
        else:
            OpenFoamUtils.exec_cmd(OpenFoamUtils.get_simulation_code(tmp_dir, self.run_code))

        if with_debug:
            cost_time = time.time() - tick_0
            print(f"[DEBUG StateSystem]: cost_time:{cost_time:.2f}")

        # ------ Step 7: convert result to VTK
        OpenFoamUtils.exec_cmd(OpenFoamUtils.get_foam2vtk_code(tmp_dir))

        # ------ Step 8: return vtk result
        res_vtk, success_flag = None, False
        vtk_dir = os.path.join(tmp_dir, 'VTK')
        for name in os.listdir(vtk_dir):
            if name.endswith('.vtk'):
                res_vtk = pyvista.read(os.path.join(vtk_dir, name))
                success_flag = True

        return {'state': success_flag, 'res_vtk': res_vtk}

    def run_robust_simulate_process(self, tmp_dir, orig_msh, orig_geo, with_debug=False):
        msh_file = os.path.join(tmp_dir, 'model.msh')
        if orig_msh != msh_file:
            shutil.copy(orig_msh, msh_file)

        geo_file = os.path.join(tmp_dir, 'model.geo')
        if orig_geo != geo_file:
            shutil.copy(orig_geo, geo_file)

        remesh_geo = os.path.join(tmp_dir, 'remesh.geo')
        ReMesher.save_remesh_msh_geo(msh_file, geo_file, remesh_geo)

        remesh_msh = os.path.join(tmp_dir, 'remesh.msh')
        ReMesher.geo2msh(tmp_dir, remesh_geo, remesh_msh, with_netgen_opt=True, msh_format='msh2')

        # ------ Step 2: create simulation configuration
        for object_name in self.openfoam_cfg['configuration'].keys():
            object_info = self.openfoam_cfg['configuration'][object_name]
            OpenFoamUtils.create_foam_file(
                tmp_dir, object_info['location'], object_info['class_name'],
                object_name=object_name, arg_dict=object_info['args']
            )

        # ------ Step 3: gmshToFoam
        OpenFoamUtils.exec_cmd(OpenFoamUtils.get_gmsh2foam_code(tmp_dir, remesh_msh))

        # ------ Step 4: modify type dict
        OpenFoamUtils.modify_boundary_type(tmp_dir, modify_dict=self.openfoam_cfg['modify_type_dict'])

        # ------ Step 5: scale unit
        if self.openfoam_cfg['unit_scale'] is not None:
            OpenFoamUtils.exec_cmd(OpenFoamUtils.get_unit_scale_code(tmp_dir, scale=self.openfoam_cfg['unit_scale']))

        # ------ Step 6: run simulation
        if with_debug:
            tick_0 = time.time()

        if self.run_parallel:
            res = OpenFoamUtils.exec_cmd(OpenFoamUtils.get_simulation_parallel_code(
                tmp_dir, num_of_process=self.num_of_threats,
                remove_conda_env=self.remove_conda_env, conda_sh=self.conda_sh
            ))
            assert res.returncode == 0
        else:
            OpenFoamUtils.exec_cmd(OpenFoamUtils.get_simulation_code(tmp_dir, self.run_code))

        if with_debug:
            cost_time = time.time() - tick_0
            print(f"[DEBUG StateSystem]: cost_time:{cost_time:.2f}")

        # ------ Step 7: convert result to VTK
        OpenFoamUtils.exec_cmd(OpenFoamUtils.get_foam2vtk_code(tmp_dir))

        # ------ Step 8: return vtk result
        res_vtk, success_flag = None, False
        vtk_dir = os.path.join(tmp_dir, 'VTK')
        for name in os.listdir(vtk_dir):
            if name.endswith('.vtk'):
                res_vtk = pyvista.read(os.path.join(vtk_dir, name))
                success_flag = True

        return {'state': success_flag, 'res_vtk': res_vtk}

    @staticmethod
    def compute_k_epsilon(U, L, viscosity):
        l = L * 0.07
        Re = np.abs(U) * L / viscosity
        # I = 0.16 * np.power(Re, -1. / 8.)
        I = 0.05
        k = 3. / 2. * np.power(U * I, 2)
        C_u = 0.09
        epsilon = np.power(C_u, 0.75) * np.power(k, 1.5) / l
        return k, epsilon, Re
