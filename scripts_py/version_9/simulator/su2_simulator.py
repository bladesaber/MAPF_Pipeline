import os
import numpy as np
import pandas as pd
import pyvista
from typing import Dict
import subprocess
import shutil


class Su2Simulator(object):
    def __init__(self, mesh_path: str):
        self.n_dim = None
        self.n_elem = None
        self.n_point = None
        self.n_mark = None
        self.cells = None
        self.points = None
        self.marker_dict = {}
        self.decode_su2(mesh_path)

    def decode_su2(self, mesh_path):
        with open(mesh_path, 'r') as f:
            state = 'None'
            cells, points = [], []

            for line in f.readlines():
                line = line.strip()
                if line.startswith('NDIME'):
                    self.n_dim = int(line.split('=')[1].strip())
                    continue
                elif line.startswith('NELEM'):
                    self.n_elem = int(line.split('=')[1].strip())
                    state = 'cell_state'
                    continue
                elif line.startswith('NPOIN'):
                    self.n_point = int(line.split('=')[1].strip())
                    state = 'point_state'
                    continue
                elif line.startswith('NMARK'):
                    self.n_mark = int(line.split('=')[1].strip())
                    continue
                elif line.startswith('MARKER_TAG'):
                    marker_tag = line.split('=')[1].strip()
                    self.marker_dict[marker_tag] = {'marker_cell': []}
                    state = f'{marker_tag}_marker'
                    continue
                elif line.startswith('MARKER_ELEMS'):
                    marker_tag = state.split('_')[0]
                    self.marker_dict[marker_tag]['n_elems'] = int(line.split('=')[1].strip())
                    continue

                if state == 'cell_state':
                    paras = line.split(' ')
                    if int(paras[0]) == 5:
                        cells.append([5, int(paras[1]), int(paras[2]), int(paras[3])])
                    elif int(paras[0]) == 10:
                        cells.append([10, int(paras[1]), int(paras[2]), int(paras[3]), int(paras[4])])
                    else:
                        raise ValueError("[Info]: Non-valid Cell.")
                elif state == 'point_state':
                    paras = line.split(' ')
                    if self.n_dim == 2:
                        points.append([float(paras[0]), float(paras[1])])
                    elif self.n_dim == 3:
                        points.append([float(paras[0]), float(paras[1]), float(paras[2])])
                elif state.endswith('_marker'):
                    marker_tag = state.split('_')[0]
                    paras = line.split(' ')
                    if int(paras[0]) == 3:
                        self.marker_dict[marker_tag]['marker_cell'].append([
                            3, int(paras[1]), int(paras[2])
                        ])
                    elif int(paras[0]) == 5:
                        self.marker_dict[marker_tag]['marker_cell'].append([
                            5, int(paras[1]), int(paras[2]), int(paras[3])
                        ])
                    else:
                        raise ValueError("[Info]: Non-valid Boundary Cell.")

            self.cells = np.array(cells, dtype=int)
            self.points = np.array(points, dtype=float)

    def encode_su2(self, mesh_path, with_idx=False):
        with open(mesh_path, 'w') as f:
            f.write(f'NDIME= {self.n_dim}\n')
            f.write(f'NELEM= {self.n_elem}\n')
            for idx, cell in enumerate(self.cells):
                if with_idx:
                    f.write(f"{' '.join([str(r) for r in cell])} {idx}\n")
                else:
                    f.write(f"{' '.join([str(r) for r in cell])}\n")
            f.write(f'NPOIN= {self.n_point}\n')
            for idx, point in enumerate(self.points):
                if with_idx:
                    f.write(f"{' '.join([str(r) for r in point])} {idx}\n")
                else:
                    f.write(f"{' '.join([str(r) for r in point])}\n")
            f.write(f'NMARK= {self.n_mark}\n')
            for tag, info in self.marker_dict.items():
                f.write(f'MARKER_TAG= {tag}\n')
                f.write(f"MARKER_ELEMS= {info['n_elems']}\n")
                for cell in info['marker_cell']:
                    f.write(f"{' '.join([str(r) for r in cell])}\n")

    def prepare_sim_proj(
            self,
            proj_dir: str, mesh_filename: str,
            inlets_info: Dict, outlets_info: Dict, walls_info: Dict,
            restart_filename='restart_flow.dat', restart_sol=False,
            density=877.0, init_vel=(1., 0., 0.), constant_viscosity=0.04,
            conv_field='RMS_PRESSURE', conv_level=-6, conv_filename='conv_history',
            run_iter=1000, body_force=(0., 0., 0.),
            with_screen_output=True, output_wrt_freq=100
    ):
        inlet_types, marker_inlet = [], []
        for name, info in inlets_info.items():
            inlet_types.append(info['type'])
            marker_inlet.extend([
                name, str(info['temperature']), str(info['vel_scale']),
                str(info['direction'][0]), str(info['direction'][1]), str(info['direction'][2])
            ])
        inlet_type_content = ', '.join(inlet_types)
        marker_inlet_content = ', '.join(marker_inlet)

        outlet_types, marker_outlet = [], []
        for name, info in outlets_info.items():
            outlet_types.append(info['type'])
            marker_outlet.extend([name, str(info['pressure'])])
        outlet_type_content = ', '.join(outlet_types)
        marker_outlet_content = ', '.join(marker_outlet)

        marker_wall = []
        for name, info in walls_info.items():
            marker_wall.extend([name, str(info['temperature'])])
        marker_wall_content = ', '.join(marker_wall)

        if restart_filename.endswith('.dat'):
            output_paras = ', '.join(['RESTART', 'PARAVIEW'])
            read_binary_restart = 'YES'
        elif restart_filename.endswith('.csv'):
            output_paras = ', '.join(['RESTART_ASCII', 'PARAVIEW'])
            read_binary_restart = 'NO'
        else:
            raise ValueError(f'[Error]: non-valid restart filename {restart_filename}.')

        if with_screen_output:
            screen_output = ', '.join(['INNER_ITER', 'WALL_TIME', 'RMS_PRESSURE'])
        else:
            screen_output = 'COARSE'

        cfg_content = f"""
% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
%
% Physical governing equations (EULER,NAVIER_STOKES,WAVE_EQUATION,HEAT_EQUATION,FEM_ELASTICITY,POISSON_EQUATION)
SOLVER= INC_RANS
% Mathematical problem (DIRECT, CONTINUOUS_ADJOINT)
MATH_PROBLEM= DIRECT
INC_NONDIM= DIMENSIONAL

KIND_TURB_MODEL= SST
SST_OPTIONS= V2003m
% SA_OPTIONS= WITHFT2 NEGATIVE EXPERIMENTAL

% KIND_TRANS_MODEL= LM  % need test
LM_OPTIONS= MENTER_LANGTRY

% FREESTREAM_TURBULENCEINTENSITY= 0.05
% FREESTREAM_TURB2LAMVISCRATIO= 10 

CFL_NUMBER= 200.0
CFL_REDUCTION_TURB= 1.0
% GAMMA_VALUE= 1.4
% AXISYMMETRIC= NO


% ------------------------- Time-dependent Simulation -------------------------------%
%
TIME_DOMAIN= NO
% TIME_STEP= 1.0
% MAX_TIME= 50.0 % Total Physical Time
% INNER_ITER= 200 % Number of internal iterations 
% TIME_ITER= 200 % Number of time steps
ITER= {run_iter}


% ----------- COMPRESSIBLE AND INCOMPRESSIBLE DEFINITION ----------%
%
FLUID_MODEL= CONSTANT_DENSITY
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {density}

BODY_FORCE= NO
BODY_FORCE_VECTOR= ({body_force[0]}, {body_force[1]}, {body_force[2]})

INC_ENERGY_EQUATION= YES
INC_TEMPERATURE_INIT= 300
INC_VELOCITY_INIT= ({init_vel[0]}, {init_vel[1]}, {init_vel[2]})

TURBULENT_CONDUCTIVITY_MODEL= CONSTANT_PRANDTL_TURB
PRANDTL_LAM= 0.72
PRANDTL_TURB= 0.9

% CONDUCTIVITY_MODEL= CONSTANT_CONDUCTIVITY
% THERMAL_CONDUCTIVITY_CONSTANT= 0.0257
FREESTREAM_DENSITY= 1.2
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 300.0
FREESTREAM_VELOCITY= (1.0, 0.0, 0.0)
VISCOSITY_MODEL= CONSTANT_VISCOSITY
THERMODYNAMIC_PRESSURE= 73528282.41
MU_CONSTANT= {constant_viscosity}
MU_POLYCOEFFS= (0.0, 0.0, 0.0, 0.0, 0.0)
MU_REF= 1.716e-05
MU_T_REF= 273.15

% --------------------------- CONVERGENCE PARAMETERS --------------------------%
%
CONV_FIELD= ({conv_field})
CONV_RESIDUAL_MINVAL= {conv_level}
CONV_STARTITER= 50
CONV_FILENAME= {conv_filename}
% HISTORY_WRT_FREQ_INNER= 1


% -------------------- TURBULENT NUMERICAL METHOD DEFINITION ------------------%
%
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO 
SLOPE_LIMITER_TURB= VENKATAKRISHNAN  % SLOPE LIMITERS
% SLOPE_LIMITER_TURB= NONE           % SLOPE LIMITERS
TIME_DISCRE_TURB= EULER_IMPLICIT


% -------------------- FLOW NUMERICAL METHOD DEFINITION -----------------------%
%
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE  % SLOPE LIMITERS
TIME_DISCRE_FLOW= EULER_IMPLICIT


% ----------------------- BOUNDARY CONDITION DEFINITION -----------------------%
%
% INC_INLET_DAMPING= 0.1
INC_INLET_USENORMAL= YES
INC_INLET_TYPE= ({inlet_type_content})
MARKER_INLET= ({marker_inlet_content})

% INC_OUTLET_DAMPING= 0.1
INC_OUTLET_TYPE= ({outlet_type_content})
MARKER_OUTLET= ({marker_outlet_content})

MARKER_ISOTHERMAL= ({marker_wall_content})


% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%
%
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_ERROR= 1e-10
LINEAR_SOLVER_ITER= 50
% LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_PREC= ILU


% ------------- COMMON PARAMETERS TO DEFINE GRADIENT --------------%
%
% NUM_METHOD_GRAD= GREEN_GAUSS
% NUM_METHOD_GRAD_RECON= GREEN_GAUSS
                 

% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%
%
MESH_FILENAME= {mesh_filename}
% MESH_FORMAT= CGNS
MESH_FORMAT= SU2

RESTART_FILENAME= {restart_filename}
READ_BINARY_RESTART= {read_binary_restart}
% WRT_RESTART_OVERWRITE= YES
RESTART_SOL= {'YES' if restart_sol else 'NO'} % Restart solution (NO, YES)

SOLUTION_FILENAME= {restart_filename} % filename that will be used to restart 
% RESTART_ITER= 1

% Output file surface flow coefficient (w/o extension)
SURFACE_FILENAME= surface_flow

% OUTPUT_FILES= (RESTART_ASCII, PARAVIEW_MULTIBLOCK)
OUTPUT_FILES= ({output_paras})
% OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_CSV)
OUTPUT_WRT_FREQ= ({output_wrt_freq})

VOLUME_FILENAME= flow
% VOLUME_OUTPUT= (COORDINATES, RESIDUAL, SOLUTION, PRIMITIVE)
VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)
% WRT_VOLUME_OVERWRITE= YES

SCREEN_OUTPUT= ({screen_output})
TABULAR_FORMAT= CSV
"""

        if os.path.exists(proj_dir):
            shutil.rmtree(proj_dir)
        os.mkdir(proj_dir)

        with open(os.path.join(proj_dir, 'simulate.cfg'), 'w') as f:
            f.write(cfg_content)
        self.encode_su2(os.path.join(proj_dir, mesh_filename))

    @staticmethod
    def run(run_dir: str, parallel_num=1, with_monitor=False) -> pyvista.UnstructuredGrid:
        cfg_file = os.path.join(run_dir, 'simulate.cfg')
        if parallel_num > 1:
            run_bash = f"cd {run_dir}; mpirun -n {parallel_num} SU2_CFD {cfg_file}"
        else:
            run_bash = f"cd {run_dir}; SU2_CFD {cfg_file}"
        if not with_monitor:
            run_bash += ' >> screen_output.log'
        subprocess.run(run_bash, shell=True, check=True)
        res = pyvista.read(os.path.join(run_dir, 'flow.vtu'))
        return res

    @staticmethod
    def msh_to_su2(msh_file: str, su2_file: str):
        assert msh_file.endswith('.msh') and su2_file.endswith('.su2')
        run_bash = f"gmsh {msh_file} -format su2 -o {su2_file} -save"
        subprocess.run(run_bash, shell=True, check=True)


def debug():
    # proj_dir = '/home/admin123456/Desktop/work/topopt_exps/new_test/example01/su2_cache'
    # simulator = Su2Simulator(
    #     mesh_path='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model.su2'
    # )
    # simulator.prepare_sim_proj(
    #     proj_dir=proj_dir,
    #     mesh_filename='model.su2',
    #     inlets_info={
    #         'inlet': {
    #             'type': 'VELOCITY_INLET',
    #             'temperature': 300.0,
    #             'vel_scale': 10.0,
    #             'direction': (1., 0., 0.)
    #         }
    #     },
    #     outlets_info={
    #         'outlet1': {
    #             'type': 'PRESSURE_OUTLET',
    #             'pressure': 0.0,
    #         },
    #         'outlet2': {
    #             'type': 'PRESSURE_OUTLET',
    #             'pressure': 0.0,
    #         }
    #     },
    #     walls_info={
    #         'wall': {
    #             'temperature': 300.0,
    #         },
    #     },
    #     conv_level=-8
    # )
    # simulator.run(proj_dir, with_monitor=True)

    # ---------------------------------------------------------------
    Su2Simulator.msh_to_su2(
        msh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model.msh',
        su2_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model_transform.su2'
    )

    proj_dir = '/home/admin123456/Desktop/work/topopt_exps/new_test/example01/su2_cache'
    simulator = Su2Simulator(
        mesh_path='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model_transform.su2'
    )
    simulator.prepare_sim_proj(
        proj_dir=proj_dir,
        mesh_filename='model_transform.su2',
        inlets_info={
            'inlet': {
                'type': 'VELOCITY_INLET',
                'temperature': 300.0,
                'vel_scale': 1.0,
                'direction': (1., 0., 0.)
            }
        },
        outlets_info={
            'outlet': {
                'type': 'PRESSURE_OUTLET',
                'pressure': 0.0,
            }
        },
        walls_info={
            'wall': {
                'temperature': 300.0,
            },
        },
        conv_level=-8
    )
    simulator.run(proj_dir, with_monitor=True)


if __name__ == '__main__':
    debug()
