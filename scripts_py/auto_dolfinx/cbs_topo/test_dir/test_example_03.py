import os
import shutil
from typing import Literal, Union, Callable

import dolfinx
import ufl
from ufl import inner, grad, div

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.cbs_topo.cbs_node import TopoFluidNode, TopoCbsNode, TopoCbsSolver


def define_boundary(cfg, boundary_type: Literal['velocity', 'pressure', 'wall']):
    return AutoFluidBoundary(
        name=cfg['name'],
        boundary_type=boundary_type,
        value=0.0 if boundary_type == 'wall' else cfg['value'],
        marker=cfg['marker'],
        face_center=None if boundary_type in ['wall', 'pressure'] else cfg['face_center'],
        face_norm_vector=None if boundary_type in ['wall', 'pressure'] else cfg['face_norm_vector'],
        face_radius=None if boundary_type in ['wall', 'pressure'] else cfg['face_radius'],
    )


boundary_cfgs = [
    {
        'name': 'inlet_1',
        'boundary_type': 'velocity',
        'value': [1.0, 0.0, 0.0],
        'marker': 26,
        'face_center': np.array([-0.05, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet_1',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 28,
    },
    {
        'name': 'inlet_2',
        'boundary_type': 'velocity',
        'value': [0.0, 1.0, 0.0],
        'marker': 27,
        'face_center': np.array([0.5, -0.05, 0.5]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet_2',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 29,
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 30,
    }
]


def generate_xdmf():
    MeshUtils.msh_to_XDMF(
        msh_file='/home/admin123456/Desktop/work/gmsh_test/t27/model.msh',
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t27/model.xdmf',
        dim=3,
    )


def debug_run_all():
    fluid_node_dir = '/home/admin123456/Desktop/work/gmsh_test/t27/fluid_node_dir'
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    cbs_node_fir = '/home/admin123456/Desktop/work/gmsh_test/t27/cbs_node_dir'
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t27/model.xdmf')
    mesh.create_neighbors_info()
    
    # ------ define fluid node 0
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t27/fluid_node_dir', 
        node_idx=0
    )
    
    v_bry = define_boundary(boundary_cfgs[0], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[1], boundary_type='pressure')
    fluid_node_0.inlet_brys[v_bry.name] = v_bry
    fluid_node_0.outlet_brys[p_bry.name] = p_bry
    
    w_bry_1 = define_boundary(boundary_cfgs[2], boundary_type='wall')
    w_bry_2 = define_boundary(boundary_cfgs[3], boundary_type='wall')
    w_bry_3 = define_boundary(boundary_cfgs[4], boundary_type='wall')
    fluid_node_0.wall_brys[w_bry_1.name] = w_bry_1
    fluid_node_0.wall_brys[w_bry_2.name] = w_bry_2
    fluid_node_0.wall_brys[w_bry_3.name] = w_bry_3
    
    fluid_node_0.object_cost = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    fluid_node_0.regularization_cost = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.1
    )
    fluid_node_0.solve()
    fluid_node_0.save()
    
    # ------ define fluid node 1
    fluid_node_1 = TopoFluidNode(
        mesh=mesh, 
        name='gb', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t27/fluid_node_dir', 
        node_idx=1
    )
    
    v_bry = define_boundary(boundary_cfgs[2], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[3], boundary_type='pressure')
    fluid_node_1.inlet_brys[v_bry.name] = v_bry
    fluid_node_1.outlet_brys[p_bry.name] = p_bry
    
    w_bry_1 = define_boundary(boundary_cfgs[0], boundary_type='wall')
    w_bry_2 = define_boundary(boundary_cfgs[1], boundary_type='wall')
    w_bry_3 = define_boundary(boundary_cfgs[4], boundary_type='wall')
    fluid_node_1.wall_brys[w_bry_1.name] = w_bry_1
    fluid_node_1.wall_brys[w_bry_2.name] = w_bry_2
    fluid_node_1.wall_brys[w_bry_3.name] = w_bry_3
    
    fluid_node_1.object_cost = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    fluid_node_1.regularization_cost = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.1
    )
    fluid_node_1.solve()
    fluid_node_1.save()
    
    cbs_node = TopoCbsNode(mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.save()
    
    cbs_node.compute_sort_value()
    cbs_node.detect_conflict()
    
    print(f'Find {len(cbs_node.conflict_items)} conflicts.')
    
    cbs_solver = TopoCbsSolver()
    cbs_solver.fluid_node_count = 2
    cbs_solver.cbs_nodes[cbs_node.name] = cbs_node
    cbs_solver.solve()


if __name__ == '__main__':
    # generate_xdmf()
    debug_run_all()
