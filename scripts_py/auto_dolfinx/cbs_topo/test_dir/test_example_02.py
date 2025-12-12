import os
import shutil
from typing import Literal, Union, Callable

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import CbsMeshObj
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import TopoFluidNode, TopoCbsNode, TopoCbsSolver


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t35/'


def prepare_nodes():
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
            'name': 'inlet1',
            'boundary_type': 'velocity',
            'value': [1.0, 0.0, 0.0],
            'marker': 32,
            'face_center': np.array([-0.05, 0.4, 0.4]),
            'face_norm_vector': np.array([1.0, 0.0, 0.0]),
            'face_radius': 0.1,
        },
        {
            'name': 'outlet1',
            'boundary_type': 'pressure',
            'value': 0.0,
            'marker': 33
        },
        {
            'name': 'inlet2',
            'boundary_type': 'velocity',
            'value': [0.0, 1.0, 0.0],
            'marker': 34,
            'face_center': np.array([0.4, -0.05, 0.4]),
            'face_norm_vector': np.array([0.0, 1.0, 0.0]),
            'face_radius': 0.1,
        },
        {
            'name': 'outlet2',
            'boundary_type': 'pressure',
            'value': 0.0,
            'marker': 35
        },
        {
            'name': 'inlet3',
            'boundary_type': 'velocity',
            'value': [0.0, 0.0, 1.0],
            'marker': 36,
            'face_center': np.array([0.4, 0.4, -0.05]),
            'face_norm_vector': np.array([0.0, 0.0, 1.0]),
            'face_radius': 0.1,
        },
        {
            'name': 'outlet3',
            'boundary_type': 'pressure',
            'value': 0.0,
            'marker': 37
        },
        {
            'name': 'wall',
            'boundary_type': 'wall',
            'value': 0.0,
            'marker': 38,
        }
    ]
    
    MeshUtils.msh_to_XDMF(
        msh_file=os.path.join(base_dir, 't35_model.msh'), xdmf_file=os.path.join(base_dir, 't35_model.xdmf'), dim=3
    )
    
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 't35_model.msh'), xdmf_file=os.path.join(base_dir, 't35_model.xdmf')
    )
    mesh.create_neighbors_info()
    
    # ------ define fluid node 0
    fluid_node_0 = TopoFluidNode(mesh=mesh, name='group_a', save_dir=fluid_node_dir, node_idx=0)
    
    v_bry = define_boundary(boundary_cfgs[0], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[1], boundary_type='pressure')
    fluid_node_0.inlet_brys[v_bry.name] = v_bry
    fluid_node_0.outlet_brys[p_bry.name] = p_bry
    
    w_bry_1 = define_boundary(boundary_cfgs[2], boundary_type='wall')
    w_bry_2 = define_boundary(boundary_cfgs[3], boundary_type='wall')
    w_bry_3 = define_boundary(boundary_cfgs[4], boundary_type='wall')
    w_bry_4 = define_boundary(boundary_cfgs[5], boundary_type='wall')
    w_bry_5 = define_boundary(boundary_cfgs[6], boundary_type='wall')
    fluid_node_0.wall_brys[w_bry_1.name] = w_bry_1
    fluid_node_0.wall_brys[w_bry_2.name] = w_bry_2
    fluid_node_0.wall_brys[w_bry_3.name] = w_bry_3
    fluid_node_0.wall_brys[w_bry_4.name] = w_bry_4
    fluid_node_0.wall_brys[w_bry_5.name] = w_bry_5
    
    fluid_node_0.object_cost = AutoFluidCost(name='obj_cost1', style='MinDissipation', weight=1.0)
    fluid_node_0.regularization_cost = AutoFluidRegularization(
        name='reg_cost1', style='Volume', weight=100.0, target_ratio=0.07
    )
    fluid_node_0.solve()
    fluid_node_0.save()
    
    # ------ define fluid node 1
    fluid_node_1 = TopoFluidNode(mesh=mesh, name='group_b', save_dir=fluid_node_dir, node_idx=1)
    
    v_bry = define_boundary(boundary_cfgs[2], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[3], boundary_type='pressure')
    fluid_node_1.inlet_brys[v_bry.name] = v_bry
    fluid_node_1.outlet_brys[p_bry.name] = p_bry
    
    w_bry_1 = define_boundary(boundary_cfgs[0], boundary_type='wall')
    w_bry_2 = define_boundary(boundary_cfgs[1], boundary_type='wall')
    w_bry_3 = define_boundary(boundary_cfgs[4], boundary_type='wall')
    w_bry_4 = define_boundary(boundary_cfgs[5], boundary_type='wall')
    w_bry_5 = define_boundary(boundary_cfgs[6], boundary_type='wall')
    fluid_node_1.wall_brys[w_bry_1.name] = w_bry_1
    fluid_node_1.wall_brys[w_bry_2.name] = w_bry_2
    fluid_node_1.wall_brys[w_bry_3.name] = w_bry_3
    fluid_node_1.wall_brys[w_bry_4.name] = w_bry_4
    fluid_node_1.wall_brys[w_bry_5.name] = w_bry_5
    
    fluid_node_1.object_cost = AutoFluidCost(name='obj_cost2', style='MinDissipation', weight=1.0)
    fluid_node_1.regularization_cost = AutoFluidRegularization(
        name='reg_cost2', style='Volume', weight=100.0, target_ratio=0.07
    )
    fluid_node_1.solve()
    fluid_node_1.save()


def prepare_cbs_node():
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 't35_model.msh'), xdmf_file=os.path.join(base_dir, 't35_model.xdmf')
    )
    mesh.create_neighbors_info()
    
    fluid_node_0 = TopoFluidNode(mesh=mesh, name='group_a', save_dir=fluid_node_dir, node_idx=0)
    fluid_node_0.load()
    
    fluid_node_1 = TopoFluidNode(mesh=mesh, name='group_b', save_dir=fluid_node_dir, node_idx=1)
    fluid_node_1.load()
    
    # # ------ visualize fluid nodes body ------
    # for fluid_node in [fluid_node_0, fluid_node_1]:
    #     plotter = pyvista.Plotter()
    #     plotter.add_mesh(pyvista.PointSet(mesh.cell_centers[fluid_node.body == 1.0]), color='red', point_size=5.0)
    #     plotter.add_mesh(mesh.grid, style='wireframe')
    #     plotter.show()
    # # ------ visualize fluid nodes body ------
    
    cbs_node = TopoCbsNode(mesh=mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.compute_sort_value()
    cbs_node.detect_conflict()
    cbs_node.save()
    
    # ------ visualize conflict items ------
    for conflict_item in cbs_node.conflict_items:
        node_i = cbs_node.fluid_nodes[conflict_item.name_i]
        node_j = cbs_node.fluid_nodes[conflict_item.name_j]
        conflict_item.visualize(mesh, node_i, node_j)
    # ------ visualize conflict items ------


def run_cbs_solver():
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 't35_model.msh'), xdmf_file=os.path.join(base_dir, 't35_model.xdmf')
    )
    mesh.create_neighbors_info()
    
    fluid_node_0 = TopoFluidNode(mesh=mesh, name='group_a', save_dir=fluid_node_dir, node_idx=0)
    fluid_node_0.load()
    
    fluid_node_1 = TopoFluidNode(mesh=mesh, name='group_b', save_dir=fluid_node_dir, node_idx=1)
    fluid_node_1.load()
    
    cbs_node = TopoCbsNode(mesh=mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.compute_sort_value()
    cbs_node.detect_conflict()
    cbs_node.save()
    
    cbs_solver = TopoCbsSolver()
    cbs_solver.fluid_node_count = len(cbs_node.fluid_nodes)
    cbs_solver.cbs_node_count = len(cbs_solver.cbs_nodes)
    cbs_solver.cbs_nodes[cbs_node.name] = cbs_node
    cbs_solver.solve()


if __name__ == '__main__':
    # prepare_nodes()
    # prepare_cbs_node()
    run_cbs_solver()
    
    pass