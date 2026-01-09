import os
import shutil
from typing import Literal, Union, Callable
import time

import dolfinx
from mpi4py import MPI

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.topo_fluid_node import TopoFluidNode, CbsMeshObj

"""
TODO:
1. 结果还是有不符合我们想象的地方,进出口处可能由于崩角原因,速度会在进出口处变得很大,导致进出口体积大,中间体积小.
2. 为什么结果存在崩塌,为什么崩塌后obj_cost反而更小
"""


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t36/'


def define_boundary(cfg, boundary_type: Literal['velocity', 'pressure', 'wall']):
    return AutoFluidBoundary(
        name=cfg['name'],
        boundary_type=boundary_type,
        value=0.0 if boundary_type == 'wall' else cfg['value'],
        marker=cfg['marker'],
        face_center=None if boundary_type == 'wall' else cfg['face_center'],
        face_norm_vector=None if boundary_type == 'wall' else cfg['face_norm_vector'],
        face_radius=None if boundary_type == 'wall' else cfg['face_radius'],
    )
        

boundary_cfgs = [
    {
        'name': 'inlet1',
        'boundary_type': 'velocity',
        'value': [1.0, 0.0, 0.0],
        'marker': 26,
        'face_center': np.array([-0.05, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
        'group': 'group_a',
    },
    {
        'name': 'outlet1',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 27,
        'face_center': np.array([1.05, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
        'group': 'group_a',
    },
    {
        'name': 'inlet2',
        'boundary_type': 'velocity',
        'value': [0.0, 1.0, 0.0],
        'marker': 28,
        'face_center': np.array([0.5, -0.05, 0.5]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.1,
        'group': 'group_b',
    },
    {
        'name': 'outlet2',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 29,
        'face_center': np.array([0.5, 1.05, 0.5]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.1,
        'group': 'group_b',
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 30,
        'group': 'all',
    }
]


def prepare_nodes():
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    mesh = CbsMeshObj(
        msh_file=os.path.join(base_dir, 'model.msh'), 
        dim=3,
        cell_mode=True, 
    )
    
    # ------ define fluid node
    for node_idx, group_name in enumerate(['group_a', 'group_b']):        
        node = TopoFluidNode(
            mesh=mesh, 
            name=group_name, 
            save_dir=fluid_node_dir, 
            node_idx=node_idx,
            weight_filter_radius=0.075,
            topo_weight_init=0.1,
            viscosity=1.0 / 400.,
            simulation_iterations=200,
            convergence_threshold=1e-4,
            velocityRelaxation=0.6,
            pressureRelaxation=0.8,
            log_step=2,
            hot_steps=10,
            parallel_num=8,
            use_tqdm=True,
            use_log=True,
        )
        
        for bry_cfg in boundary_cfgs:
            if bry_cfg['group'] == group_name:
                bry = define_boundary(bry_cfg, boundary_type=bry_cfg['boundary_type'])
                if bry_cfg['boundary_type'] == 'velocity':
                    node.inlet_brys[bry.name] = bry
                elif bry_cfg['boundary_type'] == 'pressure':
                    node.outlet_brys[bry.name] = bry
            else:
                bry = define_boundary(bry_cfg, boundary_type='wall')
                node.wall_brys[bry.name] = bry
        
        node.object_cost = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0)
        node.regularization_cost = AutoFluidRegularization(name='reg_cost', style='Volume', weight=1.0, target_ratio=0.2)

        start_time = time.time()
        node.solve()
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        node.save()
        
        break

"""
def prepare_cbs_node():
    # TODO: 由这里可以看出,topology optimization的标准grid化结果依然不足够好,表面光滑度不够,也不成凸集,可能需要更光滑的映射层
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 'model.msh'), xdmf_file=os.path.join(base_dir, 'model.xdmf')
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
    
    # # ------ visualize conflict items ------
    # for conflict_item in cbs_node.conflict_items:
    #     node_i = cbs_node.fluid_nodes[conflict_item.name_i]
    #     node_j = cbs_node.fluid_nodes[conflict_item.name_j]
    #     conflict_item.visualize(mesh, node_i, node_j)
    # # ------ visualize conflict items ------
    
    # ------ save for visualization ------
    grid = mesh.grid.copy()
    grid.cell_data[fluid_node_0.name] = fluid_node_0.body
    grid.cell_data[fluid_node_1.name] = fluid_node_1.body
    for conflict_idx, conflict_item in enumerate(cbs_node.conflict_items):
        obs_weight = np.zeros(grid.n_cells)
        obs_weight[conflict_item.conflict_idxs] = 1.0
        grid.cell_data[f'{conflict_item.name_i}_{conflict_item.name_j}_conflict_{conflict_idx}'] = obs_weight
    grid.save(os.path.join(base_dir, 'test.vtu'))


def run_cbs_solver():
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 'model.msh'), xdmf_file=os.path.join(base_dir, 'model.xdmf')
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
"""

if __name__ == '__main__':    
    prepare_nodes()
    # prepare_cbs_node()
    # run_cbs_solver()
    
    pass
