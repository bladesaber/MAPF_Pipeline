import os
import shutil
from typing import Literal, Union, Callable

import dolfinx
from mpi4py import MPI

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils

from scripts_py.auto_dolfinx.cbs_topo.utils import generate_box_mesh
from scripts_py.simulator.fipy_simulator import FipyStokeSimpleSimulator

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import CbsMeshObj
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import TopoFluidNode, TopoCbsNode, TopoCbsSolver


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
        'marker': 1,
        'face_center': np.array([0.0, 0.5, 0.4]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet1',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 2,
        'face_center': np.array([1.0, 0.5, 0.4]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'inlet2',
        'boundary_type': 'velocity',
        'value': [0.0, 1.0, 0.0],
        'marker': 3,
        'face_center': np.array([0.5, 0.0, 0.4]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet2',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 4,
        'face_center': np.array([0.5, 1.0, 0.4]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 38,
    }
]


def prepare_mesh():
    boundary_list = [define_boundary(bry_cfg, boundary_type=bry_cfg['boundary_type']) for bry_cfg in boundary_cfgs]
    mesh = generate_box_mesh(
        boundary_list=boundary_list, 
        left_bottom_corner=np.array([0.0, 0.0, 0.0]), 
        right_top_corner=np.array([1.0, 1.0, 0.8]), 
        num_cells=[30, 30, 25], 
        xdmf_file=os.path.join(base_dir, 'model.xdmf'), 
        cell_type=dolfinx.mesh.CellType.tetrahedron, comm=MPI.COMM_WORLD
    )
    print('Mesh. Info:')
    print(mesh.grid)
    
    face_tags = {}
    for boundary in boundary_list:
        face_tags[boundary.name] = boundary.marker
    
    MeshUtils.generate_msh22_file(
        mesh=mesh,
        filename=os.path.join(base_dir, 'model_v2.msh'),
        face_tags=face_tags,
        cell_tags={
            "volume": 9,
        }
    )
    FipyStokeSimpleSimulator.msh_version_convert(
        source_msh_file=os.path.join(base_dir, 'model_v2.msh'),
        target_msh_file=os.path.join(base_dir, 'model.msh'),
        target_version='msh4'
    )
    
    MeshUtils.msh_to_XDMF(
        msh_file=os.path.join(base_dir, 'model.msh'), xdmf_file=os.path.join(base_dir, 'model.xdmf'), dim=mesh.tdim
    )
    mesh = MeshObj(xdmf_file=os.path.join(base_dir, 'model.xdmf'))
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe')
    for boundary in boundary_list:
        idxs = mesh.extract_surface_indices(boundary.marker)
        if boundary.boundary_type == 'velocity':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='red', style='points')
        elif boundary.boundary_type == 'pressure':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='blue', style='points')
        elif boundary.boundary_type == 'wall':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='green', style='points')
    plotter.show()
    
    return mesh


def prepare_nodes():
    # TODO: 优化路径依然存在问题,优化过程很奇怪,而且优化道路不笔直,体积很大,要试下更小的target_ratio
    fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    cbs_node_fir = os.path.join(base_dir, 'cbs_node_dir')
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = CbsMeshObj(
        cell_mode=True, msh_file=os.path.join(base_dir, 'model.msh'), xdmf_file=os.path.join(base_dir, 'model.xdmf')
    )
    mesh.create_neighbors_info()
    
    # ------ define fluid node 0
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='group_a', 
        save_dir=fluid_node_dir, 
        node_idx=0,
        weight_filter_radius=0.06,
        topo_weight_init=1.0,
        simulation_iterations=400,
    )
        
    inlet_idx, outlet_idx = 0, 1
    v_bry = define_boundary(boundary_cfgs[inlet_idx], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[outlet_idx], boundary_type='pressure')
    fluid_node_0.inlet_brys[v_bry.name] = v_bry
    fluid_node_0.outlet_brys[p_bry.name] = p_bry
    
    for idx, bry_cfg in enumerate(boundary_cfgs):
        if idx == inlet_idx or idx == outlet_idx:
            continue
        w_bry = define_boundary(bry_cfg, boundary_type='wall')
        fluid_node_0.wall_brys[w_bry.name] = w_bry
    
    fluid_node_0.object_cost = AutoFluidCost(
        name='obj_cost', 
        style='MinDissipation', 
        weight=1.0
    )
    fluid_node_0.regularization_cost = AutoFluidRegularization(
        name='reg_cost', 
        style='Volume', 
        weight=10.0, 
        target_ratio=0.07
    )
    fluid_node_0.solve()
    fluid_node_0.save()
    
    # ------ define fluid node 1
    fluid_node_1 = TopoFluidNode(
        mesh=mesh, 
        name='group_b', 
        save_dir=fluid_node_dir, 
        node_idx=1,
        weight_filter_radius=0.06,
        topo_weight_init=1.0,
        simulation_iterations=400,
    )
    
    inlet_idx, outlet_idx = 2, 3
    v_bry = define_boundary(boundary_cfgs[inlet_idx], boundary_type='velocity')
    p_bry = define_boundary(boundary_cfgs[outlet_idx], boundary_type='pressure')
    fluid_node_1.inlet_brys[v_bry.name] = v_bry
    fluid_node_1.outlet_brys[p_bry.name] = p_bry
    
    for idx, bry_cfg in enumerate(boundary_cfgs):
        if idx == inlet_idx or idx == outlet_idx:
            continue
        w_bry = define_boundary(bry_cfg, boundary_type='wall')
        fluid_node_1.wall_brys[w_bry.name] = w_bry

    fluid_node_1.object_cost = AutoFluidCost(
        name='obj_cost', 
        style='MinDissipation', 
        weight=1.0
    )
    fluid_node_1.regularization_cost = AutoFluidRegularization(
        name='reg_cost', 
        style='Volume', 
        weight=10.0, 
        target_ratio=0.07
    )
    fluid_node_1.solve()
    fluid_node_1.save()


def prepare_cbs_node():
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


if __name__ == '__main__':
    # prepare_mesh()
    # prepare_nodes()
    # prepare_cbs_node()
    run_cbs_solver()
    
    pass
