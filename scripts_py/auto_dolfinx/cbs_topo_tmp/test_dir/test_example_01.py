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
from scripts_py.auto_dolfinx.cbs_topo.utils import generate_box_mesh
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.cbs_topo.cbs_node import TopoFluidNode, TopoCbsNode, TopoCbsSolver


'''
由于无法解决三维Topology Optimization的算力问题, 导致无法进行任何三维的实验
'''


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
        'name': 'inlet_1',
        'boundary_type': 'velocity',
        'value': [1.0, 0.0, 0.0],
        'marker': 1,
        'face_center': np.array([0.0, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.15,
    },
    {
        'name': 'outlet_1',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 2,
        'face_center': np.array([1.0, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.15,
    },
    {
        'name': 'inlet_2',
        'boundary_type': 'velocity',
        'value': [0.0, 1.0, 0.0],
        'marker': 3,
        'face_center': np.array([0.5, 0.0, 0.5]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.15,
    },
    {
        'name': 'outlet_2',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 4,
        'face_center': np.array([0.5, 1.0, 0.5]),
        'face_norm_vector': np.array([0.0, 1.0, 0.0]),
        'face_radius': 0.15,
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 5,
    }
]


def generate_mesh():
    from mpi4py import MPI
    
    boundary_list = [
        define_boundary(boundary_cfgs[0], boundary_type='velocity'),
        define_boundary(boundary_cfgs[1], boundary_type='pressure'),
        define_boundary(boundary_cfgs[2], boundary_type='velocity'),
        define_boundary(boundary_cfgs[3], boundary_type='pressure'),
        define_boundary(boundary_cfgs[4], boundary_type='wall'),
    ]
    
    mesh = generate_box_mesh(
        boundary_list=boundary_list,
        left_bottom_corner=np.array([0.0, 0.0, 0.0]),
        right_top_corner=np.array([1.0, 1.0, 1.0]),
        num_cells=[10, 10, 10],
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf',
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        comm=MPI.COMM_WORLD,
    )
    # print(np.unique(mesh.facet_tags.values))
    

def debug_fluid_node():
    fluid_node_dir = '/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir'
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    mesh.create_neighbors_info()
    
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
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
        name='cost_1',
        style='MinDissipation',
        weight=1.0,
    )
    fluid_node_0.regularization_cost = AutoFluidRegularization(
        name='weight_1',
        style='Volume',
        weight=10.0,
        target_ratio=0.1
    )
    
    # fluid_node_0.save()
    # fluid_node_0.load()
    fluid_node_0.solve()


def debug_cbs_node():
    fluid_node_dir = '/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir'
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    cbs_node_fir = '/home/admin123456/Desktop/work/gmsh_test/t25/cbs_node_dir'
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    mesh.create_neighbors_info()
    
    # ------ define fluid node 0
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
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
    
    # ------ define fluid node 1
    fluid_node_1 = TopoFluidNode(
        mesh=mesh, 
        name='gb', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
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
    
    # ------ define cbs node 0
    cbs_node = TopoCbsNode(mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.save()
    cbs_node.clear()
    cbs_node.load()


def debug_detect_conflict():
    cbs_node_fir = '/home/admin123456/Desktop/work/gmsh_test/t25/cbs_node_dir'
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    mesh.create_neighbors_info()
    
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
        node_idx=0
    )
    fluid_node_0.load()
    
    fluid_node_1 = TopoFluidNode(
        mesh=mesh, 
        name='gb', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
        node_idx=1
    )
    fluid_node_1.load()
    
    cbs_node = TopoCbsNode(mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.compute_sort_value()
    cbs_node.save()
    print(f'[Log]: cbs node loss: {cbs_node.sort_value}')
    
    cbs_node.detect_conflict()
    print(f'Find {len(cbs_node.conflict_items)} conflicts.')
    # # ------ visualize obstacle ------
    # plotter = pyvista.Plotter()
    # for conflict in cbs_node.conflict_items:
    #     plotter.add_mesh(
    #         pyvista.PointSet(mesh.grid.points[conflict.conflict_idxs]), 
    #         color=np.random.randint(0, 255, size=(3, )),
    #         point_size=10.0
    #     )
    # plotter.add_mesh(
    #     pyvista.PointSet(mesh.grid.points[fluid_node_0.body == 1.0]), 
    #     color='red',
    #     point_size=5.0
    # )
    # plotter.add_mesh(
    #     pyvista.PointSet(mesh.grid.points[fluid_node_1.body == 1.0]), 
    #     color='green',
    #     point_size=5.0
    # )
    # plotter.add_mesh(mesh.grid, style='wireframe')
    # plotter.show()
    # # ------ visualize obstacle ------
    
    (new_cbs_nodes, fluid_node_count, cbs_node_count) = cbs_node.derivative_nodes(fluid_node_count=2, cbs_node_count=1)
    print(f'[Log]: fluid_node_count:{fluid_node_count} cbs_node_count:{cbs_node_count}.')


def debug_cbs_solver():
    cbs_node_fir = '/home/admin123456/Desktop/work/gmsh_test/t25/cbs_node_dir'
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    mesh.create_neighbors_info()
    
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
        node_idx=0
    )
    fluid_node_0.load()
    
    fluid_node_1 = TopoFluidNode(
        mesh=mesh, 
        name='gb', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
        node_idx=1
    )
    fluid_node_1.load()
    
    cbs_node = TopoCbsNode(mesh, save_dir=cbs_node_fir, node_idx=0)
    cbs_node.set_fluid_node(fluid_node_0)
    cbs_node.set_fluid_node(fluid_node_1)
    cbs_node.compute_sort_value()
    cbs_node.detect_conflict()
    cbs_node.save()
    
    print(f'Find {len(cbs_node.conflict_items)} conflicts.')
    
    cbs_solver = TopoCbsSolver()
    cbs_solver.fluid_node_count = 2
    cbs_solver.cbs_nodes[cbs_node.name] = cbs_node
    cbs_solver.solve()


def debug_run_all():
    fluid_node_dir = '/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir'
    if os.path.exists(fluid_node_dir):
        shutil.rmtree(fluid_node_dir)
    os.mkdir(fluid_node_dir)
    
    cbs_node_fir = '/home/admin123456/Desktop/work/gmsh_test/t25/cbs_node_dir'
    if os.path.exists(cbs_node_fir):
        shutil.rmtree(cbs_node_fir)
    os.mkdir(cbs_node_fir)
    
    mesh = SeqMeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    mesh.create_neighbors_info()
    
    # ------ define fluid node 0
    fluid_node_0 = TopoFluidNode(
        mesh=mesh, 
        name='ga', 
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
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
        save_dir='/home/admin123456/Desktop/work/gmsh_test/t25/fluid_node_dir', 
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
        
    cbs_solver = TopoCbsSolver()
    cbs_solver.fluid_node_count = 2
    cbs_solver.cbs_nodes[cbs_node.name] = cbs_node
    cbs_solver.solve()


if __name__ == '__main__':
    # generate_mesh()
    # debug_fluid_node()
    # debug_cbs_node()
    # debug_detect_conflict()
    # debug_cbs_solver()
    # debug_run_all()

    pass