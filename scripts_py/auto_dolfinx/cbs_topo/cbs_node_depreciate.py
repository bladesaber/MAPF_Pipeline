from ntpath import isfile
import os, shutil
from typing import Tuple
from functools import partial

import json
from meshio import dolfin
import numpy as np
from pandas._config.dates import cf
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.auto_dolfinx.cbs_topo.topo_fluid_opt import TopoObstacleFluidOptimizer


class TopoFluidNode(object):
    def __init__(self, mesh: SeqMeshObj, name: str, save_dir: str, node_idx: int):
        self.name = name
        self.node_idx = node_idx
        self.save_dir = save_dir
        self.fluid_dir = os.path.join(save_dir, f'{self.name}_{self.node_idx}')
        if not os.path.exists(self.fluid_dir):
            os.mkdir(self.fluid_dir)
        
        self.mesh: SeqMeshObj = mesh
        self.inlet_brys: dict[str, AutoFluidBoundary] = {}
        self.outlet_brys: dict[str, AutoFluidBoundary] = {}
        self.wall_brys: dict[str, AutoFluidBoundary] = {}
        self.object_cost: AutoFluidCost
        self.regularization_cost: AutoFluidRegularization
        
        self.obstacle_center: np.ndarray = np.array([])
        self.obstacle_radius: float = 1e-2
        self.obstacle_center_file = os.path.join(self.fluid_dir, 'obstacle_center.npy')
        
        self.body: np.ndarray = None  # type: ignore
        self.body_file = os.path.join(self.fluid_dir, 'result.npy')        
        
        self.is_clear = True
        self.loss: float
    
    def bry_to_dict(self, bry: AutoFluidBoundary):
        cfg = {
            'name': bry.name,
            'boundary_type': bry.boundary_type,
            'marker': bry.marker,
        }
        if isinstance(bry.value, np.ndarray):
            cfg.update({'value': bry.value.tolist()})
        else:
            cfg.update({'value': bry.value})
        
        if bry.face_center is not None:
            if isinstance(bry.face_center, np.ndarray):
                cfg.update({'face_center': bry.face_center.tolist()})
            else:
                cfg.update({'face_center': bry.face_center})
        if bry.face_norm_vector is not None:
            if isinstance(bry.face_norm_vector, np.ndarray):
                cfg.update({'face_norm_vector': bry.face_norm_vector.tolist()})
            else:
                cfg.update({'face_norm_vector': bry.face_norm_vector})
        if bry.face_radius is not None:
            cfg.update({'face_radius': bry.face_radius})
        
        return cfg

    def solve(self):
        define_obstacle_func = partial(
            TopoFluidNode.define_obstacle_func, 
            obstacle_centers=self.obstacle_center,
            obstacle_radius = self.obstacle_radius
        )
        
        record_dir = os.path.join(self.fluid_dir, 'progress')
        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)
        os.mkdir(record_dir)
        optim = TopoObstacleFluidOptimizer(
            name='topo_fluid_24',
            mesh=self.mesh,
            fluid_type='stoke',
            nut_init=1.0 / 100.,
            cost_item=self.object_cost,
            regularization_item=self.regularization_cost,
            topo_weight_init=self.regularization_cost.kwargs['target_ratio'] + 0.05,
            obstacle_func=define_obstacle_func,
            obstacle_cost_weight=100.0,
            weight_filter_radius=0.3,
            weight_filter_is_cell_field=False,
            result_dir=record_dir
        )
        
        tape = Tape()
        optim.setup_tape(tape)
        
        for _, inlet_bry in self.inlet_brys.items():
            optim.add_boundary(inlet_bry)
        for _, outlet_bry in self.outlet_brys.items():
            optim.add_boundary(outlet_bry)
        for _, wall_bry in self.wall_brys.items():
            optim.add_boundary(wall_bry)
        optim.define_problem()
        
        print(optim.eval())
                
        step_list = np.array([0.1, 0.05, 0.025])
        iter_list = np.array([15, 30, 45])
        for iter in range(iter_list[-1]):
            step_idx = int(np.sum(iter >= iter_list))
            detail_loss = optim.corase_step_run(step_list[step_idx], iter=iter)
            
            log_str = f'[Log step:{iter}/{step_list[step_idx]:.3f}]:'
            for name, cost_value in detail_loss.items():
                log_str += f" {name}:{cost_value:.5f}"
            print(log_str)
        
        self.loss = detail_loss['total_loss']
        topo_weight = optim.topo_weight.x.array.copy()
        self.body = (topo_weight > 0.5).astype(np.float32)
        np.save(self.body_file, self.body)

    def save(self):
        inlet_bry_dict = {}
        for name, bry in self.inlet_brys.items():
            inlet_bry_dict[name] = self.bry_to_dict(bry)
        
        outlet_bry_dict = {}
        for name, bry in self.outlet_brys.items():
            outlet_bry_dict[name] = self.bry_to_dict(bry)
        
        wall_bry_dict = {}
        for name, bry in self.wall_brys.items():
            wall_bry_dict[name] = self.bry_to_dict(bry)
        
        np.save(self.obstacle_center_file, self.obstacle_center)
        if self.body is not None:
            np.save(self.body_file, self.body)
        
        cfg = {
            'name': self.name,
            'wall_brys': wall_bry_dict,
            'inlet_brys': inlet_bry_dict,
            'outlet_brys': outlet_bry_dict,
            'object_cost': {
                'name': self.object_cost.name,
                'style': self.object_cost.style,
                'weight': self.object_cost.weight
            },
            'regularization_cost': {
                'name': self.regularization_cost.name,
                'style': self.regularization_cost.style,
                'weight': self.regularization_cost.weight,
                'kwargs': self.regularization_cost.kwargs
            },
            'obstacle_radius': self.obstacle_radius,
            'loss': self.loss
        }
        
        with open(os.path.join(self.fluid_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=4)
    
    def load(self):
        if not os.path.exists(self.fluid_dir):
            raise ValueError('[Error]: No Fluid dir.')
        with open(os.path.join(self.fluid_dir, 'config.json'), 'r') as f:
            cfg = json.load(f)
        assert self.name == cfg['name']
        
        self.obstacle_center = np.load(self.obstacle_center_file, allow_pickle=True)
        self.obstacle_radius = cfg['obstacle_radius']
        if os.path.exists(self.body_file):
            self.body = np.load(self.body_file, allow_pickle=True)
        
        for _, item in cfg['wall_brys'].items():
            self.wall_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker']
            )

        for _, item in cfg['inlet_brys'].items():
            self.inlet_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker'],
                face_center=item['face_center'],
                face_norm_vector=item['face_norm_vector'],
                face_radius=item['face_radius'],
            )
        
        for _, item in cfg['outlet_brys'].items():
            self.outlet_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker']
            )
        
        self.object_cost = AutoFluidCost(
            name=cfg['object_cost']['name'],
            style=cfg['object_cost']['style'],
            weight=cfg['object_cost']['weight'],
        )
        self.regularization_cost = AutoFluidRegularization(
            name=cfg['regularization_cost']['name'],
            style=cfg['regularization_cost']['style'],
            weight=cfg['regularization_cost']['weight'],
            **cfg['regularization_cost']['kwargs']
        )
        
        self.is_clear = False
        self.loss = cfg['loss']
    
    def clear(self, force_clear=False):
        if not self.is_clear or force_clear:
            self.inlet_brys.clear()
            self.outlet_brys.clear()
            self.wall_brys.clear()
            del self.obstacle_center
            del self.object_cost
            del self.regularization_cost
            del self.body
            self.is_clear = True
    
    @staticmethod
    def define_obstacle_func(xyzs: np.ndarray, obstacle_centers: np.ndarray, obstacle_radius: float):
        if obstacle_centers.shape[0] == 0:
            return np.array([], dtype=int)
        
        tree = KDTree(obstacle_centers)
        dists_list, idxs_list = tree.query(xyzs, k=1, return_distance=True)
        dists_list = np.concatenate(dists_list, axis=-1)        
        obs_dofs = np.where(dists_list < obstacle_radius)[0]
        return obs_dofs

    def copy(self, node_idx: int):        
        new_node = TopoFluidNode(self.mesh, self.name, self.save_dir, node_idx)
        for file_name in os.listdir(self.fluid_dir):
            if os.path.isfile(os.path.join(self.fluid_dir, file_name)):
                shutil.copy(
                    os.path.join(self.fluid_dir, file_name),
                    os.path.join(new_node.fluid_dir, file_name),
                )
            else:
                shutil.copytree(
                    os.path.join(self.fluid_dir, file_name),
                    os.path.join(new_node.fluid_dir, file_name),
                )
        return new_node

    def update_obstacle_center(self, obstacle_center: np.ndarray):
        self.obstacle_center = obstacle_center
        np.save(self.obstacle_center_file, obstacle_center)


class TopoConflict(object):
    def __init__(
        self, 
        conflict_idxs: np.ndarray,
        min_i_layer: int,
        name_i: str,
        min_j_layer: int,
        name_j: str
    ):
        self.conflict_idxs: np.ndarray = conflict_idxs
        self.min_i_layer = min_i_layer
        self.name_i = name_i
        self.min_j_layer = min_j_layer
        self.name_j = name_j


class TopoCbsNode(object):
    def __init__(self, mesh: SeqMeshObj, save_dir:str, node_idx: int):
        self.node_idx = node_idx
        self.save_dir = save_dir
        self.node_dir = os.path.join(self.save_dir, f'cbsNode_{self.node_idx}')
        if not os.path.exists(self.node_dir):
            os.mkdir(self.node_dir)
            
        self.name = f'cbsNode_{self.node_idx}'
        self.mesh: SeqMeshObj = mesh
        
        self.sort_value: float
        self.fluid_nodes: dict[str, TopoFluidNode] = {}
        self.conflict_items: list[TopoConflict] = []
    
    def set_fluid_node(self, fluid_node: TopoFluidNode):
        self.fluid_nodes[f'{fluid_node.name}_{fluid_node.node_idx}'] = fluid_node
    
    def replace_fluid_node(self, fluid_node: TopoFluidNode):
        replace_name = None
        for old_name, old_node in self.fluid_nodes.items():
            if old_node.name == fluid_node.name:
                replace_name = old_name
                break
        
        if replace_name is None:
            raise ValueError('[Error]: Can not replace')
        
        self.fluid_nodes.pop(replace_name)
        self.set_fluid_node(fluid_node)      
    
    def save(self):
        for _, node in self.fluid_nodes.items():
            node.save()
        
        cfg = {
            'fluid_nodes': list(self.fluid_nodes.keys()),
            'conflict_items': [],
        }
        for conflict_idx, conflict_item in enumerate(self.conflict_items):
            conflict_file = os.path.join(self.node_dir, f'conflict_{conflict_idx}.npy')
            np.save(conflict_file, conflict_item.conflict_idxs)
            cfg['conflict_items'].append({
                'conflict_idx': conflict_idx,
                'conflict_file': conflict_file,
                'name_i': conflict_item.name_i,
                'min_i_layer': conflict_item.min_i_layer,
                'name_j': conflict_item.name_j,
                'min_j_layer': conflict_item.min_j_layer,
            })
        
        with open(os.path.join(self.node_dir, f'{self.name}.json'), 'w') as f:
            json.dump(cfg, f, indent=4)
    
    def load(self):
        for _, node in self.fluid_nodes.items():
            node.load()
    
    def clear(self):
        for _, node in self.fluid_nodes.items():
            node.clear()
    
    def compute_sort_value(self):
        self.sort_value = 0.0
        for _, node in self.fluid_nodes.items():
            self.sort_value += node.loss
    
    def detect_conflict(self):
        self.load()
        
        fluid_node_names = list(self.fluid_nodes.keys())
        name_count = len(fluid_node_names)
        
        layer_tag_dict: dict[str, np.ndarray] = {}
        for i in range(name_count):
            name_i = fluid_node_names[i]
            fluid_node_i = self.fluid_nodes[name_i]
            source_points = np.array([bry.face_center for _, bry in fluid_node_i.inlet_brys.items()])
            self.mesh.create_seq_tag(
                source_points=source_points,
                point_thresholds=fluid_node_i.body
            )
            layer_tag_dict[name_i] = self.mesh.seq_dense_tag.copy()
        
        conflict_items = []
        for i in range(name_count - 1):
            name_i = fluid_node_names[i]
            fluid_node_i = self.fluid_nodes[name_i]
            body_i = fluid_node_i.body
            layer_tag_i = layer_tag_dict[name_i]
            self.mesh.seq_dense_tag = layer_tag_i
            
            for j in range(i + 1, name_count, 1):
                name_j = fluid_node_names[j]
                fluid_node_j = self.fluid_nodes[name_j]
                body_j = fluid_node_j.body
                layer_tag_j = layer_tag_dict[name_j]
                
                conflict_idxs = np.where(np.bitwise_and(body_i == 1.0, body_j == 1.0))[0]
                if conflict_idxs.shape[0] == 0:
                    continue
                
                conflict_segs_idxs = self.mesh.extract_all_clusters(conflict_idxs)
                for seg_idxs in conflict_segs_idxs:
                    min_tag_i = np.min(layer_tag_i[seg_idxs])
                    min_tag_j = np.min(layer_tag_j[seg_idxs])
                    
                    conflict_items.append(TopoConflict(
                        conflict_idxs=seg_idxs,
                        min_i_layer=min_tag_i,
                        name_i = name_i,
                        min_j_layer=min_tag_j,
                        name_j=name_j
                    ))
        
        self.conflict_items.clear()
        self.conflict_items = conflict_items
    
    def derivative_nodes(self, fluid_node_count: int, cbs_node_count: int) -> Tuple[list['TopoCbsNode'], int, int]:
        mini_topo_conflict: TopoConflict = None  # type: ignore
        mini_layer_tag = np.inf
        for conflict_item in self.conflict_items:
            min_tag = min(conflict_item.min_i_layer, conflict_item.min_j_layer)
            if min_tag < mini_layer_tag:
                mini_layer_tag = min_tag
                mini_topo_conflict = conflict_item
        
        conflict_zones = [
            (mini_topo_conflict.name_i, mini_topo_conflict.conflict_idxs),
            (mini_topo_conflict.name_j, mini_topo_conflict.conflict_idxs),
        ]
        
        new_cbs_nodes = []
        for old_fluid_node_name, new_obstacle_idxs in conflict_zones:
            old_fluid_node = self.fluid_nodes[old_fluid_node_name]
            new_fluid_node = old_fluid_node.copy(node_idx=fluid_node_count)
            fluid_node_count += 1
            
            if old_fluid_node.obstacle_center.shape[0] > 0:
                new_obs_centers = np.concatenate([
                    old_fluid_node.obstacle_center.copy(),
                    self.mesh.grid.points[new_obstacle_idxs]
                ], axis=0)
            else:
                new_obs_centers = self.mesh.grid.points[new_obstacle_idxs]
            new_fluid_node.update_obstacle_center(new_obs_centers)
            
            new_fluid_node.load()
            new_fluid_node.solve()
            
            # ------ generate new cbs node
            new_cbs_node = self.copy(cbs_node_count)
            cbs_node_count += 1
            
            new_cbs_node.replace_fluid_node(new_fluid_node)
            new_cbs_node.detect_conflict()
            new_cbs_node.compute_sort_value()
            print(f'[Log]: Find Node with conflict: {new_cbs_node.is_conflict()}.')
            new_cbs_node.save()
            new_cbs_nodes.append(new_cbs_node)
                    
        return (new_cbs_nodes, fluid_node_count, cbs_node_count)
    
    def is_conflict(self):
        return len(self.conflict_items) > 0
    
    def copy(self, node_idx) -> 'TopoCbsNode':
        new_node = TopoCbsNode(self.mesh, self.save_dir, node_idx)
        for name, node in self.fluid_nodes.items():
            new_node.fluid_nodes[name] = node
        return new_node

    def merge_result(self):
        raise NotImplementedError


class TopoCbsSolver(object):
    def __init__(self):
        self.cbs_nodes: dict[str, TopoCbsNode] = {}
        self.fluid_node_count = 0
    
    def pop_node(self) -> dict:
        if len(self.cbs_nodes) == 0:
            return {'state': False, 'msg': 'list is empty'}
        
        values = np.array([self.cbs_nodes[key].sort_value for key in self.cbs_nodes.keys()])
        min_idx = np.argmin(values)
        min_name = list(self.cbs_nodes.keys())[min_idx]
        min_node = self.cbs_nodes.pop(min_name)
        
        return {'state': True, 'node': min_node}
    
    def push_node(self, node: TopoCbsNode):
        self.cbs_nodes[node.name] = node
    
    def solve(self):
        head_dict = self.pop_node()
        if not head_dict['state']:
            print('[Log]: Quene is already empty.')
            return 
        cbs_node: TopoCbsNode = head_dict['node']
        
        while cbs_node.is_conflict():
            new_cbs_nodes, self.fluid_node_count, _ = cbs_node.derivative_nodes(self.fluid_node_count, len(self.cbs_nodes))
            for new_cbs_node in new_cbs_nodes:
                self.push_node(new_cbs_node)
                        
            head_dict = self.pop_node()
            if not head_dict['state']:
                return 
            cbs_node = head_dict['node']
        
        return cbs_node
    
    def clear(self):
        self.cbs_nodes.clear()


def generate_debug_mesh():
    from scripts_py.auto_dolfinx.cbs_topo.utils import generate_debug_mesh
    generate_debug_mesh()


def debug_conflict():
    from scripts_py.dolfinx_utils import DolfinUtils
    import dolfinx
    
    from scripts_py.auto_dolfinx.cbs_topo.utils import debug_boundary_list
    boundary_list = debug_boundary_list
    
    mesh = SeqMeshObj(
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf',
    )
    mesh.create_neighbors_info()
    
    # # ------ visualize surface ------
    # plotter = pyvista.Plotter()
    # for boundary in boundary_list:
    #     surface_indices = mesh.extract_surface_indices(boundary.marker)
    #     if boundary.boundary_type == "velocity":
    #         color = 'red'
    #     elif boundary.boundary_type == "pressure":
    #         color = 'blue'
    #     elif boundary.boundary_type == "wall":
    #         color = 'green'
    #     else:
    #         raise ValueError(f"Invalid boundary type: {boundary.boundary_type}")
    #     plotter.add_mesh(mesh.grid.points[surface_indices], color=color, point_size=10)
    # plotter.show()
    # # ------ visualize surface ------
    
    point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
    pos = DolfinUtils.tabulate_dof_coordinates(point_scalar_func_space)
    
    topo_weight_1 = dolfinx.fem.Function(point_scalar_func_space, name='topo_weight_1')
    topo_weight_1.x.array[:] = 1e-3  # type: ignore
    topo_weight_1 = topo_weight_1.x.array.copy()  # type: ignore
    path_1 = np.array([
        [0.0, 0.2, 0.2],
        [0.8, 0.2, 0.2],
        [0.8, 0.2, 0.8],
        [0.8, 1.0, 0.8]
    ])
    detail_path_1 = []
    for i in range(path_1.shape[0] - 1):
        step = int(np.linalg.norm(path_1[i + 1] - path_1[i]) / 0.05)
        detail_path_1.append(np.linspace(path_1[i], path_1[i + 1], step))
    detail_path_1 = np.concatenate(detail_path_1, axis=0)
    tree = KDTree(pos)
    path_1_idxs = np.unique(np.concatenate(tree.query_radius(detail_path_1, r=0.15, return_distance=False), axis=-1))
    topo_weight_1[path_1_idxs] = 0.999
    
    topo_weight_2 = dolfinx.fem.Function(point_scalar_func_space, name='topo_weight_2')
    topo_weight_2.x.array[:] = 1e-3  # type: ignore
    topo_weight_2 = topo_weight_2.x.array.copy()  # type: ignore
    path_2 = np.array([
        [0.5, 0.0, 0.2],
        [0.5, 0.5, 0.2],
        [0.8, 0.5, 0.2],
        [0.8, 0.5, 1.0]
    ])
    detail_path_2 = []
    for i in range(path_2.shape[0] - 1):
        step = int(np.linalg.norm(path_2[i + 1] - path_2[i]) / 0.05)
        detail_path_2.append(np.linspace(path_2[i], path_2[i + 1], step))
    detail_path_2 = np.concatenate(detail_path_2, axis=0)
    tree = KDTree(pos)
    path_2_idxs = np.unique(np.concatenate(tree.query_radius(detail_path_2, r=0.15, return_distance=False), axis=-1))
    topo_weight_2[path_2_idxs] = 0.999
    
    mesh.grid.point_data['topo_weight_1'] = topo_weight_1
    mesh.grid.point_data['topo_weight_2'] = topo_weight_2
    # mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t25/fake_result.vtu', binary=True)
    
    # ------ detect conflict ------
    body_1 = (topo_weight_1 > 0.5).astype(np.float32)
    body_2 = (topo_weight_2 > 0.5).astype(np.float32)
    conflict_pcd_idxs = np.where(np.bitwise_and(body_1 == 1.0, body_2 == 1.0))[0]
    
    # # ------ visualize conflict points ------
    # conflict_pcd = pos[conflict_pcd_idxs]
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(pyvista.PointSet(conflict_pcd), color='red', point_size=10.0)
    # plotter.add_mesh(mesh.grid, style='wireframe')
    # plotter.show()
    # # ------ visualize conflict points ------

    mesh.create_seq_tag(
        source_points=np.array([boundary_list[0].face_center]),
        point_thresholds=body_1
    )
    # # ------ visualize mesh seq_dense_tag ------
    # point_mesh = pyvista.PointSet(mesh.grid.points[mesh.seq_dense_tag > 0])
    # point_mesh.point_data['layer_tag'] = mesh.seq_dense_tag[mesh.seq_dense_tag > 0]
    # point_mesh.warp_by_scalar('layer_tag')
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(point_mesh, style='points')
    # plotter.show()
    # # ------ visualize mesh seq_dense_tag ------
    
    conflict_segments = mesh.extract_all_clusters(conflict_pcd_idxs)
    
    # # ------ visualize conflict segments ------
    # plotter = pyvista.Plotter()
    # for conflict_seg_idxs in conflict_segments:
    #     plotter.add_mesh(pyvista.PointSet(pos[conflict_seg_idxs]), color=np.random.randint(0, 255, size=(3, )), point_size=5.0)
    # plotter.add_mesh(mesh.grid, style='wireframe')
    # plotter.show()
    # # ------ visualize conflict segments ------
    
    print('For body 1:')
    for seg_idx, conflict_seg_idxs in enumerate(conflict_segments):
        layer_tags = mesh.seq_dense_tag[conflict_seg_idxs]
        print(f'[Body 1 Conflict {seg_idx}]: min:{np.min(layer_tags)} max:{np.max(layer_tags)}')
    
    mesh.create_seq_tag(
        source_points=np.array([boundary_list[2].face_center]),
        point_thresholds=body_2
    )
    print('For body 1:')
    for seg_idx, conflict_seg_idxs in enumerate(conflict_segments):
        layer_tags = mesh.seq_dense_tag[conflict_seg_idxs]
        print(f'[Body 2 Conflict {seg_idx}]: min:{np.min(layer_tags)} max:{np.max(layer_tags)}')


if __name__ == '__main__':
    # generate_debug_mesh()
    # debug_conflict()
    # debug_fluid_node()
    
    pass

