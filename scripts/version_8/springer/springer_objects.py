import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from typing import Dict

from build import mapf_pipeline
from scripts.version_8.springer.springer_utils import ShapePcdUtils
from scripts.version_8.springer.springer_utils import VisulizerVista
from scripts.version_8.springer.springer_utils import get_rotateMat

'''
2023-09-05
由于在优化时，缺乏对互斥的双方做主从的划分，因此优化方向会类似受到斥力影响一样，优化会振荡进行，导致优化效率很低
'''

class EdgeUtils(object):
    @staticmethod
    def add_PoseEdge(edgeCfg, obj_info, ref_info, model):
        success = True
        if edgeCfg['type'] == 'Edge_structorToPlane_valueShift':
            success = model.addEdge_structorToPlane_valueShift(
                structorIdx=obj_info['nodeIdx'], planeIdx=ref_info['nodeIdx'],
                xyzTag=edgeCfg['xyzTag'], shiftValue=edgeCfg['shiftValue'],
                kSpring=edgeCfg['kSpring'], weight=edgeCfg['weight']
            )

        elif edgeCfg['type'] == 'Edge_connectorToStruct_valueShift':
            success = model.addEdge_connectorToStruct_valueShift(
                connectorIdx=obj_info['nodeIdx'], structorIdx=ref_info['nodeIdx'],
                xyzTag=edgeCfg['xyzTag'], shiftValue=edgeCfg['shiftValue'],
                kSpring=edgeCfg['kSpring'], weight=edgeCfg['weight']
            )

        elif edgeCfg['type'] == 'Edge_connectorToStruct_radiusFixed':
            success = model.addEdge_connectorToStruct_radiusFixed(
                connectorIdx=obj_info['nodeIdx'], structorIdx=ref_info['nodeIdx'],
                xyzTag=edgeCfg['xyzTag'], radius=edgeCfg['radius'],
                kSpring=edgeCfg['kSpring'], weight=edgeCfg['weight']
            )

        elif edgeCfg['type'] == 'Edge_connectorToStruct_poseFixed':
            success = model.addEdge_connectorToStruct_poseFixed(
                connectorIdx=obj_info['nodeIdx'], structorIdx=ref_info['nodeIdx'],
                shapeX=edgeCfg['shapeX'], shapeY=edgeCfg['shapeY'], shapeZ=edgeCfg['shapeZ'],
                kSpring=edgeCfg['kSpring'], weight=edgeCfg['weight']
            )

        else:
            raise ValueError

        return success

    @staticmethod
    def add_forceEdge(edgeCfg, obj_info, ref_info, model):
        success = True

        if edgeCfg['type'] == 'Edge_structorToPlane_planeRepel':
            warmUp_value = edgeCfg['warmUp_value']
            shape_pcd = obj_info['shapePcd_world']
            pcd_idxs = np.arange(0, shape_pcd.shape[0], 1)

            if edgeCfg['planeTag'] == 'X':
                if edgeCfg['compareTag'] == 'larger':
                    pcd_idxs = pcd_idxs[shape_pcd[:, 0] < ref_info['x'] + warmUp_value]
                else:
                    pcd_idxs = pcd_idxs[shape_pcd[:, 0] > ref_info['x'] - warmUp_value]

            elif edgeCfg['planeTag'] == 'Y':
                if edgeCfg['compareTag'] == 'larger':
                    pcd_idxs = pcd_idxs[shape_pcd[:, 1] < ref_info['y'] + warmUp_value]
                else:
                    pcd_idxs = pcd_idxs[shape_pcd[:, 1] > ref_info['y'] - warmUp_value]

            else:
                if edgeCfg['compareTag'] == 'larger':
                    pcd_idxs = pcd_idxs[shape_pcd[:, 2] < ref_info['z'] + warmUp_value]
                else:
                    pcd_idxs = pcd_idxs[shape_pcd[:, 2] > ref_info['z'] - warmUp_value]

            conflict_xyzs = []
            for idx in pcd_idxs:
                shapeX, shapeY, shapeZ = obj_info['shapePcd'][idx]
                conflict_xyzs.append((shapeX, shapeY, shapeZ))

            if len(conflict_xyzs) > 0:
                success = model.addEdge_structorToPlane_planeRepel(
                    structorIdx=obj_info['nodeIdx'], planeIdx=ref_info['nodeIdx'],
                    planeTag=edgeCfg['planeTag'], compareTag=edgeCfg['compareTag'],
                    conflict_xyzs=conflict_xyzs, bound_shift=0.0,
                    kSpring=edgeCfg['kSpring'], weight=edgeCfg['weight']
                )
                if not success:
                    return success

        elif edgeCfg['type'] == 'Edge_structorToStructor_shapeRepel':
            raise NotImplementedError

        return success

    @staticmethod
    def add_targetEdge(edgeCfg, nodeMap, name2NodeIdx, model):
        if edgeCfg['type'] == "minVolume":
            model.addEdge_minVolume(
                minPlaneIdx=name2NodeIdx[edgeCfg['planeMin_name']],
                maxPlaneIdx=name2NodeIdx[edgeCfg['planeMax_name']],
                scale=edgeCfg['scale'],
                kSpring=edgeCfg['kSpring'],
                weight=1.0
            )

        elif edgeCfg['type'] == "minAxes":
            model.addEdge_minAxes(
                minPlaneIdx=name2NodeIdx[edgeCfg['planeMin_name']],
                maxPlaneIdx=name2NodeIdx[edgeCfg['planeMax_name']],
                xyzTag=edgeCfg['xyzTag'],
                scale=edgeCfg['scale'],
                kSpring=edgeCfg['kSpring'],
                weight=1.0
            )

        elif edgeCfg['type'] == "poseCluster":
            node_info = nodeMap[name2NodeIdx[edgeCfg['obj_name']]]
            if node_info['type'] == 'structor':
                model.addEdge_structor_poseCluster(
                    structorIdx=node_info['nodeIdx'],
                    scale=edgeCfg['scale'],
                    kSpring=edgeCfg['kSpring'],
                    weight=1.0
                )
            elif node_info['type'] == 'connector':
                model.addEdge_connector_poseCluster(
                    connectorIdx=node_info['nodeIdx'],
                    scale=edgeCfg['scale'],
                    kSpring=edgeCfg['kSpring'],
                    weight=1.0
                )

class SpringTighter(object):
    def __init__(self, objs_info, targets_info):
        self.objs_info = objs_info
        self.targets_info = targets_info

        self.nodeMap = {}
        self.name2NodeIdx = {}

        self.model = mapf_pipeline.SpringerSmooth_Runner()
        self.model.initOptimizer(method="Levenberg")

        self.create_nodeInfos_from_cfgs(self.objs_info)
        self.create_graph_nodes()

    def create_nodeInfos_from_cfgs(self, objCfg_dict: Dict):
        for nodeIdx, obj_name in enumerate(objCfg_dict.keys()):
            obj_info = objCfg_dict[obj_name]
            obj_info.update({"nodeIdx": nodeIdx})
            self.nodeMap[nodeIdx] = obj_info
            self.name2NodeIdx[obj_name] = nodeIdx

    def update_structorShape(self, structor_info):
        pose = np.array([structor_info['x'], structor_info['y'], structor_info['z']])
        xyzTag, radian = structor_info['radian'], structor_info['xyzTag']

        shape_pcd = structor_info['shapePcd']
        shape_pcd_w = (get_rotateMat(xyzTag, radian).dot(shape_pcd.T)).T + pose

        structor_info['shapePcd_world'] = shape_pcd_w
        structor_info['shapeTree'] = KDTree(shape_pcd_w)

    def create_graph_nodes(self):
        for nodeIdx in self.nodeMap.keys():
            node_info = self.nodeMap[nodeIdx]

            if node_info['type'] in ['planeMin', 'planeMax']:
                state = self.model.add_Plane(
                    name=node_info['type'], nodeIdx=node_info['nodeIdx'], fixed=node_info['fixed'],
                    x=node_info['x'], y=node_info['y'], z=node_info['z']
                )
                if not state:
                    raise ValueError(f"[DEBUG]: Adding Node {node_info['type']} Fail")

            elif node_info['type'] == 'connector':
                state = self.model.add_Connector(
                    name=node_info['name'], nodeIdx=node_info['nodeIdx'], fixed=node_info['fixed'],
                    x=node_info['x'], y=node_info['y'], z=node_info['z']
                )
                if not state:
                    raise ValueError(f"[DEBUG]: Adding Node {node_info['name']} Fail")

            elif node_info['type'] == 'cell':
                state = self.model.add_Cell(
                    nodeIdx=node_info['nodeIdx'], fixed=node_info['fixed'],
                    x=node_info['x'], y=node_info['y'], z=node_info['z'], radius=node_info['radius']
                )
                if not state:
                    raise ValueError(f"[DEBUG]: Adding Node {node_info['name']} Fail")

            elif node_info['type'] == 'structor':
                state = self.model.add_Structor(
                    name=node_info['name'], nodeIdx=node_info['nodeIdx'], fixed=node_info['fixed'],
                    xyzTag=node_info['xyzTag'], x=node_info['x'], y=node_info['y'], z=node_info['z'],
                    radian=node_info['radian'], shell_radius=node_info['shell_radius']
                )
                self.update_structorShape(node_info)
                if not state:
                    raise ValueError(f"[DEBUG]: Adding Node {node_info['name']} Fail")

            else:
                raise ValueError

    def create_vertex_for_graph_node(self):
        state = self.model.add_vertexes()
        if not state:
            raise ValueError(f"[DEBUG]: Adding Vertexes Fail")

    def create_graph_edges(self):
        for nodeIdx in self.nodeMap.keys():
            node_info = self.nodeMap[nodeIdx]

            if node_info['type'] == 'structor':
                for edgeCfg in node_info['poseEdges']:
                    ref_info = self.nodeMap[self.name2NodeIdx[edgeCfg['ref_objName']]]
                    success = EdgeUtils.add_PoseEdge(edgeCfg, node_info, ref_info, self.model)
                    if not success:
                        raise ValueError(
                            f"[DEBUG]: Adding {edgeCfg['type']} between "
                            f"{node_info['name']} <-> {ref_info['name']} Fail"
                        )

                for edgeCfg in node_info['forceEdges']:
                    ref_info = self.nodeMap[self.name2NodeIdx[edgeCfg['ref_objName']]]
                    success = EdgeUtils.add_forceEdge(edgeCfg, node_info, ref_info, self.model)
                    if not success:
                        raise ValueError(
                            f"[DEBUG]: Adding {edgeCfg['type']} between "
                            f"{node_info['name']} <-> {ref_info['name']} Fail"
                        )

    def updateNode2Info(self):
        for nodeIdx in self.nodeMap.keys():
            node_info = self.nodeMap[nodeIdx]

            if node_info['type'] in ['planeMin', 'planeMax']:
                node = self.model.plane_NodeMap[nodeIdx]
                node_info['x'] = node.x
                node_info['y'] = node.y
                node_info['z'] = node.z

            elif node_info['type'] == 'connector':
                node = self.model.connector_NodeMap[nodeIdx]
                node_info['x'] = node.x
                node_info['y'] = node.y
                node_info['z'] = node.z

            elif node_info['type'] == 'cell':
                node = self.model.cell_NodeMap[nodeIdx]
                node_info['x'] = node.x
                node_info['y'] = node.y
                node_info['z'] = node.z

            elif node_info['type'] == 'structor':
                node = self.model.structor_NodeMap[nodeIdx]

                node_info['x'] = node.x
                node_info['y'] = node.y
                node_info['z'] = node.z
                node_info['radian'] = node.radian
                self.update_structorShape(node_info)

            else:
                raise ValueError

    def create_target_edges(self):
        for edgefg_name in self.targets_info.keys():
            edgefg = self.targets_info[edgefg_name]
            EdgeUtils.add_targetEdge(edgefg, self.nodeMap, self.name2NodeIdx, self.model)

    def optimize(self, outer_times, inner_times, verbose=False):
        self.model.clear_graph()

        for outer_i in range(outer_times):
            self.create_vertex_for_graph_node()

            self.create_graph_edges()
            self.create_target_edges()

            self.model.info()

            self.model.optimizeGraph(inner_times, verbose)

            self.model.update_nodeMapVertex()
            self.updateNode2Info()

            self.model.clear_graph()
            self.model.clear_vertexes()

            planeMin_node = self.nodeMap[self.name2NodeIdx['planeMin']]
            planeMax_node = self.nodeMap[self.name2NodeIdx['planeMax']]
            print(
                f"[DEBUG]: X:{planeMin_node['x']}-> {planeMax_node['x']} "
                f"Y:{planeMin_node['y']}->{planeMax_node['y']} "
                f"Z:{planeMin_node['z']}->{planeMax_node['z']}"
            )

        # self.plotEnv()

    def plotEnv(self):
        vis = VisulizerVista()

        min_plane_node = self.nodeMap[self.name2NodeIdx['planeMin']]
        max_plane_node = self.nodeMap[self.name2NodeIdx['planeMax']]
        xmin, ymin, zmin = min_plane_node['x'], min_plane_node['y'], min_plane_node['z']
        xmax, ymax, zmax = max_plane_node['x'], max_plane_node['y'], max_plane_node['z']
        # print(
        #     f"[DEBUG]: X:{xmin}-> {xmax} Y:{ymin}->{ymax} Z:{zmin}->{zmax} "
        #     f"Volume:{(xmax - xmin) * (ymax - ymin) * (zmax - zmin):.2f}"
        # )

        xmin_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("xmin", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(xmin_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        xmax_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("xmax", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(xmax_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        ymin_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("ymin", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(ymin_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        ymax_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("ymax", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(ymax_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        zmin_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("zmin", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(zmin_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        zmax_mesh = VisulizerVista.create_pointCloud(
            ShapePcdUtils.create_PlanePcd("zmax", xmin, ymin, zmin, xmax, ymax, zmax, 0.5)
        )
        vis.plot(zmax_mesh, opacity=0.65, color=(0.5, 0.5, 0.5))

        for nodeIdx in self.nodeMap.keys():
            node_info = self.nodeMap[nodeIdx]

            if node_info['type'] == 'structor':
                shapePcd_w = node_info['shapePcd_world']

                print(f"[DEBUG]: Pose:({node_info['x']:.2f}, {node_info['y']:.2f}, {node_info['z']:.2f})")
                print(f"[DEBUG]: shape_world: "
                      f"x:{shapePcd_w[:, 0].min():.2f}->{shapePcd_w[:, 0].max():.2f} "
                      f"y:{shapePcd_w[:, 1].min():.2f}->{shapePcd_w[:, 1].max():.2f} "
                      f"z:{shapePcd_w[:, 2].min():.2f}->{shapePcd_w[:, 2].max():.2f}"
                      )

                mesh = VisulizerVista.create_pointCloud(shapePcd_w)
                vis.plot(mesh, opacity=1.0, color=(1., 0., 0.))

        vis.addAxes(tip_length=0.25, tip_radius=0.05, shaft_radius=0.02)
        vis.show()

def main():
    structor0_shapePcd = ShapePcdUtils.create_BoxPcd(3, 3, 3, 5, 6, 10, reso=0.5)
    structor0_pose = np.array([4, 4.5, 6.5])
    structor0_shapePcd = structor0_shapePcd - structor0_pose

    objs_info = {
        'planeMin': {
            'type': 'planeMin', 'x': 0, 'y': 0, 'z': 0, 'fixed': True,
        },
        'planeMax': {
            'type': 'planeMax', 'x': 10, 'y': 10, 'z': 10, 'fixed': False,
        },
        'support_0': {
            'name': 'support', 'type': 'structor', 'fixed': False,
            'xyzTag': 'None', 'x': 3, 'y': 4.5, 'z': 6.5, 'radian': 0, 'shell_radius': 0.25,
            'shapePcd': structor0_shapePcd,
            'poseEdges': [
                {
                    'type': 'Edge_structorToPlane_valueShift',
                    'ref_objName': 'planeMax',
                    'xyzTag': 'Z',
                    'shiftValue': -3.5,
                    'kSpring': 5.0,
                    'weight': 1.0
                },
            ],
            'forceEdges': [
                # {
                #     'type': 'Edge_structorToPlane_planeRepel',
                #     'ref_objName': 'planeMin',
                #     'planeTag': 'X',
                #     'warmUp_value': 1.0,
                #     'compareTag': 'larger',
                #     'kSpring': 5.0,
                #     'weight': 1.0
                # },
                # {
                #     'type': 'Edge_structorToPlane_planeRepel',
                #     'ref_objName': 'planeMin',
                #     'planeTag': 'Y',
                #     'warmUp_value': 1.0,
                #     'compareTag': 'larger',
                #     'kSpring': 5.0,
                #     'weight': 1.0
                # },
                # {
                #     'type': 'Edge_structorToPlane_planeRepel',
                #     'ref_objName': 'planeMin',
                #     'planeTag': 'Z',
                #     'compareTag': 'larger',
                #     'warmUp_value': 1.0,
                #     'kSpring': 5.0,
                #     'weight': 1.0
                # },
                # {
                #     'type': 'Edge_structorToPlane_planeRepel',
                #     'ref_objName': 'planeMax',
                #     'planeTag': 'X',
                #     'compareTag': 'less',
                #     'warmUp_value': 1.0,
                #     'kSpring': 5.0,
                #     'weight': 1.0
                # },
                # {
                #     'type': 'Edge_structorToPlane_planeRepel',
                #     'ref_objName': 'planeMax',
                #     'planeTag': 'Y',
                #     'compareTag': 'less',
                #     'warmUp_value': 1.0,
                #     'kSpring': 5.0,
                #     'weight': 1.0
                # }
            ]
        },
    }

    targets_info = {
        'minVolume': {
            'type': 'minVolume',
            'planeMin_name': 'planeMin',
            'planeMax_name': 'planeMax',
            'scale': 0.1,
            'kSpring': 1.0,
            'weight': 1.0
        },
        # 'minAxes_X': {
        #     'type': 'minAxes',
        #     'planeMin_name': 'planeMin',
        #     'planeMax_name': 'planeMax',
        #     'xyzTag': "X",
        #     'scale': 0.1,
        #     'kSpring': 1.0,
        #     'weight': 1.0
        # },
        # 'minAxes_Y': {
        #     'type': 'minAxes',
        #     'planeMin_name': 'planeMin',
        #     'planeMax_name': 'planeMax',
        #     'xyzTag': "Y",
        #     'scale': 0.1,
        #     'kSpring': 1.0,
        #     'weight': 1.0
        # },
        # 'minAxes_Z': {
        #     'type': 'minAxes',
        #     'planeMin_name': 'planeMin',
        #     'planeMax_name': 'planeMax',
        #     'xyzTag': "Z",
        #     'scale': 0.1,
        #     'kSpring': 1.0,
        #     'weight': 1.0
        # },
    }

    smoother = SpringTighter(objs_info, targets_info)

    smoother.optimize(outer_times=1, inner_times=10, verbose=True)


if __name__ == '__main__':
    main()
