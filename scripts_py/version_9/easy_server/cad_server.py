import json
import os
import sys
import tempfile
from typing import List, Dict, Union
import numpy as np

import OCC.Core.TopoDS as TopoDS
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.TopologyUtils import is_edge, is_wire, discretize_edge, discretize_wire
from OCC.Display.WebGl.simple_server import start_server

from scripts_py.version_9.mapf_pkg.cad_utils import OcctUtil, OcctProfile, OcctAux, OcctVis


class CadServer:
    @staticmethod
    def generate_shape_json(name: str, shape, export_edges=False, mesh_quality=1.0):
        tess = ShapeTesselator(shape)
        tess.Compute(compute_edges=export_edges, mesh_quality=mesh_quality, parallel=True)
        return tess.ExportShapeToThreejsJSONString(name)

    @staticmethod
    def make_arbitrary_edge(edge_info) -> TopoDS.TopoDS_Edge:
        if edge_info['type'] == 'bspline':
            edge: TopoDS.TopoDS_Edge = OcctUtil.make_edge(OcctProfile.create_bspline(
                np.array(edge_info['control_pcd']), edge_info['degree'],
                left_clamp=True, right_clamp=True
            ))
        elif edge_info['type'] == 'circle':
            edge: TopoDS.TopoDS_Edge = OcctUtil.make_edge(OcctProfile.make_circle(
                np.array(edge_info['center']),
                np.array(edge_info['tangent_vec']),
                edge_info['radius']
            ))
        elif edge_info['type'] == 'arc_circle':
            edge: TopoDS.TopoDS_Edge = OcctProfile.make_arc_circle(
                point=np.array(edge_info['center']),
                vector=np.array(edge_info['tangent_vec']),
                radius=edge_info['radius'],
                start_point=np.array(edge_info['start_point']),
                end_point=np.array(edge_info['end_point']),
            )
        else:
            raise ValueError("[ERROR]: Non-valid Edge config")
        return edge

    @staticmethod
    def make_arbitrary_wire(wire_info) -> TopoDS.TopoDS_Wire:
        if wire_info['type'] == 'circle':
            circle = CadServer.make_arbitrary_edge(wire_info)
            wire: TopoDS.TopoDS_Wire = OcctUtil.make_wire([circle])
        elif wire_info['type'] == 'arbitrary':
            edge_list = []
            for edge_info in wire_info['edge_cfgs']:
                edge_list.append(CadServer.make_arbitrary_edge(edge_info))
            wire: TopoDS.TopoDS_Wire = OcctUtil.make_wire(edge_list)
        else:
            raise ValueError("[ERROR]: Non-valid Wire config")
        return wire

    @staticmethod
    def make_arbitrary_thruSections(path_info: List[Dict], ruled=True):
        wire_list: List[TopoDS.TopoDS_Wire] = []
        for info in path_info:
            wire_list.append(CadServer.make_arbitrary_wire(info))
        entity: TopoDS.TopoDS_Shell = OcctProfile.make_thruSections(wire_list, False, ruled)
        return entity

    @staticmethod
    def make_spline_surface(boundaries_info: List[Dict], guide_info: List[Dict], sample_num: int):
        boundaries = []
        for info in boundaries_info:
            boundaries.append((CadServer.make_arbitrary_edge(info), None))

        guide_points = []
        guide_lines = []
        for info in guide_info:
            if info['type'] != 'points':
                guide_lines.append(CadServer.make_arbitrary_edge(info))
            else:
                guide_points.append(np.array(info['pcd']))

        if len(guide_points) > 0:
            guide_points = np.concatenate(guide_points, axis=0)
        else:
            guide_points = None

        face: TopoDS.TopoDS_Face = OcctProfile.fill_surface_from_curve_network(
            boundaries,
            guide_lines=guide_lines,
            guide_points=guide_points,
            sample_num=sample_num + 2
        )
        shell = OcctUtil.face2shell(face)
        return shell


def debug():
    boundaries_info = [
        {
            'type': 'bspline',
            'control_pcd': [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            'degree': 3
        },
        {
            'type': 'bspline',
            'control_pcd': [
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
            ],
            'degree': 3
        },
        {
            'type': 'bspline',
            'control_pcd': [
                [3.0, 3.0, 0.0],
                [2.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            'degree': 3
        },
        {
            'type': 'bspline',
            'control_pcd': [
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            'degree': 3
        }
    ]

    guide_info_1 = [
        {
            'type': 'points',
            'pcd': [
                [0.5, 0.5, 1.],
                [0.5, 2.5, 1.],
                [2.5, 2.5, 1.],
                [2.5, 0.5, 1.],

                [1.0, 1.0, 1.],
                [2.0, 2.0, 1.],
                [2.0, 1.0, 1.],
                [1.0, 2.0, 1.],

                [1.5, 1.5, 1.],
            ]
        }
    ]

    guide_info_2 = [
        {
            'type': 'bspline',
            'control_pcd': [
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.4],
                [0.4, 0.4, 0.8],
                [0.5, 0.5, 1.0],

                [0.6, 0.6, 1.0],
                [0.8, 0.8, 1.0],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.0],
                [2.0, 2.0, 1.0],

                [2.5, 2.5, 1.0],
                [2.6, 2.6, 0.8],
                [2.8, 2.8, 0.4],
                [3.0, 3.0, 0.0],
            ],
            'degree': 3
        },
        {
            'type': 'bspline',
            'control_pcd': [
                [0.0, 3.0, 0.0],
                [0.2, 2.8, 0.4],
                [0.4, 2.6, 0.8],
                [0.5, 2.5, 1.0],

                [0.6, 2.4, 1.0],
                [0.8, 2.2, 1.0],
                [1.0, 2.0, 1.0],
                [1.5, 1.5, 1.0],
                [2.0, 1.0, 1.0],

                [2.5, 0.5, 1.0],
                [2.6, 0.4, 0.8],
                [2.8, 0.2, 0.4],
                [3.0, 0.0, 0.0],
            ],
            'degree': 3
        },
    ]

    face = CadServer.make_spline_surface(boundaries_info, guide_info_2, sample_num=50)
    vis = OcctVis()
    vis.plot(face)

    for info in guide_info_2:
        for pnt_np in info['control_pcd']:
            pnt_cad = OcctProfile.make_point(pnt_np[0], pnt_np[1], pnt_np[2])
            vis.plot(pnt_cad, color='red')

    # for pnt_np in guide_info_1[0]['pcd']:
    #     pnt_cad = OcctProfile.make_point(pnt_np[0], pnt_np[1], pnt_np[2])
    #     vis.plot(pnt_cad, color='red')

    vis.show()


if __name__ == '__main__':
    debug()
