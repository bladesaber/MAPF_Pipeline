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

from scripts_py.version_9.mapf_pkg.cad_utils import OcctUtil, OcctProfile, OcctAux


class CadServer:
    @staticmethod
    def generate_shape_json(name: str, shape, export_edges=False, mesh_quality=1.0):
        tess = ShapeTesselator(shape)
        tess.Compute(compute_edges=export_edges, mesh_quality=mesh_quality, parallel=True)
        return tess.ExportShapeToThreejsJSONString(name)

    @staticmethod
    def make_circle_thruSections(
            positions_xyz: np.ndarray, radius_list: np.ndarray, tangent_vecs: np.ndarray, ruled=True
    ):
        wire_list: List[TopoDS.TopoDS_Wire] = []
        for xyz, vec, radius in zip(positions_xyz, tangent_vecs, radius_list):
            circle = OcctProfile.make_circle(xyz, vec, radius)
            circle_wire = OcctUtil.make_wire(OcctUtil.make_edge(circle))
            wire_list.append(circle_wire)
        entity: TopoDS.TopoDS_Shell = OcctProfile.make_thruSections(
            wire_list, False, ruled
        )
        return entity

    @staticmethod
    def make_spline_surface(boundaries_info: List[Dict], guide_lines_info: Dict):
        boundaries = []
        for boundary_info in boundaries_info:
            if boundary_info['type'] == 'bspline':
                edge: TopoDS.TopoDS_Edge = OcctUtil.make_edge(OcctProfile.create_bspline(
                    np.array(boundary_info['control_pcd']), boundary_info['degree'],
                    left_clamp=True, right_clamp=True
                ))
                boundaries.append((edge, None))

        guide_lines = []
        for guide_info in guide_lines_info:
            edge: TopoDS.TopoDS_Edge = OcctUtil.make_edge(OcctProfile.create_bspline(
                np.array(guide_info['control_pcd']), guide_info['degree'],
                left_clamp=True, right_clamp=True
            ))
            guide_lines.append(edge)

        face: TopoDS.TopoDS_Face = OcctProfile.fill_surface_from_curve_network(boundaries, guide_lines)
        shell = OcctUtil.face2shell(face)
        return shell
