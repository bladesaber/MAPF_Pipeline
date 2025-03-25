import numpy as np
from typing import Dict, List, Union
import json
import h5py

import OCC.Core.TopoDS as TopoDS
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.TopologyUtils import is_edge, is_wire, discretize_edge, discretize_wire
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakeFilling
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2, gp_Circ, gp_Trsf, gp_Ax1, gp_OX, gp_OY, gp_OZ
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.GeomAbs import GeomAbs_C0


class CadServerUtils:
    @staticmethod
    def export_edgedata_to_json(edge_hash: str, point_set: List) -> str:
        """Export a set of points to a LineSegment buffergeometry"""
        points_coordinates = []
        for point in point_set:
            points_coordinates.extend(iter(point))
        
        edges_data = {
            "metadata": {
                "version": 4.4,
                "type": "BufferGeometry",
                "generator": "pythonocc",
            },
            "uuid": edge_hash,
            "type": "BufferGeometry",
            "data": {
                "attributes": {
                    "position": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": points_coordinates,
                    }
                }
            },
        }
        return edges_data
    
    @staticmethod
    def convert_shape_to_json(
            name: str,
            shape: Union[
                TopoDS.TopoDS_Shape, TopoDS.TopoDS_Compound, TopoDS.TopoDS_Edge, 
                TopoDS.TopoDS_Wire, TopoDS.TopoDS_Face, TopoDS.TopoDS_Shell, 
                TopoDS.TopoDS_Solid, TopoDS.TopoDS_CompSolid
            ],
            export_edges: bool = False,
            mesh_quality: float = 1.0
            ) -> Dict:
        """
        Convert an OpenCASCADE shape to Three.js compatible JSON format.
        
        Args:
            shape: The OpenCASCADE shape to convert
            export_edges: Whether to export edges
            mesh_quality: Quality of the mesh (default=1.0)
            
        Returns:
            Dict containing the shape data and edges data in Three.js compatible format
        """
        # Handle edges and wires specially
        if is_edge(shape):
            pnts = discretize_edge(shape)
            return {"edges": [CadServerUtils.export_edgedata_to_json(name, pnts)]}
            
        elif is_wire(shape):
            pnts = discretize_wire(shape)
            wire_data = CadServerUtils.export_edgedata_to_json(name, pnts)
            return {"edges": [wire_data]}
        else:
            # For other shapes, use tesselation
            tess = ShapeTesselator(shape)
            tess.Compute(compute_edges=export_edges, mesh_quality=mesh_quality, parallel=True)
            
            # Get the main shape data
            shape_data = {"shapes": [json.loads(tess.ExportShapeToThreejsJSONString(name))]}
            
            # Handle edges if requested
            if export_edges:
                edges_data = []
                nbr_edges = tess.ObjGetEdgeCount()
                
                for i_edge in range(nbr_edges):
                    nbr_vertices = tess.ObjEdgeGetVertexCount(i_edge)
                    edge_point_set = [
                        tess.GetEdgeVertex(i_edge, i_vert) 
                        for i_vert in range(nbr_vertices)
                    ]
                    edges_data.append(CadServerUtils.export_edgedata_to_json(f"{name}_edge_{i_edge}", edge_point_set))
                
                shape_data["edges"] = edges_data
                
            return shape_data

    @staticmethod
    def read_step(step_file: str):
        reader = STEPControl_Reader()
        reader.Read(step_file)
        reader.TransferRoot()
        shape = reader.Shape()
        return shape

    @staticmethod
    def write_step(shape: TopoDS.TopoDS_Shape, step_file: str):
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(step_file)

    @staticmethod
    def create_bspline(
            control_pcd: np.ndarray, degree: int, left_clamp: bool, right_clamp: bool,
    ) -> Geom_BSplineCurve:
        n_control = control_pcd.shape[0]
        pnt_array = TColgp_Array1OfPnt(1, n_control)
        for i, point in enumerate(control_pcd):
            pnt_array.SetValue(i + 1, gp_Pnt(float(point[0]), float(point[1]), float(point[2])))

        if left_clamp and right_clamp:
            n_knot = (n_control + degree + 1) - 2 * (degree + 1) + 2
        elif left_clamp and (not right_clamp):
            n_knot = (n_control + degree + 1) - 1 * (degree + 1) + 1
        elif (not left_clamp) and right_clamp:
            n_knot = (n_control + degree + 1) - 1 * (degree + 1) + 1
        else:
            n_knot = n_control + degree + 1
        knot = np.linspace(0.0, 1.0, n_knot)
        knot_array = TColStd_Array1OfReal(1, n_knot)
        for i, k_value in enumerate(knot):
            # print(f"[INFO]: Setting knot index:{i + 1} k_value:{k_value}")
            knot_array.SetValue(i + 1, k_value)

        if left_clamp and right_clamp:
            mul_value = np.ones(shape=(n_knot,), dtype=int)
            mul_value[0] = degree + 1
            mul_value[-1] = degree + 1
        elif left_clamp and (not right_clamp):
            mul_value = np.ones(shape=(n_knot,), dtype=int)
            mul_value[0] = degree + 1
        elif (not left_clamp) and right_clamp:
            mul_value = np.ones(shape=(n_knot,), dtype=int)
            mul_value[-1] = degree + 1
        else:
            mul_value = np.ones(shape=(n_knot,), dtype=int)

        multiplicities_array = TColStd_Array1OfInteger(1, n_knot)
        for i, v in enumerate(mul_value):
            multiplicities_array.SetValue(i + 1, int(v))

        profile = Geom_BSplineCurve(pnt_array, knot_array, multiplicities_array, degree, False)

        return profile

    @staticmethod
    def make_OCC_edge(info: Dict) -> TopoDS.TopoDS_Edge:
        if info['style'] == 'bspline':
            points_array = np.array(info['control_points'])
            curve = CadServerUtils.create_bspline(
                control_pcd=points_array,
                degree=info['degree'],
                left_clamp=info['left_clamp'],
                right_clamp=info['right_clamp']
            )
            return BRepBuilderAPI_MakeEdge(curve).Edge()
            
        elif info['style'] == 'circle':
            center = gp_Pnt(info['center'][0], info['center'][1], info['center'][2])
            normal = gp_Dir(info['normal'][0], info['normal'][1], info['normal'][2])
            axis = gp_Ax2(center, normal)
            circle = gp_Circ(axis, info['radius'])
            return BRepBuilderAPI_MakeEdge(circle).Edge()
            
        elif info['style'] == 'arc_circle':
            center = gp_Pnt(info['center'][0], info['center'][1], info['center'][2])
            normal = gp_Dir(info['normal'][0], info['normal'][1], info['normal'][2])
            axis = gp_Ax2(center, normal)
            circle = gp_Circ(axis, info['radius'])
            start_point = gp_Pnt(info['start_point'][0], info['start_point'][1], info['start_point'][2])
            end_point = gp_Pnt(info['end_point'][0], info['end_point'][1], info['end_point'][2])
            return BRepBuilderAPI_MakeEdge(circle, start_point, end_point).Edge()
            
        elif info['style'] == 'multi_lines':
            wire_builder = BRepBuilderAPI_MakeWire()
            points = info['points']
            for i in range(len(points) - 1):
                p1 = gp_Pnt(points[i][0], points[i][1], points[i][2])
                p2 = gp_Pnt(points[i+1][0], points[i+1][1], points[i+1][2])
                edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                wire_builder.Add(edge)

            if len(points) == 2:
                return wire_builder.Edge()
            else:
                return wire_builder.Wire()  # Note: This actually returns a wire

        elif info['style'] == 'poly_line':
            points = info['points']
            pnt_array = TColgp_Array1OfPnt(1, len(points))
            for i, pt in enumerate(points):
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                pnt_array.SetValue(i+1, gp_Pnt(x, y, z))
            curve = GeomAPI_PointsToBSpline(pnt_array).Curve()
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
            return edge

        else:
            raise ValueError(f"Unknown edge style: {info['style']}")

    @staticmethod
    def make_OCC_wire(info: Dict) -> TopoDS.TopoDS_Wire:
        edge = CadServerUtils.make_OCC_edge(info)
        if is_edge(edge):
            return BRepBuilderAPI_MakeWire(edge).Wire()
        else:
            return edge

    @staticmethod
    def make_OCC_thruSections(path_info: List[Dict], ruled=True, isSolid=False) -> TopoDS.TopoDS_Shell:
        wire_list: List[TopoDS.TopoDS_Wire] = []
        for info in path_info:
            wire = CadServerUtils.make_OCC_wire(info)
            wire_list.append(wire)
        generator = BRepOffsetAPI_ThruSections(isSolid, ruled)
        for wire in wire_list:
            generator.AddWire(wire)
        generator.Build()
        entity: TopoDS.TopoDS_Shell = generator.Shape()
        return entity

    @staticmethod
    def divide_edge_by_nr_of_points(edge: TopoDS.TopoDS_Edge, n_pts):
        """returns a nested list of parameters and points on the edge at the requested interval [(param, gp_Pnt),...]"""
        curve_adapt = BRepAdaptor_Curve(edge)
        _lbound, _ubound = curve_adapt.FirstParameter(), curve_adapt.LastParameter()
        if n_pts <= 1:
            # minimally two points or a Standard_ConstructionError is raised
            raise AssertionError("minimally 2 points required")
        npts = GCPnts_UniformAbscissa(curve_adapt, n_pts, _lbound, _ubound)
        if npts.IsDone():
            tmp = []
            for i in range(1, npts.NbPoints() + 1):
                param = npts.Parameter(i)
                pnt = curve_adapt.Value(param)
                tmp.append((param, pnt))
            return tmp
        raise ValueError("[ERROR]: divide_edge_by_nr_of_points Fail")

    @staticmethod
    def make_OCC_shell(boundary_infos: List[Dict], guide_infos: List[Dict]) -> TopoDS.TopoDS_Shell:
        edge_list: List[TopoDS.TopoDS_Edge] = []
        for info in boundary_infos:
            edge = CadServerUtils.make_OCC_edge(info)
            edge_list.append((edge, None))
        
        brep_plate_builder = BRepOffsetAPI_MakeFilling()
        brep_plate_builder.SetConstrParam(
            0.0001, 0.001, 0.01, 0.01
        )  # Tol2d=1.0, Tol3d=1.0, TolAng=1.0, TolCurv=1.0
        brep_plate_builder.SetApproxParam(8, 240)  # MaxDeg=8, MaxSegments=9

        for boundary_edge, constraint_face in edge_list:
            if constraint_face is None:
                brep_plate_builder.Add(boundary_edge, GeomAbs_C0)
            else:
                brep_plate_builder.Add(boundary_edge, constraint_face, GeomAbs_C0)

        for info in guide_infos:
            if info['type'] == 'edge':
                edge = CadServerUtils.make_OCC_edge(info)
                for pt in CadServerUtils.divide_edge_by_nr_of_points(edge, info['sample_num'])[2:-2]:  # 不可以包含起点与终点
                    brep_plate_builder.Add(pt[1])

            elif info['type'] == 'points':
                for pt in info['points']:
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                    brep_plate_builder.Add(gp_Pnt(x, y, z))

        brep_plate_builder.Build()
        face: TopoDS.TopoDS_Face = brep_plate_builder.Shape()

        builder = BRep_Builder()
        shell = TopoDS.TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        return shell

    @staticmethod
    def convert_h5_to_json(h5_data: h5py.File) -> Dict:
        def extract_group(group) -> Dict:
            result = {}
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    result[key] = extract_group(group[key])
                elif isinstance(group[key], h5py.Dataset):
                    result[key] = group[key][:].tolist()
            return result
        return extract_group(h5_data)

def debug():
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    cad_server_utils = CadServerUtils()
    box = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
    res_json = cad_server_utils.convert_shape_to_json('test_box', box, export_edges=True)
    with open('/home/admin123456/Desktop/work/flask_server/test_box.json', 'w') as f:
        json.dump(res_json, f)


if __name__ == "__main__":
    debug()
