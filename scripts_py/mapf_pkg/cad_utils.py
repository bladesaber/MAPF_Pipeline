import numpy as np
from typing import Union, List, Dict, Tuple
import OCC.Core.TopoDS as TopoDS

from OCC.Display.SimpleGui import init_display, load_backend
from OCC.Display.WebGl import threejs_renderer

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer, StlAPI_Reader
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Ax3, gp_Dir, gp_Circ
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BezierCurve, Geom_BSplineSurface
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeShell,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.BRepOffsetAPI import (BRepOffsetAPI_MakePipe, BRepOffsetAPI_ThruSections,
                                    BRepOffsetAPI_MakePipeShell, BRepOffsetAPI_MakeFilling)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface

from OCC.Core.TopoDS import TopoDS_Shell, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.Law import Law_Linear, Law_BSpline, Law_BSpFunc, Law_Interpolate, Law_Constant
from OCC.Core.BRepAlgoAPI import (
    BRepAlgoAPI_Fuse,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Section,
    BRepAlgoAPI_Cut,
)
from OCC.Core.GeomAbs import GeomAbs_G1, GeomAbs_G2, GeomAbs_C0
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa


# from pyspline import Surface


class OcctIO(object):
    @staticmethod
    def read_STEP(file: str):
        if file.split('.')[-1] in ['stp', 'step', 'STP', 'STEP']:
            step_reader = STEPControl_Reader()
            step_reader.ReadFile(file)
            step_reader.TransferRoot()
            shape = step_reader.Shape()
            return shape
        return None

    @staticmethod
    def write_STEP(file: str, shape: Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Solid]):
        saver = STEPControl_Writer()
        saver.Transfer(shape, STEPControl_AsIs)
        saver.Write(file)

    @staticmethod
    def read_STL(file: str):
        if file.split('.')[-1] in ['stl', 'STL']:
            stl_reader = StlAPI_Reader()
            shape = TopoDS.TopoDS_Shape()
            stl_reader.Read(shape, file)
            return shape
        return None

    @staticmethod
    def write_STL(file: str, shape: Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Solid], is_ascii=True):
        # ------ high possible fail
        saver = StlAPI_Writer()
        saver.SetASCIIMode(is_ascii)
        return saver.Write(shape, file)

    @staticmethod
    def write_IGES(file: str, shape: Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Wire]):
        writer = IGESControl_Writer()
        writer.AddShape(shape)
        writer.Write(file)


class OcctAux(object):
    @staticmethod
    def create_law(law_type: str, law_cfg: Dict):
        if law_type == 'constant':
            law = Law_Constant()
            law.Set(law_cfg['radius'], law_cfg['radius'], law_cfg['radius'])
        elif law_type == 'linear':
            law = Law_Linear()
            law.Set(0, law_cfg['start_ratio'], 1, law_cfg['end_ratio'])
        elif law_type == 'bspline':
            c_radius = law_cfg['c_radius']
            degree = law_cfg['degree']

            n_control = c_radius.shape[0]
            c_array = TColStd_Array1OfReal(1, n_control)
            for i, radius in enumerate(c_radius):
                c_array.SetValue(i + 1, float(radius))

            n_knot = (n_control + degree + 1) - 2 * (degree + 1) + 2
            knot = np.linspace(0.0, 1.0, n_knot)
            knot_array = TColStd_Array1OfReal(1, n_knot)
            for i, k_value in enumerate(knot):
                # print(f"[INFO]: Setting knot index:{i + 1} k_value:{k_value}")
                knot_array.SetValue(i + 1, k_value)

            mul_value = np.ones(shape=(n_knot,), dtype=int)
            mul_value[0] = degree + 1
            mul_value[-1] = degree + 1
            multiplicities_array = TColStd_Array1OfInteger(1, n_knot)
            for i, v in enumerate(mul_value):
                # print(f"[INFO]: Setting knot index:{i + 1} multiplicities:{v}")
                multiplicities_array.SetValue(i + 1, int(v))

            law = Law_BSpFunc(
                Law_BSpline(c_array, knot_array, multiplicities_array, degree, False), 0.0, 1.0
            )

        else:
            raise ValueError()

        return law


class OcctProfile(object):
    @staticmethod
    def make_vertex(point: gp_Pnt):
        return BRepBuilderAPI_MakeVertex(point).Vertex()

    @staticmethod
    def make_point(x, y, z) -> gp_Pnt:
        return gp_Pnt(x, y, z)

    @staticmethod
    def make_pack(
            solid_list: List[TopoDS.TopoDS_Shape], builder: BRep_Builder = None, compound: TopoDS_Compound = None
    ) -> TopoDS_Compound:
        if compound is None:
            compound = TopoDS_Compound()
        if builder is None:
            builder = BRep_Builder()
        builder.MakeCompound(compound)
        for solid in solid_list:
            builder.Add(compound, solid)
        return compound

    @staticmethod
    def make_direction_vector(vector_x, vector_y, vector_z) -> gp_Dir:
        return gp_Dir(vector_x, vector_y, vector_z)

    @staticmethod
    def make_AX2(point: gp_Pnt, direction_vector: gp_Dir) -> gp_Ax2:
        """2D Coordinate axis: point + vector"""
        return gp_Ax2(point, direction_vector)

    @staticmethod
    def create_bspline(
            control_pcd: np.ndarray, degree: int, left_clamp: bool, right_clamp: bool,
    ) -> Geom_BSplineCurve:
        n_control = control_pcd.shape[0]
        pnt_array = TColgp_Array1OfPnt(1, n_control)
        for i, point in enumerate(control_pcd):
            # print(f"[INFO]: Setting c_pcd index:{i + 1} pcd:({point[0]}, {point[1]}, {point[2]})")
            pnt_array.SetValue(i + 1, gp_Pnt(point[0], point[1], point[2]))

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
            # print(f"[INFO]: Setting knot index:{i + 1} multiplicities:{v}")
            multiplicities_array.SetValue(i + 1, int(v))

        profile = Geom_BSplineCurve(pnt_array, knot_array, multiplicities_array, degree, False)

        return profile

    @staticmethod
    def make_pipe(
            path: TopoDS.TopoDS_Wire, profile: Union[TopoDS.TopoDS_Face, TopoDS.TopoDS_Wire],
            return_shape=True
    ) -> Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Solid, TopoDS.TopoDS_Shape, BRepOffsetAPI_MakePipe]:
        if return_shape:
            return BRepOffsetAPI_MakePipe(path, profile).Shape()
        return BRepOffsetAPI_MakePipe(path, profile)

    @staticmethod
    def make_pipe_shell(
            path: TopoDS.TopoDS_Wire, profile: TopoDS.TopoDS_Wire,
            law_runner: Union[Law_Linear, Law_BSpline, Law_Interpolate, Law_Constant],
            return_shape=True
    ) -> Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Shape, BRepOffsetAPI_MakePipeShell]:
        pipe = BRepOffsetAPI_MakePipeShell(path)
        pipe.SetLaw(profile, law_runner, False, True)

        if return_shape:
            return pipe.Shape()
        return pipe

    @staticmethod
    def make_circle(
            point: Union[List, np.ndarray], vector: Union[List, np.ndarray], radius: float
    ) -> gp_Circ:
        profile = gp_Circ(
            OcctProfile.make_AX2(
                OcctProfile.make_point(point[0], point[1], point[2]),
                OcctProfile.make_direction_vector(vector[0], vector[1], vector[2])
            ), radius
        )
        return profile

    @staticmethod
    def make_arc_circle(
            point: Union[List, np.ndarray], vector: Union[List, np.ndarray], radius: float,
            start_point: Union[List, np.ndarray], end_point: Union[List, np.ndarray]
    ) -> TopoDS.TopoDS_Edge:
        profile = gp_Circ(
            OcctProfile.make_AX2(
                OcctProfile.make_point(point[0], point[1], point[2]),
                OcctProfile.make_direction_vector(vector[0], vector[1], vector[2])
            ), radius
        )
        arc = BRepBuilderAPI_MakeEdge(
            profile,
            OcctProfile.make_point(start_point[0], start_point[1], start_point[2]),
            OcctProfile.make_point(end_point[0], end_point[1], end_point[2]),
        ).Edge()
        return arc

    @staticmethod
    def make_box(center: gp_Pnt, dx: float, dy: float, dz: float) -> TopoDS.TopoDS_Shape:
        return BRepPrimAPI_MakeBox(center, dx, dy, dz).Shape()

    @staticmethod
    def make_sphere(center: gp_Pnt, radius: float) -> TopoDS.TopoDS_Shape:
        return BRepPrimAPI_MakeSphere(center, radius).Shape()

    @staticmethod
    def make_cylinder(center: gp_Ax2, radius: float, height: float) -> TopoDS.TopoDS_Shape:
        return BRepPrimAPI_MakeCylinder(center, radius, height).Shape()

    @staticmethod
    def make_thruSections(
            wires: List[TopoDS.TopoDS_Wire], is_solid=True, ruled=True
    ) -> Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Solid]:
        generator = BRepOffsetAPI_ThruSections(is_solid, ruled)
        for wire in wires:
            generator.AddWire(wire)
        generator.Build()
        return generator.Shape()

    @staticmethod
    def make_line(points: np.ndarray) -> TopoDS.TopoDS_Wire:
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(points.shape[0] - 1):
            left_point, right_point = points[i], points[i + 1]
            edge = BRepBuilderAPI_MakeEdge(
                OcctProfile.make_point(left_point[0], left_point[1], left_point[2]),
                OcctProfile.make_point(right_point[0], right_point[1], right_point[2])
            ).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    @staticmethod
    def make_point_cloud(points: np.ndarray) -> TopoDS.TopoDS_Compound:
        pcd = []
        for point in points:
            pcd.append(OcctProfile.make_vertex(OcctProfile.make_point(point[0], point[1], point[2])))
        return OcctProfile.make_pack(pcd)

    @staticmethod
    def make_bspline_surface(control_pcd: np.ndarray, degree_u: int, degree_v: int) -> Geom_BSplineSurface:
        # ku, kv = 3, 3
        # surf = Surface(x=x, y=y, z=z, ku=ku, kv=kv, nCtlu=20, nCtlv=20, nIter=100)
        # # surf.writeTecplot('/home/admin123456/Desktop/work/path_examples/test_example/surf.dat')
        # control_pcd = surf.coef

        n_u, n_v = control_pcd.shape[:2]
        control_pcd_array = TColgp_Array2OfPnt(1, n_u, 1, n_v)
        for u in range(n_u):
            for v in range(n_u):
                point = control_pcd[u, v, :]
                control_pcd_array.SetValue(u + 1, v + 1, OcctProfile.make_point(point[0], point[1], point[2]))

        # ------ knot of u
        n_u_knot = (n_u + degree_u + 1) - 2 * (degree_u + 1) + 2
        u_knot = np.linspace(0.0, 1.0, n_u_knot)
        u_knot_array = TColStd_Array1OfReal(1, n_u_knot)
        for i, k_value in enumerate(u_knot):
            u_knot_array.SetValue(i + 1, k_value)

        u_mul_value = np.ones(shape=(n_u_knot,), dtype=int)
        u_mul_value[0] = degree_u + 1
        u_mul_value[-1] = degree_u + 1
        u_multiplicities_array = TColStd_Array1OfInteger(1, n_u_knot)
        for i, v in enumerate(u_mul_value):
            u_multiplicities_array.SetValue(i + 1, int(v))

        # ------ knot of v
        n_v_knot = (n_v + degree_v + 1) - 2 * (degree_v + 1) + 2
        v_knot = np.linspace(0.0, 1.0, n_v_knot)
        v_knot_array = TColStd_Array1OfReal(1, n_v_knot)
        for i, k_value in enumerate(v_knot):
            v_knot_array.SetValue(i + 1, k_value)

        v_mul_value = np.ones(shape=(n_v_knot,), dtype=int)
        v_mul_value[0] = degree_u + 1
        v_mul_value[-1] = degree_v + 1
        v_multiplicities_array = TColStd_Array1OfInteger(1, n_v_knot)
        for i, v in enumerate(u_mul_value):
            v_multiplicities_array.SetValue(i + 1, int(v))

        # ------ Step 5: Create B-spline surface
        bspline_surface = Geom_BSplineSurface(
            control_pcd_array,
            u_knot_array, v_knot_array,
            u_multiplicities_array, v_multiplicities_array,
            degree_u, degree_v,
            False, False
        )

        return bspline_surface

    @staticmethod
    def divide_edge_by_nr_of_points(edge: TopoDS.TopoDS_Edge, n_pts):
        """returns a nested list of parameters and points on the edge
        at the requested interval [(param, gp_Pnt),...]
        """
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
    def fill_surface_from_curve_network(
            boundaries: List[Tuple[TopoDS.TopoDS_Edge, Union[TopoDS.TopoDS_Face, None]]],
            guide_lines: List[TopoDS.TopoDS_Edge] = [],
            guide_points: np.ndarray = None,
            enforce_tangency=False, sample_num: int = 12
    ):
        brep_plate_builder = BRepOffsetAPI_MakeFilling()
        if enforce_tangency:
            brep_plate_builder.SetConstrParam(
                0.0001, 0.001, 0.01, 0.01
            )  # Tol2d=1.0, Tol3d=1.0, TolAng=1.0, TolCurv=1.0
            brep_plate_builder.SetApproxParam(8, 240)  # MaxDeg=8, MaxSegments=9

        for boundary_edge, constraint_face in boundaries:
            if constraint_face is None:
                brep_plate_builder.Add(boundary_edge, GeomAbs_C0)
            else:
                brep_plate_builder.Add(boundary_edge, constraint_face, GeomAbs_C0)

        for guide_line in guide_lines:
            for pt in OcctProfile.divide_edge_by_nr_of_points(guide_line, sample_num)[2:-2]:  # 不可以包含起点与终点
                brep_plate_builder.Add(pt[1])

        if guide_points is not None:
            for guide_point in guide_points:
                x, y, z = float(guide_point[0]), float(guide_point[1]), float(guide_point[2])
                brep_plate_builder.Add(OcctProfile.make_point(x, y, z))

        brep_plate_builder.Build()
        face: TopoDS.TopoDS_Face = brep_plate_builder.Shape()
        return face


class OcctOperation(object):
    @staticmethod
    def fuse(shape1: TopoDS.TopoDS_Solid, shape2: TopoDS.TopoDS_Solid) -> TopoDS.TopoDS_Compound:
        return BRepAlgoAPI_Fuse(shape1, shape2).Shape()

    @staticmethod
    def common(shape1: TopoDS.TopoDS_Solid, shape2: TopoDS.TopoDS_Solid) -> TopoDS.TopoDS_Compound:
        return BRepAlgoAPI_Common(shape1, shape2).Shape()

    @staticmethod
    def cut(shape_orig: TopoDS.TopoDS_Solid, shape_tool: TopoDS.TopoDS_Solid) -> TopoDS.TopoDS_Compound:
        return BRepAlgoAPI_Cut(shape_orig, shape_tool).Shape()


class OcctUtil(object):
    """ Profile -> Edge -> Wire -> Face -> Shell/Solid """

    @staticmethod
    def face2shell(
            face: TopoDS.TopoDS_Face, builder: BRep_Builder = None, shell: TopoDS.TopoDS_Shell = None
    ) -> TopoDS.TopoDS_Shell:
        if builder is None:
            builder = BRep_Builder()
        if shell is None:
            shell = TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        return shell

    @staticmethod
    def shell2solid(shell: TopoDS.TopoDS_Shell) -> TopoDS.TopoDS_Solid:
        if shell.Closed():
            return BRepBuilderAPI_MakeSolid(shell).Solid()
        print("[ERROR]: Shell is not closed")
        return None

    @staticmethod
    def make_edge(profile) -> TopoDS.TopoDS_Edge:
        return BRepBuilderAPI_MakeEdge(profile).Edge()

    @staticmethod
    def make_wire(edges: List[TopoDS.TopoDS_Edge]) -> TopoDS.TopoDS_Wire:
        builder = BRepBuilderAPI_MakeWire()
        for edge in edges:
            builder.Add(edge)
        return builder.Wire()

    @staticmethod
    def make_face(wire: TopoDS.TopoDS_Wire) -> TopoDS.TopoDS_Face:
        return BRepBuilderAPI_MakeFace(wire).Face()

    @staticmethod
    def make_shell() -> TopoDS.TopoDS_Shell:
        return TopoDS.TopoDS_Shell()

    @staticmethod
    def sewing_faces(
            faces: List[Union[TopoDS.TopoDS_Face, TopoDS.TopoDS_Shell]],
            builder: BRepBuilderAPI_Sewing = None
    ) -> Union[TopoDS.TopoDS_Shell, TopoDS.TopoDS_Shape]:
        if builder is None:
            builder = BRepBuilderAPI_Sewing()
        for face in faces:
            builder.Add(face)
        builder.Perform()
        return builder.SewedShape()


class OcctVis(object):
    """OCCT是个异步可视化库,不可以与pyvista同步使用"""

    def __init__(self):
        self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()

    def plot(self, shape: TopoDS.TopoDS_Shape, color: Tuple[float, float, float] = None):
        self.display.DisplayShape(shape, update=True, color=color)

    def show(self):
        self.start_display()

    @staticmethod
    def init_plotter():
        # load_backend('pyqt5')
        display, start_display, add_menu, add_function_to_menu = init_display()
        return display, start_display, add_menu, add_function_to_menu

    @staticmethod
    def init_threejs():
        render = threejs_renderer.ThreejsRenderer()
        return render
