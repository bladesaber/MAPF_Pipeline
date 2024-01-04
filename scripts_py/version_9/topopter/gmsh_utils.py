import numpy as np
import gmsh
import math
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI


class GmshGeoUtils(object):
    @staticmethod
    def init_env():
        gmsh.initialize()

    @staticmethod
    def define_model(name: str):
        gmsh.model.add(name)

    @staticmethod
    def set_model(name):
        gmsh.model.setCurrent(name)

    @staticmethod
    def add_point(x, y, z, meshSize, tag=-1):
        """
        tag: -1 means auto define
        return: tag
        """
        return gmsh.model.geo.addPoint(x, y, z, meshSize, tag)

    @staticmethod
    def add_line(point0_tag, point1_tag, tag=-1):
        return gmsh.model.geo.addLine(point0_tag, point1_tag, tag)

    @staticmethod
    def add_circleArc(startTag, centerTag, endTag, tag=-1):
        return gmsh.model.geo.addCircleArc(startTag, centerTag, endTag, tag)

    @staticmethod
    def add_bspline(pointTags, tag=-1):
        """
        Add a cubic b-spline curve
        """
        return gmsh.model.geo.addBSpline(pointTags, tag)

    @staticmethod
    def add_curveLoop(lineTags: list[int], tag=-1):
        """
        lineTag_set: if the sign of tag is negative, it means the orientation of line is opposite.
        if there is hole in the surface, please include the line tags of hole.
        """
        return gmsh.model.geo.addCurveLoop(lineTags, tag)

    @staticmethod
    def add_surface(curveLoop_tags: list[int], tag=-1):
        """
        multi curve loop -> surface
        """
        return gmsh.model.geo.addPlaneSurface(curveLoop_tags, tag)

    @staticmethod
    def add_surfaceLoop(surface_tags: list[int], tag=-1):
        return gmsh.model.geo.addSurfaceLoop(surface_tags, tag)

    @staticmethod
    def add_volume(surfaceLoop: int, tag=-1):
        return gmsh.model.geo.addVolume([surfaceLoop], tag)

    @staticmethod
    def set_line_segment(line_tag, nPoints):
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag, nPoints)

    @staticmethod
    def setTransfiniteSurface():
        # gmsh.model.geo.mesh.setTransfiniteSurface
        raise NotImplementedError

    @staticmethod
    def extrude(dimTags, dx, dy, dz, numElements=[], heights=[], recombine=False):
        """
        numElements: give the number of elements in each layer
        heights: give the (cumulative) height of the different layers
        return: outDimTags
        """
        return gmsh.model.geo.extrude(dimTags, dx, dy, dz, numElements, heights, recombine)

    @staticmethod
    def revolve(dimTags, center_xyz, center_norm, angle, numElements=[], heights=[], recombine=False):
        """
        extrude of rotate version
        """
        return gmsh.model.geo.revolve(
            dimTags,
            center_xyz[0], center_xyz[1], center_xyz[2],
            center_norm[0], center_norm[1], center_norm[2], angle, numElements, heights, recombine
        )

    @staticmethod
    def translate(dimTags, dx, dy, dz):
        """
        dimTags: vector of (dim, tag) pairs, [(dim, tag)]
        """
        gmsh.model.geo.translate(dimTags, dx, dy, dz)

    @staticmethod
    def rotate(dimTags, center_xyz, center_norm, angle):
        """
        dimTags rotate based on center_xyz with the norm vector center_norm in clockwise
        """
        gmsh.model.geo.rotate(
            dimTags,
            center_xyz[0], center_xyz[1], center_xyz[2],
            center_norm[0], center_norm[1], center_norm[2], angle
        )

    @staticmethod
    def remove(dimTags, recursive=False):
        gmsh.model.geo.remove(dimTags, recursive)

    @staticmethod
    def synchronize():
        """
        before mesh or define group, must call synchronize to upload data to engine
        """
        gmsh.model.geo.synchronize()

    @staticmethod
    def define_group(dim, tags_set: list[int], tag=-1, name=""):
        """
        dim: specify the type of tag:
            dim-0: mean point
            dim-1: mean line
            dim-2: mean surface
            dim-3: mean volume
        """
        return gmsh.model.addPhysicalGroup(dim, tags_set, tag, name)

    @staticmethod
    def generate_mesh(dim: int):
        assert dim in [2, 3]
        gmsh.model.mesh.generate(dim)

    @staticmethod
    def get_entityFromDim(dim):
        """
        return all the dimTag pairs of specify dim
        Return: dimTags
        """
        return gmsh.model.getEntities(dim)

    @staticmethod
    def get_entityInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim):
        """
        Return: dimTags
        """
        return gmsh.model.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim)

    @staticmethod
    def get_entityFromBoundary(dimTags, combined=False, oriented=False, recursive=False):
        """
        volume dimTags: vector of (dim, tag) pairs, [(dim, tag)], dim is 3 here
        oriented: whether multiplied by the sign of the boundary entity
        recursive: if true, always return point(dim=0)
        """
        return gmsh.model.getBoundary(dimTags, combined, oriented, recursive)

    @staticmethod
    def get_node_info(dim, tag):
        """
        return: nodeTags, nodeCoords(x, y, z), nodeParams
        """
        return gmsh.model.mesh.getNodes(dim, tag)

    @staticmethod
    def get_element_info(dim, tag):
        """
        return: elemTypes, elemTags: list[tag], elemNodeTags
        """
        return gmsh.model.mesh.getElements(dim, tag)

    @staticmethod
    def get_entityName(dim, tag):
        return gmsh.model.getEntityName(dim, tag)

    @staticmethod
    def get_adjacencies(dim, tag):
        """
        upward vector: returns the tags of adjacent entities of dimension `dim' + 1
        downward vector: returns the tags of adjacent entities of dimension `dim' - 1
        """
        return gmsh.model.getAdjacencies(dim, tag)

    @staticmethod
    def get_group(dim, tag):
        return gmsh.model.getPhysicalGroupsForEntity(dim, tag)

    @staticmethod
    def set_mesh_size(dimTags, mesh_size):
        """
        control the element size of mesh
        """
        return gmsh.model.mesh.setSize(dimTags, mesh_size)

    @staticmethod
    def show():
        gmsh.option.setNumber("Geometry.PointNumbers", 1)
        gmsh.option.setColor("Geometry.Color.Points", 255, 165, 0)
        gmsh.option.setColor("General.Color.Text", 255, 255, 255)
        gmsh.option.setColor("Mesh.Color.Points", 255, 0, 0)
        r, g, b, a = gmsh.option.getColor("Geometry.Points")
        gmsh.option.setColor("Geometry.Surfaces", r, g, b, a)
        gmsh.fltk.run()

    @staticmethod
    def finish():
        gmsh.finalize()

    @staticmethod
    def output_XDMF(comm: MPI.Comm, model: gmsh.model, name: str, file, mode, dim=3):
        assert file.endswith('.xdmf') and mode in ['w', 'r', 'a']
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, rank=0, gdim=dim)
        mesh.name = name
        cell_tags.name = f"{mesh.name}_cells"
        facet_tags.name = f"{mesh.name}_facets"
        with XDMFFile(mesh.comm, file, mode) as f:
            mesh.topology.create_connectivity(2, 3)
            f.write_mesh(mesh)
            f.write_meshtags(
                cell_tags, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )
            f.write_meshtags(
                facet_tags, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )

    @staticmethod
    def output_msh(file: str, save_all=False):
        # Remember that by default, if physical groups are defined, Gmsh will export in
        # the output mesh file only those elements that belong to at least one physical group.
        assert file.endswith('.msh')
        if save_all:
            gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(file)


class GmshOccUtils(GmshGeoUtils):
    @staticmethod
    def add_point(x, y, z, meshSize, tag=-1):
        return gmsh.model.occ.addPoint(x, y, z, meshSize, tag)

    @staticmethod
    def add_line(point0_tag, point1_tag, tag=-1):
        return gmsh.model.occ.addLine(point0_tag, point1_tag, tag)

    @staticmethod
    def add_curveLoop(lineTags: list[int], tag=-1):
        return gmsh.model.occ.addCurveLoop(lineTags, tag)

    @staticmethod
    def add_surface(curveLoop_tags: list[int], tag=-1):
        return gmsh.model.occ.addPlaneSurface(curveLoop_tags, tag)

    @staticmethod
    def add_surfaceLoop(surface_tags: list[int], tag=-1):
        return gmsh.model.occ.addSurfaceLoop(surface_tags, tag)

    @staticmethod
    def add_volume(surfaceLoop: int, tag=-1):
        return gmsh.model.occ.addVolume([surfaceLoop], tag)

    @staticmethod
    def add_circleArc(startTag, centerTag, endTag, tag=-1):
        return gmsh.model.occ.addCircleArc(startTag, centerTag, endTag, tag)

    @staticmethod
    def add_bspline(pointTags, tag=-1):
        return gmsh.model.occ.addBSpline(pointTags, tag)

    @staticmethod
    def add_box(x0, y0, z0, dx, dy, dz, tag=-1):
        """
        return: volume tag
        """
        return gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz, tag)

    @staticmethod
    def add_sphere(xc, yc, zc, radius, tag=-1, angle1=-math.pi / 2, angle2=math.pi / 2, angle3=2 * math.pi):
        return gmsh.model.occ.addSphere(xc, yc, zc, radius, tag, angle1, angle2, angle3)

    @staticmethod
    def add_cylinder(x, y, z, dx, dy, dz, r, tag=-1, angle=2 * math.pi):
        return gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r, tag, angle)

    @staticmethod
    def add_cone(x, y, z, dx, dy, dz, r1, r2, tag=-1, angle=2 * math.pi):
        return gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2, tag, angle)

    @staticmethod
    def cut_volume(objectDimTags, toolDimTags, tag=-1, removeObject=True, removeTool=True):
        """
        Return: outDimTags
        """
        return gmsh.model.occ.cut(objectDimTags, toolDimTags, tag, removeObject, removeTool)[0]

    @staticmethod
    def merge_crossover_element(objectDimTags, toolDimTags, tag=-1, removeObject=True, removeTool=True):
        """
        if two surface or two lines overlap, use fragment avoid remesh same surface or line
        Return: outDimTags
        """
        return gmsh.model.occ.fragment(objectDimTags, toolDimTags, tag, removeObject, removeTool)[0]

    @staticmethod
    def extrude(dimTags, dx, dy, dz, numElements=[], heights=[], recombine=False):
        return gmsh.model.occ.extrude(dimTags, dx, dy, dz, numElements, heights, recombine)

    @staticmethod
    def revolve(dimTags, center_xyz, center_norm, angle, numElements=[], heights=[], recombine=False):
        return gmsh.model.occ.revolve(
            dimTags, center_xyz[0], center_xyz[1], center_xyz[2],
            center_norm[0], center_norm[1], center_norm[2], angle, numElements, heights, recombine
        )

    @staticmethod
    def translate(dimTags, dx, dy, dz):
        gmsh.model.occ.translate(dimTags, dx, dy, dz)

    @staticmethod
    def rotate(dimTags, center_xyz, center_norm, angle):
        gmsh.model.occ.rotate(
            dimTags, center_xyz[0], center_xyz[1], center_xyz[2],
            center_norm[0], center_norm[1], center_norm[2], angle
        )

    @staticmethod
    def get_centerOfEntity(dim, tag):
        xyz = gmsh.model.occ.getCenterOfMass(dim, tag)
        return xyz

    @staticmethod
    def remove(dimTags, recursive=False):
        gmsh.model.occ.remove(dimTags, recursive)

    @staticmethod
    def synchronize():
        gmsh.model.occ.synchronize()

    @staticmethod
    def output_CAD(file: str, save_all=False):
        assert file.endswith('.step') or file.endswith('.brep')
        if save_all:
            gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(file)

    @staticmethod
    def import_data(file: str):
        assert file.endswith('step')
        return gmsh.model.occ.importShapes(file)

    @staticmethod
    def log_msg():
        gmsh.logger.start()


if __name__ == '__main__':
    pass
