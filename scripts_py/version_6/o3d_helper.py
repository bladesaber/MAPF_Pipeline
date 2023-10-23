import numpy as np
import open3d as o3d
from scipy.spatial import transform

def create_Z_screw_mesh(x, y, z, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius = radius, height=height
    )
    mesh.translate(np.array([x, y, z - height/2.0]), relative=False)
    mesh.compute_vertex_normals()
    return mesh

def create_Z_valve_mesh(x, y, z, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius = radius, height=height
    )
    mesh.translate(np.array([x, y, z - height/2.0]), relative=False)
    mesh.compute_vertex_normals()
    return mesh

def create_X_valve_mesh(x, y, z, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius = radius, height=height
    )
    R = transform.Rotation.from_euler(seq='xyz', angles=[0.0, 90.0, 0.0], degrees=True).as_matrix()
    mesh.rotate(R)
    mesh.translate(np.array([x - height, y, z]), relative=False)
    mesh.compute_vertex_normals()
    return mesh

def create_Z_support_mesh(x, y, z, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius = radius, height=height
    )
    mesh.translate(np.array([x, y, z - height/2.0]), relative=False)
    mesh.compute_vertex_normals()
    return mesh

def create_create_sphere(x, y, z, radius):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(np.array([x, y, z]), relative=False)
    mesh.compute_vertex_normals()
    return mesh

