import open3d as o3d
import numpy as np

def create_line_3d(xyz, color=np.array([0.0, 0.0, 1.0])):
    path_o3d = o3d.geometry.LineSet()

    path_o3d.points = o3d.utility.Vector3dVector(xyz)
    path_o3d.colors = o3d.utility.Vector3dVector(np.tile(
        color.reshape((1, -1)), (xyz.shape[0], 1)
    ))

    lines = []
    for idx in range(xyz.shape[0]-1):
        lines.append([idx, idx+1])

    path_o3d.lines = o3d.utility.Vector2iVector(np.array(lines))

    return path_o3d

def generate_3d_points_from_circle(translate, normal, radius, num_points):
    # Ensure normal vector is unit length
    normal = normal / np.linalg.norm(normal)

    # Generate a random orthogonal vector to the normal
    ortho1 = np.cross(normal, np.random.rand(3))
    ortho1 = ortho1 / np.linalg.norm(ortho1)

    # Generate another orthogonal vector to the normal and first orthogonal
    ortho2 = np.cross(normal, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)

    # Use the three orthogonal vectors to form a rotation matrix
    rotation_matrix = np.concatenate([
        ortho1.reshape((-1, 1)),
        ortho2.reshape((-1, 1)),
        normal.reshape((-1, 1)),
    ], axis=1)

    # Generate evenly spaced points around the circle
    angle = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    points_2d = np.concatenate([
        x.reshape((-1, 1)),
        y.reshape((-1, 1)),
        np.zeros((x.shape[0], 1))
    ], axis=1)

    # Rotate and translate the 2D points to form the final 3D points
    points_3d = points_2d.dot(rotation_matrix.T) + translate.reshape((1, -1))

    axes: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.8, origin=translate
    )
    axes = axes.rotate(rotation_matrix)

    return points_3d, axes

def create_tunnel_3d(xyz, radius, color):
    pcds, axes_list = [], []
    for idx in range(xyz.shape[0] - 1):
        vec = xyz[idx+1] - xyz[idx]
        pcd, axes = generate_3d_points_from_circle(xyz[idx], vec, radius, num_points=60)

        pcds.append(pcd)
        axes_list.append(axes)

    pcds = np.concatenate(pcds, axis=0)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcds)
    np_colors = np.tile(color.reshape((1, -1)), (pcds.shape[0], 1))
    pcd_o3d.colors = o3d.utility.Vector3dVector(np_colors)

    return pcd_o3d, axes_list

def sample_tunnel(xyz, radius):
    pcds_list = []
    for idx in range(xyz.shape[0] - 1):
        vec = xyz[idx + 1] - xyz[idx]
        pcd, axes = generate_3d_points_from_circle(xyz[idx], vec, radius, num_points=60)
        pcds_list.append(pcd)
    return pcds_list
