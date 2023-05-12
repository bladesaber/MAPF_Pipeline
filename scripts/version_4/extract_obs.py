import numpy as np
import pandas as pd
import open3d as o3d

# mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=2.0, height=15.0)
# mesh.compute_vertex_normals()
# mesh.translate(np.array([7.5, 7.5, 7.5]))
# o3d.io.write_triangle_mesh(
#     '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/obs.stl', 
#     mesh,
#     write_vertex_colors=True
# )

config = {
    'sample_num': 2000,
    'resolution': 0.25,

    'stl_file': '/home/quan/Desktop/MAPF_Pipeline/scripts/application/replace.STL',
    'csv_file': '/home/quan/Desktop/MAPF_Pipeline/scripts/application/obs.csv',
}

mesh = o3d.io.read_triangle_mesh(config['stl_file'])
mesh.compute_vertex_normals()

pcd = mesh.sample_points_poisson_disk(config['sample_num'])
pcd_np = np.asarray(pcd.points)

reso = config['resolution']
pcd_df = pd.DataFrame(pcd_np, columns=['x', 'y', 'z'])
pcd_df['x'] = np.round(pcd_df['x'] / reso, decimals=0) * reso
pcd_df['y'] = np.round(pcd_df['y'] / reso, decimals=0) * reso
pcd_df['z'] = np.round(pcd_df['z'] / reso, decimals=0) * reso
pcd_df['tag'] = pcd_df['x'].astype(str) + pcd_df['y'].astype(str) + pcd_df['z'].astype(str)
pcd_df.drop_duplicates(subset=['tag'], inplace=True)
pcd_df['radius'] = 0.1

pcd_df[['x', 'y', 'z', 'radius']].to_csv(config['csv_file'])
