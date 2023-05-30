import numpy as np
from stl import mesh

import plotly.graph_objects as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def stl2mesh(stl_mesh):
    p, q, r = stl_mesh.vectors.shape
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K

mesh_obj = mesh.Mesh.from_file('/home/quan/Desktop/MAPF_Pipeline/scripts/application/company.STL')
vertices, i, j, k = stl2mesh(mesh_obj)
x, y, z = vertices.T

mesh_3d = go.Mesh3d(
    x=x, y=y, z=z, i=i, j=j, k=k, name='test',
    flatshading=True, intensity=z, showscale=True
)

fig = go.Figure(data=[mesh_3d])
py.plot(fig)

