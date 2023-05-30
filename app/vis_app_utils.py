import numpy as np
from stl import mesh

import plotly.graph_objects as go

### offline show
# import plotly.offline as py
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)

def read_stl_data(file:str):
    assert file.endswith('.stl') or file.endswith('.STL')

    stl_mesh = mesh.Mesh.from_file(file)

    p, q, r = stl_mesh.vectors.shape
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])

    x, y, z = vertices.T

    return (x, y, z), (I, J, K)

def plotly_mesh3d(x, y, z, i, j, k, name:str):
    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, name=name,
        flatshading=True, intensity=z, showscale=True
    )
    return mesh
