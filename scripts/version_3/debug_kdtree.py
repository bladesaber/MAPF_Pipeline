import numpy as np
from build import mapf_pipeline

radius = 5.0

def KDTreeBaise_debug():
    length = 1.0
    kdtreeData = mapf_pipeline.KDTreeData(radius, length)
    print('[DEBUG]: KDTreeData: radius:%f length:%f' % (kdtreeData.radius, kdtreeData.length))

    x, y, z = 1.0, 2.0, 3.0
    kdtreeRes = mapf_pipeline.KDTreeRes(x, y, z, kdtreeData)
    print('[DEBUG]: KDTreeRes: x:%f y:%f z:%f radius:%f length:%f' % (
        kdtreeRes.x, kdtreeRes.y, kdtreeRes.z, kdtreeRes.data.radius, kdtreeRes.data.length
    ))

def debug_insertAndSearch():
    kdtree = mapf_pipeline.KDTreeWrapper()

    ### Debug Basic Use
    # kdtree.debug_insert()
    # kdtree.debug_search()

    # ### you must save KDTreeData here, other than it will be delete by python
    # datas = []
    # for i in range(5):
    #     data = mapf_pipeline.KDTreeData(radius, i * 1.5)
        
    #     kdtree.insertPoint3D(
    #         # i + np.random.normal(0.0, 0.5), 
    #         # i + np.random.normal(0.0, 0.5), 
    #         # i + np.random.normal(0.0, 0.5), 
    #         i, i, i,
    #         data
    #     )
    #     datas.append(data)

    # res = mapf_pipeline.KDTreeRes()
    # kdtree.nearest(1.0, 1.0, 1.0, res)
    # print(res.x, res.y, res.z, res.data.radius, res.data.length)

    # kdtree.nearest(2.0, 2.0, 2.0, res)
    # print(res.x, res.y, res.z, res.data.radius, res.data.length)

    # kdtree.nearest(3.0, 3.0, 3.0, res)
    # print(res.x, res.y, res.z, res.data.radius, res.data.length)

def debug_insertPath():
    kdtree = mapf_pipeline.KDTreeWrapper()

    path = [
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 25.0),
        (3.0, 3.0, 3.0, 3.0),
        (4.0, 4.0, 4.0, 4.0),
        (5.0, 5.0, 5.0, 10.0)
    ]
    kdtree.insertPath3D(path, 5.0)

    res = mapf_pipeline.KDTreeRes()
    kdtree.nearest(3.0, 3.0, 3.0, res)
    print(res.x, res.y, res.z, res.data.radius, res.data.length)

# debug_insertAndSearch()
debug_insertPath()
