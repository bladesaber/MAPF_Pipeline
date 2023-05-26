import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

path0 = [
    (0.0, 0.0, 0.0, 0.5),
    (1.0, 0.0, 0.0, 0.5),
    (2.0, 0.0, 0.0, 0.5),
    (3.0, 0.0, 0.0, 0.5),
    (3.0, 1.0, 0.0, 0.5),
    (3.0, 2.0, 0.0, 0.5),
    (3.0, 3.0, 0.0, 0.5),
    (3.0, 4.0, 0.0, 0.5),
]
startDire0 = (0.0, 0.0)
endDire0 = (np.deg2rad(90.0), 0.0)

path1 = [
    (8.0, 0.0, 0.0, 0.5),
    (7.0, 0.0, 0.0, 0.5),
    (6.0, 0.0, 0.0, 0.5),
    (5.0, 0.0, 0.0, 0.5),
    (4.0, 0.0, 0.0, 0.5),
    (3.1, 0.0, 0.0, 0.5),
    (3.1, 1.0, 0.0, 0.5),
    (3.1, 2.0, 0.0, 0.5),
    (3.5, 3.0, 0.0, 0.5),
    (4.0, 4.0, 0.0, 0.5),
    (5.0, 4.0, 0.0, 0.5),
]
startDire1 = (np.deg2rad(180.0), 0.0)
endDire1 = (np.deg2rad(0.0), 0.0)

model = mapf_pipeline.SmootherG2O()

### Step 1 Add Paths
model.addPath(
    groupIdx=0, pathIdx=0, path_xyzr=path0, startDire=startDire0, endDire=endDire0
)
# model.addPath(
#     groupIdx=0, pathIdx=1, path_xyzr=path1, startDire=startDire1, endDire=endDire1
# )

### Step 2 Update Direction
for group_key in model.groupMap.keys():
    groupPath = model.groupMap[group_key]
    groupPath.updatePathDirection()

# for group_key in model.groupMap.keys():
#     groupPath = model.groupMap[group_key]
#     for key in groupPath.nodeMap.keys():
#         node = groupPath.nodeMap[key]
#         print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
#             node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
#         ))

model.initOptimizer()
for outer_i in range(3):
    
    ### Debug Check Param Before Optimize
    # for group_key in model.groupMap.keys():
    #     groupPath = model.groupMap[group_key]
    #     for key in groupPath.nodeMap.keys():
    #         node = groupPath.nodeMap[key]
    #         print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
    #             node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
    #         ))

    ### Step 3.1 Build Graph 
    model.build_graph(
        elasticBand_weight=1.0,
        crossPlane_weight=1.0,
        curvature_weight=1.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
    )

    model.info()
    
    ### Step 3.2 Optimize
    model.optimizeGraph(10, False)

    ### Step 3.3 Update Vertex to Node
    model.update2groupVertex()

    ### Debug Show Data
    # for group_key in model.groupMap.keys():
    #     groupPath = model.groupMap[group_key]
    #     for node_key in groupPath.nodeMap.keys():
    #         node = groupPath.nodeMap[node_key]
    #         # print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f alpha:%f theta:%f' % (
    #         #     node.nodeIdx, node.groupIdx, 
    #         #     node.vertex_x(), node.vertex_y(), node.vertex_z(), 
    #         #     node.vertex_alpha(), node.vertex_theta()
    #         # ))
    #         print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
    #             node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
    #         ))
    # print('----------------------------')

    ### Step 3.4 Clear Graph
    model.clear_graph()

    ### Debug Vis
    for group_key in model.groupMap.keys():
        groupPath = model.groupMap[group_key]
        for pathIdx in groupPath.pathIdxs_set:
            nodeIdxs_path = groupPath.extractPath(pathIdx)

            path_xy = []
            alphas = []
            for nodeIdx in nodeIdxs_path:
                node = groupPath.nodeMap[nodeIdx]
                path_xy.append([node.x, node.y])
                alphas.append(node.alpha)

            path_xy = np.array(path_xy)
            plt.plot(path_xy[:, 0], path_xy[:, 1], '-*')
            # plt.scatter(path_xy[:, 0], path_xy[:, 1], s=10.0)
            for (x, y), alpha in zip(path_xy, alphas):
                plt.arrow(x, y, 0.3 * np.cos(alpha), 0.3 * np.sin(alpha))
    plt.show()

print('Finish')
