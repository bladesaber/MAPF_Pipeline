import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

from build import mapf_pipeline

model = mapf_pipeline.SmootherXYZG2O()

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
# path0 = [
#     (0.0, 0.0, 0.0, 0.5),
#     (1.0, 0.0, 0.0, 0.5),
#     (2.0, 0.0, 0.0, 0.5),
#     (3.0, 0.0, 0.0, 0.5),
#     (3.0, 1.0, 0.0, 0.5),
#     (3.0, 2.0, 0.0, 0.5),
#     (3.0, 3.0, 0.0, 0.5),
#     (3.0, 4.0, 0.0, 0.5),
#     (4.0, 4.0, 0.0, 0.5),
#     (5.0, 4.0, 0.0, 0.5),
#     (6.0, 4.0, 0.0, 0.5),
#     (7.0, 4.0, 0.0, 0.5),
#     (8.0, 4.0, 0.0, 0.5),
# ]
# startDire0 = (0.0, 0.0)
# endDire0 = (0.0, 0.0)
path0 = model.detailSamplePath(path0, 0.5)

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
path1 = model.detailSamplePath(path1, 0.5)

path2 = [
    (6.0, 0.0, 0.0, 0.5),
    (5.0, 0.0, 0.0, 0.5),
    (4.0, 0.0, 0.0, 0.5),
    (3.5, 0.0, 0.0, 0.5),
    (3.5, 1.0, 0.0, 0.5),
    (3.5, 2.0, 0.0, 0.5),
    (4.5, 3.0, 0.0, 0.5),
    (4.5, 4.0, 0.0, 0.5),
    (4.5, 4.0, 0.0, 0.5),
]
startDire2 = (np.deg2rad(180.0), 0.0)
endDire2 = (np.deg2rad(90.0), 0.0)
path2 = model.detailSamplePath(path2, 0.5)

### Step 1 Add Paths
model.addPath(
    groupIdx=0, pathIdx=0, path_xyzr=path0, startDire=startDire0, endDire=endDire0
)
# model.addPath(
#     groupIdx=0, pathIdx=1, path_xyzr=path1, startDire=startDire1, endDire=endDire1
# )
model.addPath(
    groupIdx=1, pathIdx=0, path_xyzr=path2, startDire=startDire2, endDire=endDire2
)

# for group_key in model.groupMap.keys():
#     groupPath = model.groupMap[group_key]
#     for key in groupPath.nodeMap.keys():
#         node = groupPath.nodeMap[key]
#         print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
#             node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
#         ))

elasticBand_weight = 0.02
kinematic_weight = 1.0
obstacle_weight = 1.0
pipeConflict_weight = 1.0

# obs_df = pd.DataFrame({
#     'x': [2.0, 4.0],
#     'y': [1.0, 3.0],
#     'z': [0.0, 0.0],
#     'radius': [0.0, 0.0],
#     'alpha':  [0.0, 0.0],
#     'theta':  [0.0, 0.0],
# })
obs_df = pd.DataFrame({
    'x': [1.5, 5.0],
    'y': [1.0, 1.5],
    'z': [0.0, 0.0],
    'radius': [0.0, 0.0],
    'alpha':  [0.0, 0.0],
    'theta':  [0.0, 0.0],
})
for idx, row in obs_df.iterrows():
    model.insertStaticObs(row.x, row.y, row.z, row.radius, row.alpha, row.theta)


model.initOptimizer()
for outer_i in range(10):
    
    ### Debug Check Param Before Optimize
    # for group_key in model.groupMap.keys():
    #     groupPath = model.groupMap[group_key]
    #     for key in groupPath.nodeMap.keys():
    #         node = groupPath.nodeMap[key]
    #         print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
    #             node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
    #         ))

    ### Step 2.1 Build Graph 
    model.build_graph(
        elasticBand_weight=elasticBand_weight,
        kinematic_weight=kinematic_weight,
        obstacle_weight=obstacle_weight,
        pipeConflict_weight=pipeConflict_weight
    )

    if outer_i == 0:
        model.info()
        # model.loss_info(
        #     elasticBand_weight=elasticBand_weight,
        #     kinematic_weight=kinematic_weight,
        #     obstacle_weight=obstacle_weight,
        #     pipeConflict_weight=pipeConflict_weight
        # )
    
    ### Step 2.2 Optimize
    model.optimizeGraph(1, False)

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

    # print('------------------------------')
    # model.loss_info(
    #     elasticBand_weight=elasticBand_weight,
    #     kinematic_weight=kinematic_weight,
    #     obstacle_weight=obstacle_weight,
    #     pipeConflict_weight=pipeConflict_weight
    # )

    # ### Step 2.4 Clear Graph
    model.clear_graph()

    ### Debug Vis
    figure, axes = plt.subplots()
    for group_key in model.groupMap.keys():
        groupPath = model.groupMap[group_key]
        for pathIdx in groupPath.pathIdxs_set:
            nodeIdxs_path = groupPath.extractPath(pathIdx)

            path_xy = []
            for idx, nodeIdx in enumerate(nodeIdxs_path):
                node = groupPath.nodeMap[nodeIdx]
                path_xy.append([node.x, node.y])

                if (idx == 0) or (idx == len(nodeIdxs_path)-1):
                    axes.arrow(node.x, node.y, 0.3 * np.cos(node.alpha), 0.3 * np.sin(node.alpha), head_width=0.1)
                
                axes.add_patch( patches.Circle((node.x, node.y), radius=node.radius, fill=False) )

            path_xy = np.array(path_xy)
            axes.plot(path_xy[:, 0], path_xy[:, 1], '-*')
            # plt.scatter(path_xy[:, 0], path_xy[:, 1], s=10.0)

    for idx, row in obs_df.iterrows():
        # plt.scatter(obs_df['x'].values, obs_df['y'].values, s=5.0, c='r')
        axes.add_patch( patches.Circle((row.x, row.y), radius=0.5, fill=False) )
    
    plt.axis("equal")
    plt.show()

print('Finish')