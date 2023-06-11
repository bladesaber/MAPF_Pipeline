
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import patches

from build import mapf_pipeline

path_xyzr_3 = [
    (0.0, 10.0, 10.0, 0.5),
    (0.0, 9.0, 10.0, 0.5),
    (0.0, 8.0, 10.0, 0.5),
    (0.0, 7.0, 10.0, 0.5),
    (0.0, 6.0, 10.0, 0.5),
    (0.0, 5.0, 10.0, 0.5),
    (0.0, 4.0, 10.0, 0.5),
    (0.0, 3.0, 10.0, 0.5),
    (0.0, 2.0, 10.0, 0.5),
    (0.0, 1.0, 10.0, 0.5),
    (0.0, 0.0, 10.0, 0.5),

    (1.0, 0.0, 10.0, 0.5),
    (2.0, 0.0, 10.0, 0.5),
    (3.0, 0.0, 10.0, 0.5),
    (4.0, 0.0, 10.0, 0.5),
    (5.0, 0.0, 10.0, 0.5),
    (6.0, 0.0, 10.0, 0.5),
    (7.0, 0.0, 10.0, 0.5),
    (8.0, 0.0, 10.0, 0.5),
    (9.0, 0.0, 10.0, 0.5),
    (10.0, 0.0, 10.0, 0.5),

    (10.0, 0.0, 9.0, 0.5),
    (10.0, 0.0, 8.0, 0.5),
    (10.0, 0.0, 7.0, 0.5),
    (10.0, 0.0, 6.0, 0.5),
    (10.0, 0.0, 5.0, 0.5),
    (10.0, 0.0, 4.0, 0.5),
    (10.0, 0.0, 3.0, 0.5),
    (10.0, 0.0, 2.0, 0.5),
    (10.0, 0.0, 1.0, 0.5),
    (10.0, 0.0, 0.0, 0.5),

]
startDire3 = (-1.57, 0.0)
endDire3 = (0.0, -1.57)

# ### --------- debug extractPath
# groupPath = mapf_pipeline.GroupPath(0)
# groupPath.insertPath(path_xyzr_0)
# groupPath.insertPath(path_xyzr_1)

# pathIdxs = groupPath.extractPath(
#     start_x=0.0, start_y=0.0, start_z=0.0,
#     end_x=4.0, end_y=10.0, end_z=0.0
# )

# path_xyz = []
# for idx in pathIdxs:
#     node = groupPath.pathNodeMap[idx]
#     path_xyz.append([node.x, node.y, node.z])
# path_xyz = np.array(path_xyz)
# print(path_xyz)
# ### ----------------------------------------------------------

smoother = mapf_pipeline.SmootherXYZG2O()
smoother.initOptimizer()

xmin=-0.5
ymin=-0.5
zmin=0
xmax=10.5
ymax=10.5
zmax=2.0
smoother.setBoundary(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

# path_xyzr_3_tem = path_xyzr_3.copy()
# path_xyzr_3 = []
# for (x, y, z, r, l) in mapf_pipeline.sampleDetailPath(path_xyzr_3_tem, 0.5):
#     path_xyzr_3.append((x, y, z, r))
smoother.add_Path(0, path_xyzr_3)

success = smoother.add_OptimizePath(
    groupIdx=0, 
    pathIdx=0,
    start_x=path_xyzr_3[0][0], 
    start_y=path_xyzr_3[0][1], 
    start_z=path_xyzr_3[0][2],
    end_x=path_xyzr_3[-1][0], 
    end_y=path_xyzr_3[-1][1], 
    end_z=path_xyzr_3[-1][2],
    startDire=startDire3,
    endDire=endDire3,
    startFlexRatio=0.0, endFlexRatio=0.0
)

smoother.setMaxRadius(0, 0.5)

groupPath = smoother.groupMap[0]
for pathIdx in groupPath.graphPathMap.keys():
    print(groupPath.graphPathMap[pathIdx])

### -------------------------------------
for outer_i in range(100):

    ### Step 2.1 Build Graph 
    success = smoother.build_graph(
        elasticBand_weight=0.1,
        kinematic_weight=10.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
        boundary_weight=0.0
    )
    # print("build Graph:", success)

    # if outer_i == 0:
    #     smoother.info()
    #     smoother.loss_info(
    #         elasticBand_weight=0.0,
    #         kinematic_weight=0.0,
    #         obstacle_weight=0.0,
    #         pipeConflict_weight=0.0,
    #         boundary_weight=1.0
    #     )
    #     print()

    ### Step 2.2 Optimize
    smoother.optimizeGraph(10, False)

    # smoother.loss_info(
    #     elasticBand_weight=0.0,
    #     kinematic_weight=0.0,
    #     obstacle_weight=0.0,
    #     pipeConflict_weight=0.0,
    #     boundary_weight=1.0
    # )

    ### Step 3.3 Update Vertex to Node
    smoother.update2groupVertex()

    # ### Step 2.4 Clear Graph
    smoother.clear_graph()

    ### ------ Debug Vis path
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-1, 11)
    ax.set_ylim3d(-1, 11)
    ax.set_zlim3d(-1, 11)

    colors = np.random.uniform(0.0, 1.0, (5, 3))
    groupPath = smoother.groupMap[0]

    for pathIdx in groupPath.graphPathMap.keys():
        nodeIdxs_path = groupPath.graphPathMap[pathIdx]

        path_xyz = []
        for idx, nodeIdx in enumerate(nodeIdxs_path):
            node = groupPath.graphNodeMap[nodeIdx]
            path_xyz.append([node.x, node.y, node.z])
                
            if (node.fixed):
                dz = math.sin(node.theta)
                dl = math.cos(node.theta)
                dx = dl * math.cos(node.alpha)
                dy = dl * math.sin(node.alpha)
                ax.quiver(
                    node.x, node.y, node.z, 
                    dx, dy, dz, 
                    length=2.0, normalize=True, color='r'
                )

        path_xyz = np.array(path_xyz)
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], '*-', c=(0.0, 0.0, 1.0))
    
    plt.show()

print('Finsih')