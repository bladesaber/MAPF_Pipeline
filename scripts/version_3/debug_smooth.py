import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

from scripts import vis_utils

cond_param = {
    'x': 10,
    'y': 10,
    'z': 10
}

instance = mapf_pipeline.Instance(
    cond_param['x'], cond_param['y'], cond_param['z']
)
cbs = mapf_pipeline.CBS()

smoother = mapf_pipeline.Smoother()
smoother.kappaMax = 1.0 / 3.0
smoother.alpha = 0.1
smoother.gradMax = 1.0

# fakePath_xyz = [
#     (0, 2, 2),
#     (4, 2, 2),
#     (4, 7, 7),
#     (9, 7, 7)
# ]
fakePath_xyz = [
    (0, 2, 0),
    (4, 2, 0),
    (4, 7, 0),
    (9, 7, 0)
]
fakePath_loc = []
for xyz in fakePath_xyz:
    loc = instance.linearizeCoordinate(xyz)
    fakePath_loc.append(loc)

stepLength = 0.5
startShift = (stepLength, 0., 0.)
endShift = (stepLength, 0., 0.)
detailPath = cbs.sampleDetailPath(fakePath_loc, instance, stepLength)
detailPath = smoother.paddingPath(detailPath, startShift, endShift)

# detailPath_np = np.array(detailPath)
# ax = vis_utils.create_Graph3D(xmax=cond_param['x'], ymax=cond_param['y'], zmax=cond_param['z'])
# ax.scatter(detailPath_np[:, 0], detailPath_np[:, 1], detailPath_np[:, 2])
# plt.show()

smoother.addAgentObj(agentIdx=0, radius=0.5, detailPath=detailPath)

# smoother.wSmoothness = 1.0
# smoother.wCurvature = 0.0
# smoother.wObstacle = 0.0

def show(old_path, new_path):
    # ax = vis_utils.create_Graph3D(xmax=cond_param['x'], ymax=cond_param['y'], zmax=cond_param['z'])
    # ax.scatter(detailPath_np[:, 0], detailPath_np[:, 1], detailPath_np[:, 2], c='r', s=6.0)
    # ax.scatter(new_detailPath_np[:, 0], new_detailPath_np[:, 1], new_detailPath_np[:, 2], c='g')

    plt.scatter(old_path[:, 0], old_path[:, 1], c='r', s=5.0)
    plt.scatter(new_path[:, 0], new_path[:, 1], c='g')

    plt.show()

def extractPath(pathXYZ):
    detailPath = []
    for i in pathXYZ:
        x = i.getX()
        y = i.getY()
        z = i.getZ()
        detailPath.append([x, y, z])
    detailPath = np.array(detailPath)
    return detailPath

oldPath = np.array(detailPath)
for _ in range(10):
    smoother.wSmoothness = 1.0
    smoother.wCurvature = 0.0
    smoother.smoothPath(updateTimes=30)
    
    newPath = extractPath(smoother.agentMap[0].pathXYZ)
    show(oldPath, newPath)
    oldPath = newPath

    ### -------------------------------------
    smoother.wSmoothness = 1.0
    smoother.wCurvature = 0.2
    smoother.smoothPath(updateTimes=30)

    newPath = extractPath(smoother.agentMap[0].pathXYZ)
    show(oldPath, newPath)
    oldPath = newPath


print("finish")