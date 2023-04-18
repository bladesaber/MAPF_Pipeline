import open3d as o3d
import numpy as np

from scripts.continueAstar.instance import Instance
from scripts.continueAstar.constrainTable import ConstrainTable
from scripts.continueAstar.hybridAstar import HybridAstarWrap
from scripts.utils import polar2RotMatrix

class VisulizerO3d_HybridA(object):
    def __init__(self, model:HybridAstarWrap, constrainTable:ConstrainTable):
        self.model = model
        self.constrainTable = constrainTable
        self.arrowTable = {}
        
        self.current_rgb = np.array([1.0, 0.0, 0.0])
        self.notExpanded_rgb = np.array([0.0, 1.0, 0.0])
        self.expanded_rgb = np.array([0.0, 0.0, 1.0])
        self.start_rgb = np.array([0.0, 0.0, 0.0])
        self.goal_rgb = np.array([0.0, 1.0, 1.0])
        self.findValidDubinsPath_rgb = np.array([1.0, 0.0, 0.0])
        self.findNonValidDubinsPath_rgb = np.array([0.0, 0.0, 1.0])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        self.coodr = self.create_Axis(size=5.0)
        self.vis.add_geometry(self.coodr, reset_bounding_box=True)

        ### ------ Add Start Arrow
        start_arrow = self.create_Arrow(
            self.model.startNode.x, self.model.startNode.y, self.model.startNode.z,
            self.model.startNode.alpha, self.model.startNode.beta
        )
        start_arrow.paint_uniform_color(self.start_rgb)
        self.vis.add_geometry(start_arrow, reset_bounding_box=True)
        self.arrowTable[self.model.startNode.hashTag] = start_arrow

        ### ------ Add goal Arrow
        goal_arrow = self.create_Arrow(
            self.model.goalNode.x, self.model.goalNode.y, self.model.goalNode.z,
            self.model.goalNode.alpha, self.model.goalNode.beta
        )
        goal_arrow.paint_uniform_color(self.goal_rgb)
        self.vis.add_geometry(goal_arrow, reset_bounding_box=True)
        self.arrowTable[self.model.goalNode.hashTag] = goal_arrow

        ### ------ Add constrain
        self.add_constrain()

        ### ------ Add dubins line
        self.dubinsPath = o3d.geometry.PointCloud()
        self.dubins_hideShow = True
        self.vis.add_geometry(self.dubinsPath, reset_bounding_box=False)

        ### ------ Store Last Arrow and Last Res
        self.last_arrow: o3d.geometry.TriangleMesh = None
        self.isFinsih = False
        self.res = None
        ### --------------------------------------------------

        self.vis.register_key_callback(ord(','), self.step_visulize)
        # self.vis.register_key_callback(ord('.'), self.step_dubinsTest)
        # self.vis.register_key_callback(ord('1'), self.step_dubinsTest)

        self.vis.run()
        self.vis.destroy_window()

    def create_Axis(self, size=1.0, origin=np.array([0., 0., 0.])):
        coodr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        return coodr

    def create_Arrow(
            self, x, y, z, alpha, beta,
            cylinder_radius=0.05, cone_radius=0.2, cylinder_height=1.0, cone_height=0.25
        ):
        rot_mat = polar2RotMatrix(alpha, beta)
        arrow: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, 
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
        )
        arrow.rotate(rot_mat, center=np.array([0., 0., 0.]))
        arrow.translate(np.array([x, y, z]))
        return arrow

    def create_Sphere(self, x, y, z, radius):
        sphere: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(np.array([x, y, z]))
        return sphere

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.model.hybridAstar.is_openList_empty() or self.isFinsih:
            return
        
        res = self.model.findPath_step()
        self.res = res

        if res['state'] == 'find_direct_goal':
            self.isFinsih = True
            self.updateCurrentArrow(res['cur_node'])
        
        elif res['state'] == 'searching':
            self.updateCurrentArrow(res['cur_node'])
            for new_node in self.res['nodes']:
                # print('x:%.2f y:%.2f z:%.2f alpha:%.2f beta:%.2f' % (
                #     new_node.x, new_node.y, new_node.z, np.rad2deg(new_node.alpha), np.rad2deg(new_node.beta)
                # ))

                new_arrow = self.create_Arrow(new_node.x, new_node.y, new_node.z, new_node.alpha, new_node.beta)
                self.arrowTable[new_node.hashTag] = new_arrow

                self.vis.add_geometry(new_arrow, reset_bounding_box=False)

        else:
            raise ValueError()

    def updateCurrentArrow(self, cur_node):
        if self.last_arrow is not None:
            self.last_arrow.paint_uniform_color(self.expanded_rgb)
            self.vis.update_geometry(self.last_arrow)
                
        cur_arrow = self.arrowTable[cur_node.hashTag]
        cur_arrow.paint_uniform_color(self.current_rgb)
        self.vis.update_geometry(cur_arrow)
        self.last_arrow = cur_arrow

    def add_constrain(self):
        for x, y, z, radius in self.constrainTable.CT:
            sphere = self.create_Sphere(x, y, z, radius)
            sphere.paint_uniform_color(np.array([0.0, 0.0, 0.0]))
            self.vis.add_geometry(sphere, reset_bounding_box=False)

    '''
    def update_dubinsPath(self, path_xyzs, color:np.array):
        dist = path_xyzs[1:, :] - path_xyzs[:-1, :]
        detail_path = np.concatenate([
            path_xyzs,
            # path_xyzs[:-1, :] + dist * 0.25,
            path_xyzs[:-1, :] + dist * 0.5,
            # path_xyzs[:-1, :] + dist * 0.75,
        ], axis=0)

        self.dubinsPath.points = o3d.utility.Vector3dVector(detail_path)
        color_np = np.tile(color.reshape((1, -1)), [detail_path.shape[0], 1])
        self.dubinsPath.colors = o3d.utility.Vector3dVector(color_np)
        self.vis.update_geometry(self.dubinsPath)

    def step_dubinsTest(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        node = self.res['cur_node']
        
        if node.findValidDubinsPath:
            self.update_dubinsPath(
                np.array(node.dubinsPath3D), self.findValidDubinsPath_rgb
            )
        else:
            self.update_dubinsPath(
                np.array(node.dubinsPath3D), self.findNonValidDubinsPath_rgb
            )

    def hideOrShow_dubinsPath(self):
        if self.dubins_hideShow:
            self.vis.remove_geometry(self.dubinsPath)
            self.dubins_hideShow = False

        else:
            self.vis.add_geometry(self.dubinsPath, reset_bounding_box=False)
            self.dubins_hideShow = True
    '''

if __name__ == '__main__':
    print("[DEBUG]: Simulation Start ......")

    instance = Instance(20, 20, 20, radius=1.5, cell_size=1.0, horizon_discrete_num=24, vertical_discrete_num=12)

    constrainTable = ConstrainTable()
    constrainTable.insert2CT(x=1., y=1., z=1., radius=2.0)
    constrainTable.update_numpy()

    start_pos=(0.0, 15.0, 15.0, np.deg2rad(0.), np.deg2rad(0.))
    goal_pose=(20.0, 5.0, 5.0, np.deg2rad(0.), np.deg2rad(0.))

    model = HybridAstarWrap(
        radius=0.5, 
        instance=instance, 
        constrainTable=constrainTable
    )
    model.findPath_init(start_pos=start_pos, goal_pose=goal_pose)

    vis = VisulizerO3d_HybridA(model, constrainTable)
    
