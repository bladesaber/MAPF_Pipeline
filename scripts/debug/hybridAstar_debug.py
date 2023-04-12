import open3d as o3d
import numpy as np

from scripts.continueAstar.instance import Instance
from scripts.continueAstar.constrainTable import ConstrainTable
from scripts.continueAstar.hybridAstar import HybridAstarWrap
from scripts.utils import polar2RotMatrix

class VisulizerO3d_HybridA(object):
    def __init__(self, model:HybridAstarWrap):
        self.model = model
        self.arrowTable = {}
        
        self.current_rgb = np.array([1.0, 0.0, 0.0])
        self.notExpanded_rgb = np.array([0.0, 1.0, 0.0])
        self.expanded_rgb = np.array([0.0, 0.0, 1.0])
        self.start_rgb = np.array([1.0, 1.0, 1.0])
        self.goal_rgb = np.array([0.0, 1.0, 1.0])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(height=720, width=960)

        self.coodr = self.create_Axis()
        self.vis.add_geometry(self.coodr, reset_bounding_box=True)

        ### ------ Add Start Arrow
        start_arrow = self.create_Arrow(
            self.model.startNode.x, self.model.startNode.y, self.model.startNode.z,
            self.model.startNode.alpha, self.model.startNode.beta
        )
        start_arrow.paint_uniform_color(self.start_rgb)
        self.vis.add_geometry(start_arrow, reset_bounding_box=False)
        self.arrowTable[self.model.startNode.hashTag] = start_arrow

        ### ------ Add goal Arrow
        goal_arrow = self.create_Arrow(
            self.model.goalNode.x, self.model.goalNode.y, self.model.goalNode.z,
            self.model.goalNode.alpha, self.model.goalNode.beta
        )
        goal_arrow.paint_uniform_color(self.goal_rgb)
        self.vis.add_geometry(goal_arrow, reset_bounding_box=False)
        self.arrowTable[self.model.goalNode.hashTag] = goal_arrow

        ### ------ Last Arrow
        self.last_arrow: o3d.geometry.TriangleMesh = None
        self.isFinsih = False

        ### --------------------------------------------------

        self.vis.register_key_callback(ord(','), self.step_visulize)

        self.vis.run()
        self.vis.destroy_window()

    def create_Axis(self, size=1.0, origin=np.array([0., 0., 0.])):
        coodr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        return coodr

    def create_Arrow(
            self, x, y, z, alpha, beta,
            cylinder_radius=0.1, cone_radius=0.2, cylinder_height=0.4, cone_height=0.1
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

    def step_visulize(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        if self.model.hybridAstar.is_openList_empty() or self.isFinsih:
            return
        
        res = self.model.findPath_step()
        if res['state'] == 'skip':
            return

        elif res['state'] == 'find_direct_goal':
            self.isFinsih = True
            self.updateCurrentArrow(res['cur_node'])
        
        else:
            self.updateCurrentArrow(res['cur_node'])
            for new_node in res['nodes']:
                new_arrow = self.create_Arrow(
                    new_node.x, new_node.y, new_node.z, new_node.alpha, new_node.beta
                )
                self.arrowTable[new_node.hashTag] = new_arrow

                self.vis.add_geometry(new_arrow, reset_bounding_box=False)
    
    def updateCurrentArrow(self, cur_node):
        if self.last_arrow is not None:
            self.last_arrow.paint_uniform_color(self.expanded_rgb)
            self.vis.update_geometry(self.last_arrow)
                
        cur_arrow = self.arrowTable[cur_node.hashTag]
        cur_arrow.paint_uniform_color(self.current_rgb)
        self.vis.update_geometry(cur_arrow)
        self.last_arrow = cur_arrow

if __name__ == '__main__':
    instance = Instance(40, 40, 40, radius=1.5, cell_size=1.0, horizon_discrete_num=24, vertical_discrete_num=12)

    constrainTable = ConstrainTable()
    constrainTable.insert2CT(x=20, y=20, z=20, radius=1.0)
    constrainTable.update_numpy()

    start_pos=(0.0, 10.0, 5.0, np.deg2rad(0.), np.deg2rad(0.))
    goal_pose=(40.0, 20.0, 30.0, np.deg2rad(0.), np.deg2rad(0.))

    model = HybridAstarWrap(
        radius=0.5, 
        instance=instance, 
        constrainTable=constrainTable
    )
    model.findPath_init(start_pos=start_pos, goal_pose=goal_pose)

    vis = VisulizerO3d_HybridA(model)
    
