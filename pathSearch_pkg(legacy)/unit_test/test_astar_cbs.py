from build import MapfPkg
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def visualize_2d_path(grid, path, starts, goals, radius=0.5, padding=2, save_path=None):
    """Visualize a 2D path with obstacles, starts, and goals using matplotlib"""
    # Get grid dimensions
    width = grid.get_x_size()
    height = grid.get_y_size()
    
    # Find bounds including path, starts, and goals
    min_x = max(0, min([p.x for p in path] + [p.x for p in starts] + [p.x for p in goals]) - padding)
    max_x = min(width - 1, max([p.x for p in path] + [p.x for p in starts] + [p.x for p in goals]) + padding)
    min_y = max(0, min([p.y for p in path] + [p.y for p in starts] + [p.y for p in goals]) - padding)
    max_y = min(height - 1, max([p.y for p in path] + [p.y for p in starts] + [p.y for p in goals]) + padding)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot grid and obstacles
    for y in range(int(min_y), int(max_y + 1)):
        for x in range(int(min_x), int(max_x + 1)):
            if grid.is_obstacle(MapfPkg.LocXyz(x, y, 0), radius):
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray'))
    
    # Plot path
    path_x = [p.x for p in path]
    path_y = [p.y for p in path]
    plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    # Plot start points
    start_x = [p.x for p in starts]
    start_y = [p.y for p in starts]
    plt.plot(start_x, start_y, 'go', markersize=10, label='Start')
    
    # Plot goal points
    goal_x = [p.x for p in goals]
    goal_y = [p.y for p in goals]
    plt.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Set plot properties
    plt.grid(True)
    plt.legend()
    plt.xlim(min_x - 0.5, max_x + 0.5)
    plt.ylim(min_y - 0.5, max_y + 0.5)
    plt.title('Path Planning Visualization')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def visualize_3d_path(grid, paths_dict, radius=0.5):
    """Visualize 3D paths with obstacles using PyVista"""
    # Create PyVista plotter
    plotter = pv.Plotter()
    
    # Set up grid dimensions
    x_size = grid.get_x_size()
    y_size = grid.get_y_size()
    z_size = grid.get_z_size()
    
    # Create color map for different paths
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths_dict)))
    
    # Plot obstacles
    obstacle_points = []
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                loc = MapfPkg.LocXyz(x, y, z)
                if grid.is_obstacle(loc, radius):
                    obstacle_points.append([x, y, z])
    
    if obstacle_points:
        obstacles = pv.PolyData(np.array(obstacle_points))
        plotter.add_mesh(obstacles.glyph(geom=pv.Cube()), color='gray', opacity=0.5)
    
    # Plot paths
    for (group_name, path_data), color in zip(paths_dict.items(), colors):
        points = []
        for path in path_data.paths:
            # Create path points
            path_points = [[p.x, p.y, p.z] for p in path.path]
            points.extend(path_points)
            
            # Create line through path points
            if len(path_points) > 1:
                line = pv.Line(path_points[0], path_points[-1])
                plotter.add_mesh(line, color=color[:3], line_width=3, label=group_name)
            
            # Add spheres for start and end points
            start_sphere = pv.Sphere(radius=radius, center=path_points[0])
            end_sphere = pv.Sphere(radius=radius, center=path_points[-1])
            plotter.add_mesh(start_sphere, color='green')
            plotter.add_mesh(end_sphere, color='red')
    
    # Set up camera and display
    plotter.show_grid()
    plotter.show_axes()
    plotter.add_legend()
    plotter.show()


class AstarTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.grid_3d = MapfPkg.StandardGrid(10, 10, 10, 0.1)
        self.grid_2d = MapfPkg.StandardGrid(10, 10, 1, 0.1)
        self.astar = MapfPkg.AstarSearchBoost(0.5, 100000, False)
        self.weighted_astar = MapfPkg.WeightedAstarSearchBoost(0.5, 100000, 1.0, 1.0, False)
        self.theta_star = MapfPkg.ThetaStarSearchBoost(0.5, 100000, 1.0, 1.0)
        self.bidirectional_astar = MapfPkg.BiDirectionalAstarSearchBoost(0.5, 100000, 1.0, 1.0)

    def test_simple_pathfinding_2d(self):
        """Test basic 2D pathfinding"""
        starts = [MapfPkg.LocXyz(6, 8, 0)]
        goals = [MapfPkg.LocXyz(1, 1, 0)]
        path_store = MapfPkg.PyWrapAstarStruct(self.grid_2d, self.astar)
        
        self.assertTrue(path_store.run(starts, goals))
        self.assertGreater(len(path_store.path), 0)
        self.assertEqual(path_store.path[0].x, starts[0].x)
        self.assertEqual(path_store.path[0].y, starts[0].y)
        self.assertEqual(path_store.path[-1].x, goals[0].x)
        self.assertEqual(path_store.path[-1].y, goals[0].y)
        # visualize_2d_path(self.grid_2d, path_store.path, starts, goals)

    def test_simple_pathfinding_3d(self):
        """Test basic 3D pathfinding"""
        starts = [MapfPkg.LocXyz(1, 1, 1)]
        goals = [MapfPkg.LocXyz(8, 8, 8)]
        path_store = MapfPkg.PyWrapAstarStruct(self.grid_3d, self.astar)
        
        self.assertTrue(path_store.run(starts, goals))
        self.assertGreater(len(path_store.path), 0)

    def test_blocked_path(self):
        """Test pathfinding with obstacles"""
        # Create wall of obstacles
        for i in range(10):
            for j in range(10):
                self.grid_3d.set_obstacle(MapfPkg.LocXyz(5, i, j), 1.0)
        
        starts = [MapfPkg.LocXyz(1, 5, 5)]
        goals = [MapfPkg.LocXyz(8, 5, 5)]
        path_store = MapfPkg.PyWrapAstarStruct(self.grid_3d, self.astar)
        self.assertFalse(path_store.run(starts, goals))

    def test_theta_star_pathfinding(self):
        """Test Theta* pathfinding"""
        starts = [MapfPkg.LocXyz(1, 1, 0)]
        goals = [MapfPkg.LocXyz(8, 8, 0)]
        path_store = MapfPkg.PyWrapAstarStruct(self.grid_2d, self.theta_star)
        
        self.assertTrue(path_store.run(starts, goals))
        self.assertGreater(len(path_store.path), 0)
        # visualize_2d_path(self.grid_2d, path_store.path, starts, goals)

    def test_bidirectional_astar(self):
        """Test Bidirectional A* pathfinding"""
        starts = [MapfPkg.LocXyz(1, 1, 0)]
        goals = [MapfPkg.LocXyz(8, 8, 0)]
        path_store = MapfPkg.PyWrapAstarStruct(self.grid_2d, self.bidirectional_astar)
        
        self.assertTrue(path_store.run(starts, goals))
        self.assertGreater(len(path_store.path), 0)
        # visualize_2d_path(self.grid_2d, path_store.path, starts, goals)


class CbsTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create grid
        self.grid = MapfPkg.TreeGrid(10, 10, 10, 1.0)
        
        # Create CBS instance
        self.cbs = MapfPkg.CbsSearchNearEnd(self.grid, 1.0, 1.0, 1000)
        
        # Create planners and task sets for different groups
        # Group 1: Moving vertically
        self.astar1 = MapfPkg.AstarSearchBoost(0.5, 100000, False)
        self.planner1 = MapfPkg.SetPairNodePlanner(self.astar1)
        self.task_set1 = MapfPkg.PairNodeTaskSet("group1")
        self.task_set1.add_task(
            "task1", "start1", "end1",
            [MapfPkg.LocXyz(5, 0, 5)],
            [MapfPkg.LocXyz(5, 9, 5)],
            0.5
        )
        self.cbs.add_task_set(self.task_set1, self.planner1)
        
        # Group 2: Moving horizontally
        self.astar2 = MapfPkg.AstarSearchBoost(0.5, 100000, False)
        self.planner2 = MapfPkg.SetPairNodePlanner(self.astar2)
        self.task_set2 = MapfPkg.PairNodeTaskSet("group2")
        self.task_set2.add_task(
            "task2", "start2", "end2",
            [MapfPkg.LocXyz(0, 5, 5)],
            [MapfPkg.LocXyz(9, 5, 5)],
            0.5
        )
        self.cbs.add_task_set(self.task_set2, self.planner2)
        
        # Group 3: Moving in z-direction
        self.astar3 = MapfPkg.AstarSearchBoost(0.5, 100000, False)
        self.planner3 = MapfPkg.SetPairNodePlanner(self.astar3)
        self.task_set3 = MapfPkg.PairNodeTaskSet("group3")
        self.task_set3.add_task(
            "task3", "start3", "end3",
            [MapfPkg.LocXyz(5, 5, 0)],
            [MapfPkg.LocXyz(5, 5, 9)],
            0.5
        )
        self.cbs.add_task_set(self.task_set3, self.planner3)

    def test_grid_properties(self):
        """Test basic grid properties"""
        self.assertEqual(self.grid.get_x_size(), 10)
        self.assertEqual(self.grid.get_y_size(), 10)
        self.assertEqual(self.grid.get_z_size(), 10)
        self.assertEqual(self.grid.get_grid_resolution(), 1.0)
        self.assertTrue(self.grid.is_in_grid(MapfPkg.LocXyz(0, 0, 0)))
        self.assertFalse(self.grid.is_in_grid(MapfPkg.LocXyz(10, 10, 10)))

    def test_cbs_search(self):
        """Test CBS search with multiple groups"""
        # Create CBS wrapper
        cbs_store = MapfPkg.PyWrapCbsStruct(self.cbs)
        
        # Run CBS search
        self.assertTrue(cbs_store.run())
        
        # Verify results
        self.assertGreater(len(cbs_store.result.paths), 0)
        
        # Check each group's path
        for group_name, path_set in cbs_store.result.paths.items():
            self.assertGreater(path_set.get_size(), 0)
            for path in path_set.paths:
                # Verify path connects start to goal
                self.assertGreater(len(path.path), 0)
                if group_name == "group1":
                    self.assertEqual(path.path[0].x, 5)
                    self.assertEqual(path.path[0].y, 0)
                    self.assertEqual(path.path[-1].x, 5)
                    self.assertEqual(path.path[-1].y, 9)
                elif group_name == "group2":
                    self.assertEqual(path.path[0].x, 0)
                    self.assertEqual(path.path[0].y, 5)
                    self.assertEqual(path.path[-1].x, 9)
                    self.assertEqual(path.path[-1].y, 5)
                
                # # Visualize 2D paths
                # if path.path[0].z == path.path[-1].z:  # Only visualize paths in same z-plane
                #     starts = [path.path[0]]
                #     goals = [path.path[-1]]
                #     visualize_2d_path(self.grid, path.path, starts, goals)

if __name__ == '__main__':
    unittest.main()
