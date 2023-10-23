import numpy as np
from typing import Dict, List
import multiprocessing
from copy import copy

from scripts_py.visulizer import VisulizerVista
from build import mapf_pipeline

cond_params = {
    'row': 20,
    'col': 20,
    'z': 20,
    'num_of_agents': 30,

    'load_path': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/map.npy',
    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/map',
    "load": False,
}

def create_state():
    yxz = [
        np.random.randint(0, cond_params['row']),
        np.random.randint(0, cond_params['col']),
        np.random.randint(0, cond_params['z'])
    ]
    yxz_orign = copy(yxz)

    cut = np.random.randint(0, 5)
    if cut == 0:
        yxz[0] = 0
        yxz_orign[0] = -1

    elif cut == 1:
        yxz[0] = cond_params['row'] - 1
        yxz_orign[0] = cond_params['row']

    elif cut == 2:
        yxz[1] = 0
        yxz_orign[1] = -1

    elif cut == 3:
        yxz[1] = cond_params['col'] - 1
        yxz_orign[1] = cond_params['col']

    elif cut == 4:
        yxz[2] = 0
        yxz_orign[2] = -1

    elif cut == 5:
        yxz[2] = cond_params['z'] - 1
        yxz_orign[2] = cond_params['z']

    return yxz, yxz_orign

def getMap(num_of_agents):
    instance = mapf_pipeline.Instance3D(cond_params['row'], cond_params['col'], cond_params['z'])

    start_states = {}
    goal_states = {}
    start_states_from = {}
    goal_states_to = {}

    if cond_params['load']:
        states = np.load(cond_params['load_path'], allow_pickle=True).item()

        start_states = states['start_states']
        start_states_from = states['start_states_from']

        goal_states = states['goal_states']
        goal_states_to = states['goal_states_to']

    else:
        records = []
        for agent_idx in range(num_of_agents):
            while True:
                start_yxz, start_yxz_orign = create_state()
                loc = instance.linearizeCoordinate(start_yxz)

                if loc not in records:
                    records.append(loc)
                    start_states[agent_idx] = tuple(start_yxz)
                    start_states_from[agent_idx] = start_yxz_orign
                    break
            
            while True:
                goal_yxz, goal_yxz_origin = create_state()

                loc = instance.linearizeCoordinate(goal_yxz)
                if loc not in records:
                    records.append(loc)
                    goal_states[agent_idx] = tuple(goal_yxz)
                    goal_states_to[agent_idx] = goal_yxz_origin
                    break
        
        np.save(
            cond_params['save_path'], 
            {
                'start_states': start_states,
                'goal_states': goal_states,
                'start_states_from': start_states_from,
                'goal_states_to': goal_states_to
            }
        )

    return instance, (start_states, start_states_from), (goal_states, goal_states_to)

def easy_debug():
    num_of_agents = cond_params['num_of_agents']
    instance, (start_states, _), (goal_states, _) = getMap(num_of_agents=num_of_agents)
    # print("start states: ", start_states)
    # print("goal states: ", goal_states)

    cbs_planner = mapf_pipeline.CBS(num_of_agents, instance, start_states, goal_states)
    cbs_planner.print()

    root = mapf_pipeline.CBSNode()
    for agent_idx in range(num_of_agents):
        cbs_planner.solvePath(root, agent_idx)
        print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (
            root.runtime_search, root.runtime_build_CT, root.runtime_build_CAT
        ))

    for agent_idx in range(num_of_agents):
        path = root.getPath(agent_idx)
        print(path)

    conflicts = cbs_planner.findConflicts(root)
    for conflict in conflicts:
        print("[Debug] %d(%d) <-> %d(%d) in %d" % (
            conflict.a1, conflict.a2, 
            conflict.a1_timeStep, conflict.a2_timeStep, 
            conflict.loc
            )
        )

class CBS_Solver(object):
    def __init__(self, num_of_agents, instance, start_states:Dict, goal_states:Dict):
        self.num_of_agents = num_of_agents
        self.instance = instance
        self.start_states = start_states
        self.goal_states = goal_states
        self.cbs_planner = mapf_pipeline.CBS(num_of_agents, instance, start_states, goal_states)
        self.agent_idxs = self.start_states.keys()

        self.allNodes = {}
        # 由于CBSNode是由Python构造，必须由Python自己管理
        self.node_id = 0

    def solve(self):
        self.cbs_planner.print()

        root = mapf_pipeline.CBSNode()
        for a1 in self.agent_idxs:
            for a2 in self.agent_idxs:
                if a1 != a2:
                    start_state = self.start_states[a2]
                    start_loc = self.instance.linearizeCoordinate(start_state)
                    root.insertConstraint(a1, (a1, start_loc, 0, mapf_pipeline.constraint_type.VERTEX))

                    goal_state = self.goal_states[a2]
                    goal_loc = self.instance.linearizeCoordinate(goal_state)
                    root.insertConstraint(a1, (a1, goal_loc, 0, mapf_pipeline.constraint_type.VERTEX))

        for agent_idx in range(self.num_of_agents):
            self.cbs_planner.solvePath(root, agent_idx)

        if not self.check_valid(root):
            return None

        root.depth = 0
        root.findConflicts()
        root.updateGval()
        root.updateTiebreaking(len(root.conflicts))
        self.pushNode(root)
        # mapf_pipeline.printPointer(root, "[Debug] push node: ")

        run_times = 1
        success_node = None
        found_bypass = False
        while not self.cbs_planner.is_openList_empty():
            self.cbs_planner.updateFocalList()

            node = self.popNode()
            # mapf_pipeline.printPointer(node, "[Debug] Pop Parent node: ")

            if len(node.conflicts) == 0:
                success_node = node
                break

            select_conflict, minum_conflict_timeStep = None, np.inf
            for conflict in node.conflicts:
                # print("[Debug] %d(%d) <-> %d(%d) in %d" % (
                #     conflict.a1, conflict.a2, conflict.a1_timeStep, conflict.a2_timeStep, conflict.loc
                # ))
                min_timeStep = min(conflict.a1_timeStep, conflict.a2_timeStep)
                if min_timeStep < minum_conflict_timeStep:
                    minum_conflict_timeStep = min_timeStep
                    select_conflict = conflict
            select_conflict.vertexConflict()

            ### ------ debug
            print('Parent Node:')
            self.nodeDescribe(node)
            ### ---------------------

            constrains = [select_conflict.constraint1, select_conflict.constraint2]
            print('Find Constrain: ', constrains)

            child_nodes = []
            for constrain in constrains:
                agent_idx = constrain[0]
                child_node = mapf_pipeline.CBSNode()
                child_node.copy(node)
                child_node.depth = node.depth + 1
                child_node.insertConstraint(agent=agent_idx, constraint=constrain)
                self.cbs_planner.solvePath(child_node, agent_idx)

                if not self.check_valid(child_node):
                    continue
                
                child_node.findConflicts()
                child_node.updateGval()
                child_node.updateTiebreaking(len(child_node.conflicts))

                ### ------ debug
                print('Child node:')
                self.nodeDescribe(child_node, select_idx=agent_idx)
                ### ---------------------

                if child_node.g_val <= node.g_val and len(child_node.conflicts) < len(node.conflicts):
                    child_nodes.clear()
                    child_nodes.append(child_node)
                    found_bypass = True
                    break

                else:
                    child_nodes.append(child_node)

            print('[DEBUG] foundBypass: %d, childNode num:%d' % (found_bypass, len(child_nodes)))
            for child_node in child_nodes:
                ### ------ debug
                # self.nodeDescribe(child_node, detail=False)
                # mapf_pipeline.printPointer(child_node, "[Debug] Push Child node: ")
                ### ---------------------------------

                self.pushNode(child_node)

            run_times += 1
            # if run_times >= 30:
            #     break

        return success_node

    def check_valid(self, node):
        for agent_idx in range(self.num_of_agents):
            if len(node.getPath(agent_idx)) == 0:
                return False
        return True

    def pushNode(self, node):
        node.node_id = self.node_id
        self.node_id += 1
        self.cbs_planner.pushNode(node)
        self.allNodes[node.node_id] = node

    def popNode(self):
        node = self.cbs_planner.popNode()
        del self.allNodes[node.node_id]
        return node

    def nodeDescribe(self, node, detail=True, select_idx=None):
        print('[Debug] Node with f_val: %f conflict:%d depth:%d' % (node.getFVal(), len(node.conflicts), node.depth))
        if detail:
            for agent_idx in self.agent_idxs:
                if select_idx is not None:
                    if agent_idx != select_idx:
                        continue

                path = node.getPath(agent_idx)
                print('Agent %d:' % (agent_idx), ' length:%d'%len(path))
                print('  path: ', path)
                constrains = node.getConstrains(agent_idx)
                constrain_locs = []
                for constrain in constrains:
                    constrain_locs.append(constrain[1])
                print('  constrain(%d) loc: '%(len(constrain_locs)), constrain_locs)
            print('--------------------------------------------')

def main():
    num_of_agents = cond_params['num_of_agents']
    instance, (start_states, start_states_from), (goal_states, goal_states_to) = getMap(num_of_agents=num_of_agents)

    # for i in range(num_of_agents):
    #     print(start_states_from[i], start_states[i])
    #     print(goal_states_to[i], goal_states[i])
    #     print('----------------------------')

    cbs_searcher = CBS_Solver(
        num_of_agents, instance, start_states, goal_states
    )

    print("Starting ...")
    success_node = cbs_searcher.solve()

    random_colors = np.random.uniform(0.0, 1.0, size=(num_of_agents, 3))
    vis = VisulizerVista()
    for i, agent_idx in enumerate(cbs_searcher.agent_idxs):
        path = success_node.getPath(agent_idx)
        xyzs = []

        start_from = start_states_from[agent_idx]
        xyzs.append([start_from[1], start_from[0], start_from[2]])
        for j, loc in enumerate(path):
            (y, x, z) = instance.getCoordinate(loc)
            xyzs.append([x, y, z])

        goal_to = goal_states_to[agent_idx]
        xyzs.append([goal_to[1], goal_to[0], goal_to[2]])

        xyzs = np.array(xyzs)

        tube_mesh = vis.create_tube(xyzs)
        vis.plot(tube_mesh, color=tuple(random_colors[i]))
    vis.show()

def debug_singleSolve():
    num_of_agents = cond_params['num_of_agents']
    instance, start_states, goal_states = getMap(num_of_agents=num_of_agents)

    astar = mapf_pipeline.SpaceTimeAStar(0)
    constrains_table = {0: []}
    locs = [15, 60, 0, 45, 90, 25, 110, 27, 11, 41, 111]
    start_yxz = (0, 1, 4)
    goal_yxz = (4, 1, 0)
    for loc in locs:
        constrains_table[0].append(
            (0, loc, 0, mapf_pipeline.constraint_type.VERTEX)
        )
    
    for _ in range(15):
        path: List = astar.findPath(
            paths={}, constraints=constrains_table, instance=instance, start_state=start_yxz, goal_state=goal_yxz
        )
        print(path)
        # print("num_expanded:%d, num_generated:%d" % (astar.num_expanded, astar.num_generated))
        # print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (
        #     astar.runtime_search, astar.runtime_build_CT, astar.runtime_build_CAT
        # ))

if __name__ == '__main__':
    # easy_debug()
    main()
    # debug_singleSolve()
