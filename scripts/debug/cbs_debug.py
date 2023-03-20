import numpy as np
import multiprocessing

from build import mapf_pipeline

cond_params = {
    'row': 5,
    'col': 5,
    'z': 5,
    'num_of_agents': 5,

    'load_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/map.npy',
    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/map',
    "load": True,
}

def getMap(num_of_agents):
    instance = mapf_pipeline.Instance3D(cond_params['row'], cond_params['col'], cond_params['z'])

    start_states = {}
    goal_states = {}

    if cond_params['load']:
        states = np.load(cond_params['load_path'], allow_pickle=True).item()
        start_states = states['start_states']
        goal_states = states['goal_states']

    else:
        records = []
        for agent_idx in range(num_of_agents):
            while True:
                start_yxz = [
                    np.random.randint(0, cond_params['row']),
                    np.random.randint(0, cond_params['col']),
                    np.random.randint(0, cond_params['z'])
                ]
            
                cut = np.random.uniform(0.0, 1.0)
                if cut <= 1.0 / 3.0:
                    start_yxz[0] = 0
                elif cut > 2.0 / 3.0:
                    start_yxz[1] = 0
                else:
                    start_yxz[2] = 0
            
                loc = instance.linearizeCoordinate(start_yxz)
                if loc not in records:
                    records.append(loc)
                    start_states[agent_idx] = tuple(start_yxz)
                    break
            
            while True:
                goal_yxz = [
                    np.random.randint(0, cond_params['row']),
                    np.random.randint(0, cond_params['col']),
                    np.random.randint(0, cond_params['z'])
                ]

                cut = np.random.uniform(0.0, 1.0)
                if cut <= 1.0 / 3.0:
                    goal_yxz[0] = 0
                elif cut > 2.0 / 3.0:
                    goal_yxz[1] = 0
                else:
                    goal_yxz[2] = 0
                
                loc = instance.linearizeCoordinate(goal_yxz)
                if loc not in records:
                    records.append(loc)
                    goal_states[agent_idx] = tuple(goal_yxz)
                    break
        
        np.save(
            cond_params['save_path'], 
            {
                'start_states': start_states,
                'goal_states': goal_states
            }
        )

    return instance, start_states, goal_states

def easy_debug():
    num_of_agents = cond_params['num_of_agents']
    instance, start_states, goal_states = getMap(num_of_agents=num_of_agents)
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
    def __init__(self, num_of_agents, instance, start_states, goal_states):
        self.num_of_agents = num_of_agents
        self.instance = instance
        self.start_states = start_states
        self.goal_states = goal_states
        self.cbs_planner = mapf_pipeline.CBS(num_of_agents, instance, start_states, goal_states)

    def solve(self):
        # cbs_planner.print()

        root = mapf_pipeline.CBSNode()
        for agent_idx in range(self.num_of_agents):
            self.cbs_planner.solvePath(root, agent_idx)
            # print("runtime_search:%f, runtime_build_CT:%f, runtime_build_CAT:%f" % (
                # root.runtime_search, root.runtime_build_CT, root.runtime_build_CAT
            # ))

        if not self.check_valid(root):
            return None

        root.findConflicts()
        # root.updateMakespan()
        root.updateGval()
        self.cbs_planner.pushNode(root)

        success_node = None
        while not self.cbs_planner.is_openList_empty():
            self.cbs_planner.updateFocalList()
            node = self.cbs_planner.popNode()

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

            constrains = [select_conflict.constraint1, select_conflict.constraint2]
            child_nodes = []
            for constrain in constrains:
                agent_idx = constrain[0]
                child_node = mapf_pipeline.CBSNode()
                child_node.copy(node)
                child_node.insertConstraint(agent=agent_idx, constraint=constrain)

                self.cbs_planner.solvePath(child_node, agent_idx)
                if not self.check_valid(child_node):
                    continue

                child_node.findConflicts()
                # child_node.updateMakespan(agent_idx)
                child_node.updateGval()

                if child_node.g_val <= node.g_val and len(child_node.conflicts) < len(node.conflicts):
                    child_nodes.clear()
                    child_nodes.append(child_node)
                    break

                else:
                    child_nodes.append(child_node)

            for child_node in child_nodes:
                print('[Debug] Node with f_val: %f conflict:%d' % (
                    child_node.getFVal(), len(child_node.conflicts)
                ))
                self.cbs_planner.pushNode(child_node)

        return success_node

    def check_valid(self, node):
        for agent_idx in range(self.num_of_agents):
            if len(node.getPath(agent_idx)) == 0:
                return False
        return True

def main():
    num_of_agents = cond_params['num_of_agents']
    instance, start_states, goal_states = getMap(num_of_agents=num_of_agents)

    cbs_searcher = CBS_Solver(
        num_of_agents, instance, start_states, goal_states
    )
    success_node = cbs_searcher.solve()
    print(success_node)

if __name__ == '__main__':
    # easy_debug()
    main()
