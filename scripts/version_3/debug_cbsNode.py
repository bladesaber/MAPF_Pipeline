import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

def debug_agentInfo():
    agent = mapf_pipeline.AgentInfo(agentIdx=0, radius=0.5)
    print(agent.agentIdx, agent.radius)

    detailPath = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0, 2.0),
        (3.0, 0.0, 0.0, 3.0),
        (3.0, 1.0, 0.0, 4.0),
        (3.0, 2.0, 0.0, 5.0),
        (3.0, 3.0, 0.0, 6.0)
    ]
    constrains = [
        (4.0, 4.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.5)
    ]

    agent.update_Constrains(constrains)
    print(agent.getConstrains())

    agent.update_DetailPath_And_Tree(detailPath)
    print(agent.getDetailPath())

def debug_CBSNode_Conflict():
    cbsNode = mapf_pipeline.CBSNode(num_of_agents=2)

    detailPath_1 = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0, 2.0),
        (3.0, 0.0, 0.0, 3.0),
        (3.0, 1.0, 0.0, 4.0),
        (3.0, 2.0, 0.0, 5.0),
        (3.0, 2.8, 0.0, 6.0)
    ]
    agent1 = mapf_pipeline.AgentInfo(agentIdx=0, radius=0.5)
    agent1.update_DetailPath_And_Tree(detailPath_1)
    cbsNode.setAgentInfo(agent1.agentIdx, agent1)

    detailPath_2 = [
        (0.0, 3.0, 0.0, 0.0),
        (1.0, 3.0, 0.0, 1.0),
        (2.0, 3.0, 0.0, 2.0),
        (3.0, 3.0, 0.0, 3.0),
        (4.0, 3.0, 0.0, 4.0),
        (5.0, 3.0, 0.0, 5.0),
        (6.0, 3.0, 0.0, 6.0)
    ]
    agent2 = mapf_pipeline.AgentInfo(agentIdx=1, radius=0.5)
    agent2.update_DetailPath_And_Tree(detailPath_2)
    cbsNode.setAgentInfo(agent2.agentIdx, agent2)

    cbsNode.findAllAgentConflict()

    for key in cbsNode.agentMap.keys():
        agent = cbsNode.agentMap[key]
        agent.info()

    ### After Way 
    print("\n")
    detailPath_1_after = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0, 2.0),
        (3.0, 0.0, 0.0, 3.0),
        (3.0, 1.0, 0.0, 4.0),
        (3.0, 2.0, 0.0, 5.0),
        (4.0, 4.0, 0.0, 8.0)
    ]
    cbsNode.update_DetailPath_And_Tree(agentIdx=0, path=detailPath_1_after)
    cbsNode.findAllAgentConflict()

    for key in cbsNode.agentMap.keys():
        agent = cbsNode.agentMap[key]
        agent.info()

    # cbsNode.compute_Gval()
    # print("g_val: ", cbsNode.g_val)
    # cbsNode.compute_Heuristics()

    print('finishsd')

debug_CBSNode_Conflict()
