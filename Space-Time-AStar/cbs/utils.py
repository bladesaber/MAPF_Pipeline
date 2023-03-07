from typing import Dict, Tuple, Set
import numpy as np
from copy import deepcopy

class Agent:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    def __hash__(self):
        return int(str(self.start[0]) + str(self.start[1]))

    def __eq__(self, other: 'Agent'):
        return np.array_equal(self.start, other.start) and np.array_equal(self.goal, other.goal)

    def __str__(self):
        return str(self.start.tolist())

    def __repr__(self):
        return self.__str__()

class Constraints:
    def __init__(self):
        #                                   time,         obstacles
        self.agent_constraints: Dict[Agent: Dict[int, Set[Tuple[int, int]]]] = dict()

    '''
    Deepcopy self with additional constraints
    '''
    def fork(self, agent: Agent, obstacle: Tuple[int, int], start: int, end: int) -> 'Constraints':
        agent_constraints_copy = deepcopy(self.agent_constraints)
        for time in range(start, end):
            agent_constraints_copy.setdefault(agent, dict()).setdefault(time, set()).add(obstacle)

        new_constraints = Constraints()
        new_constraints.agent_constraints = agent_constraints_copy
        return new_constraints

    def setdefault(self, key, default):
        return self.agent_constraints.setdefault(key, default)

    def __getitem__(self, agent):
        return self.agent_constraints[agent]

    def __iter__(self):
        for key in self.agent_constraints:
            yield key

    def __str__(self):
        return str(self.agent_constraints)

class CTNode:
    def __init__(
            self,
            constraints: Constraints,
            solution: Dict[Agent, np.ndarray]
    ):
        self.constraints = constraints
        self.solution = solution
        self.cost = self.sic(solution)

    # Sum-of-Individual-Costs heuristics
    @staticmethod
    def sic(solution):
        return sum(len(sol) for sol in solution.items())

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return str(self.constraints.agent_constraints)
