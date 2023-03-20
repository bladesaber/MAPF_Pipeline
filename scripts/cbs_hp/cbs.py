from typing import Dict
import numpy as np
from heapq import heappush, heappop
import multiprocessing

class CBS_Node(object):
    def __init__(self) -> None:
        self.paths = {}
        self.constraints = {}


class CBS_Planner(object):
    pass
