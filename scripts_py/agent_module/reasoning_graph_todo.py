import os
from abc import abstractmethod
from typing import Union, Literal

import numpy as np
import pandas as pd


class BaseNode(object):
    def __init__(self, id_tag: Union[str, int]):
        self.id_tag = id_tag
        self.style: Literal["CoT", "DoT"]
        self.observation: str
        self.markov_memory_nodes: list[BaseNode] = []

        self.run_times: int = 0
        self.is_success: bool = False

    @abstractmethod
    def select_child(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs) -> bool:
        raise NotImplementedError


class CoTNode(BaseNode):
    def __init__(self, id_tag):
        super().__init__(id_tag)
        self.style = "CoT"
        self.childs: dict[str, BaseNode] = {}
        self.parent: BaseNode = None

    @abstractmethod
    def select_child(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError

    def is_root(self):
        return self.parent is None

    def _child(self, i: Union[str, int]):
        if isinstance(i, int):
            return self.childs[list(self.childs.keys())[i]]
        elif isinstance(i, str):
            return self.childs[i]
        raise ValueError

    def _parent(self):
        return self.parent

    def num_childs(self):
        return len(self.childs)


class DoTNode(CoTNode):
    def __init__(self, id_tag):
        super().__init__(id_tag)
        self.style = "DoT"
        self.sub_problems: dict[str, BaseNode] = {}


class IncorporationGraph(object):
    def __init__(self):
        self.nodes: dict[Union[str, int], BaseNode] = {}
        self.cur_node_id: Union[str, int]

    def pop_node(self) -> BaseNode:
        return self.nodes[self.cur_node_id]

    def jump_next_node(self, **kwargs) -> dict:
        current_node: Union[CoTNode, DoTNode] = self.pop_node()
        if current_node.style == "CoT":
            if current_node.num_childs == 0:
                parent_node: BaseNode = current_node._parent()
                while (parent_node != None) and (parent_node.style == "CoT"):
                    parent_node: BaseNode = parent_node._parent()

                if parent_node is None:
                    return {"node": current_node.id_tag, "is_finish": True}
                else:
                    self.cur_node_id = parent_node.id_tag
                    return {"node": parent_node.id_tag, "is_finish": False}

            else:
                child_node: BaseNode = current_node.select_child(**kwargs)
                self.cur_node_id = child_node.id_tag
                return {"state": True, "node": child_node.id_tag, "is_finish": False}

        elif current_node.style == "DoT":
            for _, sub_problem in current_node.sub_problems.items():
                if sub_problem.run_times == 0:
                    self.cur_node_id = sub_problem.id_tag
                    return {"node": sub_problem.id_tag, "is_finish": True}

            if current_node.num_childs == 0:
                parent_node: BaseNode = current_node._parent()
                while (parent_node != None) and (parent_node.style == "CoT"):
                    parent_node: BaseNode = parent_node._parent()
                
                if parent_node is None:
                    return {"node": current_node.id_tag, "is_finish": True}
                else:
                    self.cur_node_id = parent_node.id_tag
                    return {"node": parent_node.id_tag, "is_finish": False}

            else:
                child_node: BaseNode = current_node.select_child(**kwargs)
                self.cur_node_id = child_node.id_tag
                return {"state": True, "node": child_node.id_tag, "is_finish": False}
