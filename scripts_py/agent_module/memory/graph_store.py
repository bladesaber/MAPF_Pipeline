import os
from typing import Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

import numpy as np
import pandas as pd
import networkx as nx


class StoreEntity(BaseModel):
    topic: str
    description: str
    meta_data: dict
    embedding: np.ndarray


class StoreRelation(StoreEntity):
    weight: float = 1.0

    
class StoreTriple(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    source_entity: StoreEntity
    target_entity: StoreEntity
    relation: StoreRelation
    topic: str
    description: str
    meta_data: dict
    embedding: np.ndarray


class BaseGraphStore(object):
    def __init__(self, with_direct=False):
        self.entities: dict[str, StoreEntity] = {}
        self.relations: dict[str, StoreRelation] = {}
        self.triples: dict[str, StoreTriple] = {}
        
        self.is_direct_graph = with_direct
        if with_direct:
            self.graph = nx.DiGraph()
        else:
            self.graph: nx.Graph = nx.Graph()
    
    def add_triple(self, source_node: StoreEntity, relation: StoreRelation, target_node: StoreEntity):
        self.graph.add_edge(source_node.topic, target_node.topic, relation=relation.topic, weight=relation.weight)

    @abstractmethod
    def search_near_entities(self, topic: str):
        raise NotImplementedError    

    @abstractmethod
    def search_near_relations(self, topic: str):
        raise NotImplementedError

    def score_graph_ppr(self, person_distribution: dict, max_iter: int):
        '''
        Score Graph through Personalized PageRank
        Args:
            person_distribution: initialization score of each entity
        '''
        return nx.pagerank(self.graph, personalization=person_distribution, max_iter=max_iter)
