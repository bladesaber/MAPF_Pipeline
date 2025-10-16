import os
import yaml
from abc import ABC, abstractmethod
from typing import Callable, Any
from pydantic import BaseModel

import numpy as np
import pandas as pd

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent_module.openai_llm import OpenaiConfig, OpenaiLlm
from agent_module.env_base import BaseEnv


class UserPrompt(object):
    def __init__(self, name: str, backend: list[str], description: str):
        self.name: str = name
        self.backend: list[str] = backend
        self.description = description
        self.system: str
        self.user: str
    
    def to_json(self):
        assert self.user is not None
        prompt_json = {
            'name': self.name,
            'backend': self.backend,
            'description': self.description,
            'system': self.system,
            'user': self.user
        }
        return prompt_json
    
    def save(self, file: str):
        assert file.endswith('yaml')
        with open(file, 'w') as f:
            yaml.dump(self.to_json(), f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def load(file: str):
        assert file.endswith('yaml')
        with open(file, 'r') as f:
            prompt_json = yaml.safe_load(f.read())
        prompt = UserPrompt(
            name=prompt_json['name'], backend=prompt_json['backend'],
            description=prompt_json['description']
        )
        prompt.system = prompt_json['system']
        prompt.user = prompt_json['user']
        return prompt


class BaseContext(TypedDict):
    name: str = 'BaseContext'
    description: str = 'Description: Using for agent workflow context.'
    history: list[Any]
    memory: list[Any]


class WorldModelContext(BaseContext):
    name: str = 'WorldModelContext'
    description: str = 'Description: Using for agent workflow context with world model.'
    env: BaseEnv


class ActionNode(object, ABC):
    def __init__(self, name: str, description: str):
        '''Action/Tool Node'''
        super().__init__()
        self.name = name
        self.description = description
        self.is_from_start = False
        self.is_to_end = False
    
    @abstractmethod
    def step(self, context: BaseContext) -> dict:
        raise NotImplementedError

    def distribute_function(self) -> str|list[str]:
        raise NotImplementedError

    def from_start(self, workflow: StateGraph):
        workflow.add_edge(START, self.name)
        self.is_from_start = True

    def to_end(self, workflow: StateGraph):
        workflow.add_edge(self.name, END)
        self.is_to_end = True


class DecisionNode(ActionNode):
    def __init__(self, name, description):
        super().__init__(name, description)
        
    @abstractmethod
    def condition_function(self, context: BaseContext) -> str|list[str]:
        raise NotImplementedError


class BaseAgent(ActionNode):
    def __init__(self, name: str, description: str, context_type: Callable):
        super().__init__(name, description)
        self.actions: dict[str, ActionNode|DecisionNode|BaseAgent] = {}
        self.context_type = context_type
        self.context: BaseContext
        self.workflow_builder = StateGraph(context_type)
        self.workflow: CompiledStateGraph
    
    def assemble_graph(self):
        for _, action in self.actions.items():
            if isinstance(action, ActionNode):
                assert action.name not in self.workflow.nodes.keys(), f'[ERROR]: {action.name} already in graph.'
                self.workflow_builder.add_node(action.name, action.step)
            
                for target_name in action.distribute_function():
                    self.workflow_builder.add_edge(action.name, target_name)
                
            elif isinstance(action, DecisionNode):
                assert action.name not in self.workflow_builder.nodes.keys(), f'[ERROR]: {action.name} already in graph.'
                self.workflow_builder.add_node(action.name, action.step)
                self.workflow_builder.add_conditional_edges(action.name, action.condition_function)
            
            elif isinstance(action, BaseAgent):
                assert action.name not in self.workflow_builder.nodes.keys(), f'[ERROR]: {action.name} already in graph.'
                self.workflow_builder.add_node(action.name, action.step)
                
                if (action.distribute_function is None) and (action.condition_function is None):
                    raise ValueError(f'[Error]: [BaseAgent {self.name}]/[{action.name}] has no edges.')
                elif (action.distribute_function is not None) and (action.condition_function is not None):
                    raise ValueError(f'[Error]: [BaseAgent {self.name}]/[{action.name}] has conflict edges.')
                elif action.distribute_function is None:
                    for target_name in action.distribute_function():
                        self.workflow_builder.add_edge(action.name, target_name)
                else:
                    self.workflow_builder.add_conditional_edges(action.name, action.condition_function)
            
            else:
                raise ValueError(f'[ERROR]: type of [BaseAgent {self.name}]/[{action.name}] non-valiable.')
        self.workflow = self.workflow_builder.compile()

    def step(self, parent_context: BaseContext) -> dict:
        child_context: BaseContext = self.parent2child_schemas(parent_context)
        child_context: dict = self.run_workflow(child_context)
        return self.child2parent_schemas(child_context)
    
    @abstractmethod
    def parent2child_schemas(self, parent_context: BaseContext) -> BaseContext:
        raise NotImplementedError
    
    @abstractmethod
    def child2parent_schemas(self, child_context: dict) -> dict:
        raise NotImplementedError
    
    def distribute_function(self) -> str|list[str]:
        return None
    
    def condition_function(self, context: BaseContext) -> str|list[str]:
        return None
    
    def run_action(self, action_name: str, context: BaseContext) -> BaseContext:
        res = self.actions[action_name].step(context)
        for key, value in res.items():
            context[key] = value
        return context

    def run_workflow(self, context: BaseContext) -> dict:
        self.context = context
        return self.workflow.invoke(self.context)
    
    def run_deepthink(self, context: BaseContext) -> dict:
        '''For more complex: like ToT'''
        raise NotImplementedError


class LlmAgent(BaseAgent):
    def __init__(self, name, description, context_type, config: OpenaiConfig):
        super().__init__(name, description, context_type)
        self.llm_config = config
        self.llm: OpenaiLlm = OpenaiLlm(self.llm_config, setup=True)


class PoolMessage(BaseModel):
    from_address: str
    to_address: str
    msg_type: str
    context: Any


class PoolAgent(BaseAgent):
    def __init__(self, name, description, config, context_type):
        super().__init__(name, description, config, context_type)
        self.email_address: str
        self.email_msg: list[PoolMessage] = []
    
    @abstractmethod
    def belong_to(self, msg: PoolMessage) -> bool:
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    @abstractmethod
    def send_email(self):
        raise NotImplementedError
    

class PoolMultiAgents(object):
    PoolMessageTags = [
        '<any>'
    ]
    
    def __init__(self):
        self.agents: dict[str, PoolAgent] = {}
        self.msg_pool: list[PoolMessage] = []

    def run(self):
        for msg in self.msg_pool:
            for _, agent in self.agents.items():
                if agent.belong_to(msg):
                    agent.email_msg.append(msg)
        
        for _, agent in self.agents.items():
            agent.solve()
    