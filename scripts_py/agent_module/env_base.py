from abc import ABC, abstractmethod
from typing import Any
import yaml


class EnvAction(object, ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def to_json(self):
        raise NotImplementedError


class EnvState(object, ABC):
    @abstractmethod
    def to_obs(self) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def to_json(self):
        raise NotImplementedError
    
    @abstractmethod
    @staticmethod
    def load_json(json_content: dict):
        raise NotImplementedError

        
class BaseEnv(object, ABC):
    def __init__(self):
        self.action: EnvAction
        self.obs: list[Any]
        self.state: EnvState
        self.reward: float
        self.terminated: bool
        self.meta_data: dict
        
        self.enable_actions: list[EnvAction]
        self.history: list[tuple[dict, dict, float, dict, bool, dict]] = []
    
    @abstractmethod
    def step(self, action: EnvAction) -> tuple[Any, float, bool, dict]:
        '''
        Return:
            observation, reward, terminated, metadata
        '''
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> tuple[Any, dict]:
        '''
        Return:
            observation, metadata
        '''
        raise NotImplementedError

    @abstractmethod
    def restore_from_state(self, state: EnvState):
        raise NotImplementedError
    
    @abstractmethod
    def restore_from_actions(self, actions: list[EnvAction]):
        raise NotImplementedError
    
    @abstractmethod
    def copy(self, env):
        raise NotImplementedError

    def record(self):
        self.history.append((
            self.action.to_json(), self.obs, self.reward, self.state.to_json(), 
            self.terminated, self.meta_data
        ))

    def save(self, file: str):
        assert file.endswith('yaml')
        with open(file, 'w') as f:
            yaml.dump({'history': self.history}, f, sort_keys=False, allow_unicode=True)

    @abstractmethod
    def load(self, file: str) -> dict:
        assert file.endswith('yaml')
        with open(file, 'r') as f:
            env_json = yaml.safe_load(f.read())
            
        