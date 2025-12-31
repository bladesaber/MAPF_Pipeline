import numpy as np


class EarlyStopper(object):
    def __init__(
        self, 
        hot_step: int, window_size: int = 30, shift_step: int = 5, 
        atol: float = 1e-6, rtol: float = 1e-3, atol_inv: float = 1e-3,
    ):
        self.hot_step = hot_step
        self.window_size = window_size
        self.shift_step = shift_step
        self.atol = atol
        self.rtol = rtol
        self.atol_inv = atol_inv
        
        self.record: dict[str, list] = {}
        self.convergence_threshold: dict[str, float] = {}
        self.record_min: dict[str, float] = {}
    
    def define_var(self, name: str, convergence_threshold: float):
        self.record[name] = []
        self.convergence_threshold[name] = convergence_threshold
        self.record_min[name] = np.inf
    
    def add_var(self, value_dict: dict[str, float]):
        for name, value in value_dict.items():
            self.record[name].append(value)
            if len(self.record[name]) > self.window_size:
                self.record[name].pop(0)
            if len(self.record[name]) > self.hot_step:
                if value < self.record_min[name]:
                    self.record_min[name] = value
    
    def is_var_converged(self, name: str) -> bool:
        data = np.array(self.record[name])
        
        if data[-1] < self.convergence_threshold[name]:
            return True
        
        if data.shape[0] > self.shift_step:
            if np.abs(data[-1] - data[-self.shift_step]) < self.atol:
                return True
            
            if np.abs(data[-1] - data[-self.shift_step]) / data[-1] < self.rtol:
                return True
        
        return False
    
    def is_var_inverted(self, name: str) -> bool:
        data = np.array(self.record[name])
        if data[-1] > self.record_min[name] + self.atol_inv:
            return True
        return False
    
    def is_converged(self) -> bool:
        for name in self.record:
            if self.is_var_inverted(name):
                return True
        
        for name in self.record:
            if not self.is_var_converged(name):
                return False
        return True
        