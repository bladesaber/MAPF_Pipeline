import numpy as np


class BaseGradOptimizer(object):
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.grads = np.zeros(shape)
        self.step = 0

    def update(self, grad: np.ndarray):
        self.grads = grad
        self.step += 1
        return self.grads


class MomentumSGDOptimizer(BaseGradOptimizer):
    def __init__(self, shape: tuple[int, ...], momentum: float = 0.9):
        super().__init__(shape)
        self.momentum = momentum
    
    def update(self, grad: np.ndarray):
        if self.step == 0:
            self.grads = grad
        else:
            self.grads = self.grads * self.momentum + grad * (1 - self.momentum)
        self.step += 1
        return self.grads


class AdamOptimizer(BaseGradOptimizer):
    def __init__(self, shape: tuple[int, ...], first_moment: float = 0.9, second_moment: float = 0.999, epsilon: float = 1e-8):
        super().__init__(shape)
        self.epsilon = epsilon
        self.first_moment = first_moment
        self.second_moment = second_moment
        self.first_moment_grad = np.zeros(shape)
        self.second_moment_grad = np.zeros(shape)
        self.grads = np.zeros(shape)
    
    def update(self, grad: np.ndarray):
        if self.step == 0:
            self.first_moment_grad = grad
            self.second_moment_grad = grad ** 2
            self.grads = grad
        else:
            self.first_moment_grad = self.first_moment_grad * self.first_moment + grad * (1 - self.first_moment)
            self.second_moment_grad = self.second_moment_grad * self.second_moment + grad ** 2 * (1 - self.second_moment)
            self.grads = self.first_moment_grad / (np.sqrt(self.second_moment_grad) + self.epsilon)
        self.step += 1
        return self.grads
    
