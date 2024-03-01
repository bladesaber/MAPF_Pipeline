import numpy as np


class CostConvergeHandler(object):
    def __init__(self, stat_num: int, warm_up_num: int, tol: float, scale: float, cost_variation_init=1.0):
        self.stat_num = stat_num
        self.warm_up_num = warm_up_num
        self.tol = tol
        self.scale = scale
        self.scale_cost_variation = cost_variation_init
        self.cost_list = []

    def compute_scale_loss(self, loss):
        return loss * self.scale

    def is_converge(self, cost: float):
        self.cost_list.append(cost)
        if len(self.cost_list) < self.warm_up_num:
            return False

        scale_costs = np.array(self.cost_list[-self.stat_num:]) * self.scale
        self.scale_cost_variation = np.std(scale_costs)

        # ------ converge is not about better or wrong
        # dif = self.cost_list[-1] - self.cost_list[-self.warm_up_num]
        # z_dif = dif / cost_std * -1.0  # multiply -1.0 is for rescale to positive number
        # return z_dif <= self.tol

        return self.scale_cost_variation < self.tol
