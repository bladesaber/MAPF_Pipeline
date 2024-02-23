import numpy as np


class CostConvergeHandler(object):
    def __init__(self, stat_num: int, warm_up_num: int, tol: float, scale: float):
        self.stat_num = stat_num
        self.warm_up_num = warm_up_num
        self.tol = tol
        self.scale = scale
        self.cost_list = []

    def is_converge(self, cost: float):
        self.cost_list.append(cost)
        if len(self.cost_list) < self.warm_up_num:
            return False

        scale_cost = np.array(self.cost_list[-self.stat_num:]) * self.scale
        cost_std = np.std(scale_cost)

        # ------ converge is not about better or wrong
        # dif = self.cost_list[-1] - self.cost_list[-self.warm_up_num]
        # z_dif = dif / cost_std * -1.0  # multiply -1.0 is for rescale to positive number
        # return z_dif <= self.tol

        return cost_std < self.tol
