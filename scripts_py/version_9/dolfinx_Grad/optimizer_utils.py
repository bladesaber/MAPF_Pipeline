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


class CostWeightHandler(object):
    def __init__(self):
        self.tag_costs = {}
        self.tag_nums = {}
        self.tag_weight = {}
        self.tags = []

    def clear(self):
        self.tag_costs.clear()
        self.tag_nums.clear()

    def add_cost(self, name, cost):
        if name not in self.tags:
            self.tags.append(name)
            self.tag_costs[name] = cost
            self.tag_nums[name] = 1
        else:
            self.tag_costs[name] += cost
            self.tag_nums[name] += 1

    def compute_weight(self, cost_weight: dict):
        for tag in self.tag_costs.keys():
            cost_sum = self.tag_costs[tag]
            num = self.tag_nums[tag]
            self.tag_weight[tag] = cost_weight[tag] / cost_sum / num

    def get_weight(self, name):
        return self.tag_weight[name]

    def get_cost(self, name):
        cost_sum = self.tag_costs[name]
        num = self.tag_nums[name]
        return cost_sum, num

