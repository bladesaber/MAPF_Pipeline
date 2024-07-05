import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from typing import List, Callable, Dict
import scipy


class TorchUtils(object):
    @staticmethod
    def np2tensor(data: np.ndarray, require_grad: bool, device: str = 'cpu'):
        tensor = torch.tensor(data, dtype=torch.float64)
        tensor.requires_grad = require_grad
        if device == 'cuda':
            tensor.cuda()
        elif device == 'cpu':
            pass
        else:
            tensor.to(device)
        return tensor

    @staticmethod
    def np2parameter(data: np.ndarray, require_grad: bool, device: str = 'cpu'):
        para = nn.parameter.Parameter(TorchUtils.np2tensor(data), requires_grad=require_grad)
        if device == 'cuda':
            para.cuda()
        elif device == 'cpu':
            pass
        else:
            para.to(device)
        return para

    @staticmethod
    def tensor2np(data: torch.Tensor):
        if data.requires_grad:
            return data.detach().numpy()
        else:
            return data.numpy()

    @staticmethod
    def tensor_grad2np(data: torch.Tensor):
        return TorchUtils.tensor2np(data.grad)

    @staticmethod
    def compute_grad(loss: torch.Tensor):
        loss.backward()

    @staticmethod
    def concatenate(datas: List[torch.Tensor], dim=-1):
        return torch.concatenate(datas, dim=dim)

    @staticmethod
    def compute_segment_length(path_pcd: torch.Tensor):
        return torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1)

    @staticmethod
    def compute_menger_curvature(path_pcd: torch.Tensor):
        """
        这个方法的问题是，它属于3点成圆方法，其优化的曲率是离散的
        menger curvature无法提供好的精梯度指引
        """
        pcd0 = path_pcd[:-2, :]
        pcd1 = path_pcd[1:-1, :]
        pcd2 = path_pcd[2:, :]

        a_lengths = torch.norm(pcd1 - pcd0, p=2, dim=1)
        b_lengths = torch.norm(pcd2 - pcd1, p=2, dim=1)
        c_lengths = torch.norm(pcd0 - pcd2, p=2, dim=1)
        s = (a_lengths + b_lengths + c_lengths) * 0.5
        # be careful, if value equal to 0, will cause nan grad
        A = torch.sqrt(s * (s - a_lengths) * (s - b_lengths) * (s - c_lengths) + 1e-16)
        radius = a_lengths * b_lengths * c_lengths * 0.25 / A
        return radius

    @staticmethod
    def compute_approximate_curvature(path_pcd: torch.Tensor):
        vec0 = path_pcd[1:-1, :] - path_pcd[:-2, :]
        vec1 = path_pcd[2:, :] - path_pcd[1:-1, :]
        vec0_length = torch.norm(vec0, p=2, dim=1)
        vec1_length = torch.norm(vec1, p=2, dim=1)

        # pytorch存在比较大的浮点算术误差，这里cos_val可能大于1.0或小于-1.0
        cos_val = torch.sum(vec0 * vec1, dim=1) / (vec0_length * vec1_length + 1e-16)
        cos_val = torch.clamp(cos_val, -0.9999999999, 0.9999999999)

        arc_theta = torch.arccos(cos_val)
        # arc_theta = torch.log((1.0 - cos_val) * 500.0 + 1.0) * 0.48
        curvature = arc_theta / torch.minimum(vec0_length, vec1_length)
        return curvature

    @staticmethod
    def compute_spline_curvature(path_pcd: torch.Tensor, order1_mat: torch.Tensor, order2_mat: torch.Tensor):
        """
        这个方法无法用于拼接的样条
        """
        pcd_order1 = order1_mat.matmul(path_pcd)
        pcd_order2 = order2_mat.matmul(path_pcd)

        a = torch.cross(pcd_order1, pcd_order2, dim=1)
        a = torch.norm(a, dim=1, p=2)

        b = torch.norm(pcd_order1, dim=1, p=2)
        b = torch.pow(b, 3.0)

        return a / (b + 1e-8)

    @staticmethod
    def compute_cos_val(path_pcd: torch.Tensor):
        vec0 = path_pcd[1:-1, :] - path_pcd[:-2, :]
        vec1 = path_pcd[2:, :] - path_pcd[1:-1, :]
        a = torch.sum(vec0 * vec1, dim=1)
        vec0_length = torch.norm(vec0, p=2, dim=1)
        vec1_length = torch.norm(vec1, p=2, dim=1)
        return a / (vec0_length * vec1_length + 1e-16)

    @staticmethod
    def get_optimizer(parameters: List[torch.Tensor], lr: float):
        return optim.Adam(parameters, lr=lr)

    @staticmethod
    def update_through_optimizer(optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def update_by_raw(loss: torch.Tensor, parameters: List[torch.Tensor], lr: float):
        loss.backward()
        with torch.no_grad():
            for para in parameters:
                para -= para.grad * lr
                para.grad.zero_()


class MeanTestLearner(object):
    def __init__(self, init_lr: float, stat_num=100, min_lr=1e-3):
        self.losses = []
        self.lr = init_lr
        self.stat_num = stat_num
        self.min_lr = min_lr

    def get_lr(self, loss: float) -> Dict:
        self.losses.append(loss)

        num = len(self.losses)
        if num < (self.stat_num * 2):
            return {'state': False, 'lr': self.lr}

        if num % self.stat_num == 0:
            loss_seq0 = np.array(self.losses[-self.stat_num:])
            loss_seq1 = np.array(self.losses[-self.stat_num * 2: -self.stat_num])
            p_value = scipy.stats.ttest_ind(loss_seq0, loss_seq1).pvalue
            if p_value > 0.5:
                self.lr *= 0.5

            if self.lr <= self.min_lr:
                return {'state': True, 'lr': None}
            else:
                return {'state': False, 'lr': self.lr}

        else:
            return {'state': False, 'lr': self.lr}


def main():
    a_np = np.random.random(size=(10, 3))
    b_np = np.random.random(size=(10, 3))
    a_tensor = torch.from_numpy(a_np)
    b_tensor = torch.from_numpy(b_np)
    dist_np = np.linalg.norm(a_np - b_np, axis=1, ord=2)
    dist_tensor = torch.norm(a_tensor - b_tensor, p=2, dim=1)

    print(dist_np)
    print(dist_tensor)


if __name__ == '__main__':
    main()