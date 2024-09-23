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
    def grad2np(data: torch.Tensor):
        return TorchUtils.tensor2np(data.grad)

    @staticmethod
    def compute_grad(loss: torch.Tensor):
        loss.backward()

    @staticmethod
    def concatenate(datas: List[torch.Tensor], dim=-1):
        return torch.concatenate(datas, dim=dim)

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


class CostUtils(object):
    @staticmethod
    def compute_spring_length_cost(path_pcd: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['type'] == 'norm_sum':
            length = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1).sum()
            return length * weight
        elif cfg['type'] == 'specify_value':
            length = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1).sum()
            ratio = length / cfg['length'] - 1.0
            return torch.sqrt(torch.pow(ratio, 2.0)) * weight
        elif cfg['type'] == 'bottom_value':
            length = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1).sum()
            ratio = F.relu(cfg['length'] / (length + 1e-2) - 1.0)
            return ratio * weight
        elif cfg['type'] == 'upper_value':
            length = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1).sum()
            ratio = F.relu(length / cfg['length'] - 1.0)
            return ratio * weight
        elif cfg['type'] == 'square_mean':
            square_length = (torch.pow(path_pcd[1:, :] - path_pcd[:-1, :], 2.0)).sum(dim=1)
            cost = square_length.mean()
            return cost * weight
        else:
            raise ValueError()

    @staticmethod
    def compute_spring_variance_cost(path_pcd: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['type'] == 'square_mean':
            sub_lengths = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1)
            mean_length = sub_lengths.mean()
            variance = (torch.pow(sub_lengths - mean_length, 2.0)).mean()
            return variance * weight
        elif cfg['type'] == 'range_mean':
            sub_lengths = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1)
            mean_length = sub_lengths.mean()
            variance = F.relu(torch.abs(sub_lengths / mean_length - 1.0) - cfg['range_value'])
            variance = variance.mean()
            return variance * weight
        else:
            raise ValueError()

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
    def compute_cos_val(path_pcd: torch.Tensor):
        vec0 = path_pcd[1:-1, :] - path_pcd[:-2, :]
        vec1 = path_pcd[2:, :] - path_pcd[1:-1, :]
        a = torch.sum(vec0 * vec1, dim=1)
        vec0_length = torch.norm(vec0, p=2, dim=1)
        vec1_length = torch.norm(vec1, p=2, dim=1)
        return a / (vec0_length * vec1_length + 1e-16)

    @staticmethod
    def compute_curvature_cost(path_pcd: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['type'] == 'raw':
            curvature = CostUtils.compute_approximate_curvature(path_pcd)
            return (curvature.max() + curvature.mean()) * weight
        elif cfg['type'] == 'upper_value':
            curvature = CostUtils.compute_approximate_curvature(path_pcd)
            ratio = (curvature - cfg['value']) / cfg['value']
            return ((F.relu(ratio)).max() + (F.relu(ratio)).mean()) * weight
        else:
            raise ValueError

    @staticmethod
    def compute_segment_constraint(path_pcd: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['type'] == 'upper_value':
            seg_length = torch.norm(path_pcd[1:, :] - path_pcd[:-1, :], p=2, dim=1)
            ratio = seg_length / cfg['value'] - 1.0
            return ((F.relu(ratio)).max() + (F.relu(ratio)).mean()) * weight
        else:
            raise ValueError

    @staticmethod
    def compute_connector_curvature_cost(left_pcd: torch.Tensor, right_pcd: torch.Tensor, cfg: Dict):
        path_pcd = TorchUtils.concatenate([
            left_pcd[-cfg['left_num']:], right_pcd[:cfg['right_num']]
        ], dim=0)
        return CostUtils.compute_curvature_cost(path_pcd, cfg)

    @staticmethod
    def compute_shape_conflict_cost(pcd0: torch.Tensor, pcd1: torch.Tensor, require_dist: np.ndarray, weight: float):
        require_dist = TorchUtils.np2tensor(require_dist.reshape((-1, 1)), require_grad=False)
        return F.relu(require_dist - torch.norm(pcd0 - pcd1, p=2, dim=1)) * weight


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
