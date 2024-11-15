import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
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
    def get_optimizer(parameters: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
        return optim.Adam(parameters, lr=lr)

    @staticmethod
    def get_lr_scheduler(
            optimizer: torch.optim.Optimizer, factor=0.25, threshold=1e-2, patience=1000, cooldown=1000, min_lr=0.001
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, threshold=threshold,
            cooldown=cooldown, min_lr=min_lr
        )

    @staticmethod
    def update_through_optimizer(
            optimizer: torch.optim.Optimizer, loss: torch.Tensor, scheduler: torch.optim.lr_scheduler.LRScheduler = None
    ):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step(loss)

    @staticmethod
    def update_by_raw(loss: torch.Tensor, parameters: List[torch.Tensor], lr: float):
        loss.backward()
        with torch.no_grad():
            for para in parameters:
                para -= para.grad * lr
                para.grad.zero_()


class TorchMath(object):
    @staticmethod
    def compute_cos_val(data_xyz: torch.Tensor):
        vec0, vec1 = data_xyz[1:-1, :] - data_xyz[:-2, :], data_xyz[2:, :] - data_xyz[1:-1, :]
        vec_sum = torch.sum(vec0 * vec1, dim=1)
        vec0_length, vec1_length = torch.norm(vec0, p=2, dim=1), torch.norm(vec1, p=2, dim=1)
        return vec_sum / (vec0_length * vec1_length + 1e-16)

    @staticmethod
    def compute_approximate_curvature(data_xyz: torch.Tensor):
        vec0, vec1 = data_xyz[1:-1, :] - data_xyz[:-2, :], data_xyz[2:, :] - data_xyz[1:-1, :]
        vec0_length, vec1_length = torch.norm(vec0, p=2, dim=1), torch.norm(vec1, p=2, dim=1)

        # pytorch存在比较大的浮点算术误差，这里cos_val可能大于1.0或小于-1.0
        cos_val = torch.sum(vec0 * vec1, dim=1) / (vec0_length * vec1_length + 1e-16)
        cos_val = torch.clamp(cos_val, -0.9999999999, 0.9999999999)

        arc_theta = torch.arccos(cos_val)
        # arc_theta = torch.log((1.0 - cos_val) * 500.0 + 1.0) * 0.48

        curvature = arc_theta / torch.minimum(vec0_length, vec1_length)
        return curvature

    @staticmethod
    def sigmoid(data: torch.Tensor, drift=0.0, scale=1.0, sharp=1.0):
        return scale / (1.0 + torch.exp(-1.0 * (data - drift) * sharp))


class CostUtils(object):
    @staticmethod
    def compute_length_cost(data_xyz: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['method'] == 'norm_sum':
            cost = torch.sum(torch.norm(data_xyz[1:, :] - data_xyz[:-1, :], p=2, dim=1)) * weight
        elif cfg['method'] == 'fix_value':
            length = torch.sum(torch.norm(data_xyz[1:, :] - data_xyz[:-1, :], p=2, dim=1))
            cost = torch.pow(length / cfg['value'] - 1.0, 2.0) * weight
        elif cfg['method'] == 'floor_bound':
            length = torch.sum(torch.norm(data_xyz[1:, :] - data_xyz[:-1, :], p=2, dim=1))
            cost = F.relu(1.0 - length / cfg['value']) * weight
        elif cfg['method'] == 'ceil_bound':
            length = torch.sum(torch.norm(data_xyz[1:, :] - data_xyz[:-1, :], p=2, dim=1))
            cost = F.relu(length / cfg['value'] - 1.0) * weight
        elif cfg['method'] == 'square_sum':
            square_length = (torch.pow(data_xyz[1:, :] - data_xyz[:-1, :], 2.0)).sum(dim=1)
            cost = square_length.sum() * weight
        else:
            raise ValueError(f"[ERROR]: Non-valid cost method {cfg['method']}")
        return cost

    @staticmethod
    def compute_curvature_cost(data: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['curvature_method'] == 'approximate_curvature':
            curvature = TorchMath.compute_approximate_curvature(data)
        elif cfg['curvature_method'] == 'analytical_curvature':
            curvature = data
        else:
            raise ValueError(f"[ERROR]: Non-valid curvature method {cfg['curvature_method']}")

        if cfg['method'] == 'max':
            cost = (curvature.max() + curvature.mean()) * weight
        elif cfg['method'] == 'mean':
            cost = curvature.mean() * weight
        elif cfg['method'] == 'sigmoid':
            cur_sigmoid = TorchMath.sigmoid(curvature, drift=cfg['value'], sharp=20.0)
            cost = (curvature * cur_sigmoid).sum() * weight
        elif cfg['method'] == 'ceil_bound_1':
            ratio = F.relu(curvature / cfg['value'] - 1.0)
            cost = (ratio.max() + ratio.mean()) * weight
        elif cfg['method'] == 'ceil_bound_2':
            # 调整曲率并不单调整某几个控制点而是应该所有控制点一起调整，relu会cut掉其他控制点的梯度
            ratio = F.relu(curvature / cfg['value'] - 1.0)
            ratio_sigmoid = TorchMath.sigmoid(ratio, scale=2.0, sharp=10.0) - 1.0
            cost = (ratio * ratio_sigmoid).sum() * weight
        else:
            raise ValueError(f"[ERROR]: Non-valid cost method {cfg['method']}")
        return cost

    @staticmethod
    def compute_boundary_cost(data: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['method'] == 'ceil_bound':
            ratio = F.relu(data / cfg['value'] - 1.0)
            cost = (ratio.max() + ratio.mean()) * weight
        elif cfg['method'] == 'bottom_value':
            ratio = F.relu(1.0 - data / cfg['value'])
            cost = (ratio.max() + ratio.mean()) * weight
        else:
            raise ValueError(f"[ERROR]: Non-valid cost method {cfg['method']}")
        return cost

    @staticmethod
    def compute_distribution_cost(data: torch.Tensor, cfg: Dict):
        weight = cfg['weight']
        if cfg['method'] == 'length_variance':
            sub_lengths = torch.norm(data[1:, :] - data[:-1, :], p=2, dim=1)
            cost = torch.pow(sub_lengths - sub_lengths.mean(), 2.0).sum() * weight
        elif cfg['method'] == 'center_cluster':
            xyz_np = TorchUtils.tensor2np(data)
            center_t = TorchUtils.np2tensor((xyz_np[:-2, :] + xyz_np[2:, :]) * 0.5, require_grad=False)
            cost = torch.pow(data[1:-1] - center_t, 2.0).sum() * weight
        elif cfg['method'] == 'value_variance':
            cost = torch.pow(data - data.mean(), 2.0).sum() * weight
        else:
            raise ValueError(f"[ERROR]: Non-valid cost method {cfg['method']}")
        return cost

    @staticmethod
    def compute_vector_cost(data_xyz: torch.Tensor, cfg: Dict, method: str):
        weight = cfg['weight']

        vector = cfg['vector']
        vector = vector / np.linalg.norm(vector, ord=2)
        vector = TorchUtils.np2tensor(vector, require_grad=False)

        if method == 'left2any':
            vecs = data_xyz[1:, :] - data_xyz[0]
        elif method == 'any2right':
            vecs = data_xyz[:-1, :] - data_xyz[-1]
        else:
            raise ValueError(f"[ERROR]: Non-valid cost method {cfg['method']}")

        vecs_length = torch.norm(vecs, p=2, dim=1)
        cos_value = torch.sum(vecs * vector, dim=1) / vecs_length
        sin_square = 1.0 - torch.pow(cos_value, 2.0)
        cost = (torch.pow(vecs_length, 2.0) * sin_square).sum() * weight

        return cost

    @staticmethod
    def compute_shape_conflict_cost(pcd0: torch.Tensor, pcd1: torch.Tensor, require_dist: np.ndarray, weight: float):
        require_dist = TorchUtils.np2tensor(require_dist.reshape((-1, 1)), require_grad=False)
        return F.relu(require_dist - torch.norm(pcd0 - pcd1, p=2, dim=1)) * weight


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
