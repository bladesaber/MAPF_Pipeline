import os.path
import shutil
import numpy as np
import dolfinx
from typing import Dict
from mpi4py import MPI

from .grad_runner import RasGradRunner
from .topology_optimize_utils import StopLearner
from .grid_mesh_tool import GridSmoothRecorder
from ..dolfinx_Grad.vis_mesh_utils import VisUtils
from ..dolfinx_Grad.recorder_utils import TensorBoardRecorder


class TopologyPathSolver(object):
    def __init__(self, grad_runner: RasGradRunner, learn_cfg: Dict):
        self.grad_runner = grad_runner
        self.learn_cfg = learn_cfg

        self.grid_recorder = GridSmoothRecorder(
            VisUtils.convert_to_grid(self.grad_runner.domain),
            xyz_num=learn_cfg['grid_nums'],
            record_dir=learn_cfg['grid_smooth_dir']
        )

    def run_optimize(self):
        stop_learner = StopLearner(stat_num=self.learn_cfg.get('stat_num', 100))
        max_iteration = self.learn_cfg.get('max_iteration', 500)
        log_step = self.learn_cfg.get('log_step', 10)
        record_step = self.learn_cfg.get('record_step', 5)

        if os.path.exists(self.learn_cfg['tensorboard_dir']):
            shutil.rmtree(self.learn_cfg['tensorboard_dir'])
        os.mkdir(self.learn_cfg['tensorboard_dir'])
        logger = TensorBoardRecorder(log_dir=self.learn_cfg['tensorboard_dir'])

        # ------ for debug
        self.grad_runner.simulate()
        last_loss = self.grad_runner.opt_problem.evaluate_cost_functional(
            self.grad_runner.domain.comm, update_state=False
        )
        print(f"[Debug]: Init loss:{last_loss:.3f}")
        cost_log = ''
        for name, value in self.grad_runner.opt_problem.get_cost_info(comm=MPI.COMM_WORLD, update_state=False):
            cost_log += f"{name}:{value:.3f} "
        print(cost_log)
        # ------

        step = 0
        min_loss = np.inf
        while True:
            step += 1

            grads_dict: Dict[str, dolfinx.fem.Function] = self.grad_runner.compute_gradient()

            loss = self.grad_runner.opt_problem.evaluate_cost_functional(
                self.grad_runner.domain.comm, update_state=False
            )
            print(f"[Debug]: Step_{step} loss:{loss:.3f}")

            self.grad_runner.update_control(grads_dict)

            if step % record_step == 0:
                self.grad_runner.log_info(step)

                for name, control in self.grad_runner.controls_dicts.items():
                    self.grid_recorder.mapping_scalar(
                        name, control, self.grad_runner.domain, default_value=0.0, is_point_field=False
                    )
                self.grid_recorder.write(step)

            loss_dict = dict([(name, value) for name, value in self.grad_runner.opt_problem.get_cost_info(
                comm=MPI.COMM_WORLD, update_state=False
            )])
            logger.write_scalars('loss', loss_dict, step)

            if step % log_step == 0:
                cost_log = ''
                for name, value in loss_dict.items():
                    cost_log += f"{name}:{value:.3f} "
                print(cost_log)

            if step > max_iteration:
                break

            # if stop_learner.is_stop(loss):
            #     break

            # min_loss = np.minimum(min_loss, loss)
            # if loss > min_loss * 1.1:
            #     break
