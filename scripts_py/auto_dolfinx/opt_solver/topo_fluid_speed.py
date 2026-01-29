import os
import shutil

import numpy as np

from scripts_py.vtu_recorder import TensorBoardRecorder
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import TopoFluidOptimizer


class TopoFluidSpeedOptimizer(TopoFluidOptimizer):
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01],
        max_iteration=200,
        hot_step: int = 20,
        momentum_weight: float = 0.8,
        full_run: bool = False,
        recorder: TensorBoardRecorder|None = None,
        result_vtu_file: str|None = None
    ) -> dict:
        with_line_search = len(line_search_steps) > 1
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        if full_run:
            assert not with_line_search, 'Full run and line search cannot be used together'
        
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(step)
        
        with_momentum = momentum_weight > 0.0
        if with_momentum:
            cum_direction_np = np.zeros_like(self.topo_weight.x.array)  # type: ignore
        
        while True:
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.5f} opt_length:{step_length_last:.5f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.5f}"
                print(log)
                
                if recorder is not None:
                    for name, value in log_dict.items():
                        if isinstance(value, dict):
                            recorder.write_scalars(tag=name, scalar_dict=value, step=step)
                        else:
                            recorder.write_scalar(tag=name, scalar_value=value, step=step)
                
                if np.abs(loss_current - loss_last) < 1e-6 and (not full_run) and step > hot_step:
                    is_solved = True
                    break
                
            cur_direction_np = self.compute_gradient(
                adj_input=np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2), 
                with_filter=True, with_norm=True
            )
            if with_momentum:
                cum_direction_np = cum_direction_np * momentum_weight + cur_direction_np * (1.0 - momentum_weight)
                direction_np = cum_direction_np / np.max(np.abs(cum_direction_np))
            else:
                direction_np = cur_direction_np
            
            find_best_step = False
            if with_line_search:
                topo_phi_origin: np.ndarray = topo_phi.x.array.copy()  # type: ignore
                last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
                line_search_start_index = np.maximum(last_step_index - 1, 0)
                for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                    step_length = line_search_steps_np[i]
                    new_topo_phi = topo_phi_origin - direction_np * step_length
                    new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
                    topo_phi.x.array[:] = new_topo_phi  # type: ignore
                    
                    loss_new = self.opt_problem(values=[topo_phi])  # type: ignore
                    if loss_new < loss_current or step < hot_step:
                        find_best_step = True
                        loss_last = loss_current
                        loss_current = loss_new * 0.2 + loss_last * 0.8
                        step_length_last = step_length
                        # self.topo_weight.x.array[:] = topo_phi.x.array  # type: ignore          
                        self.record_Upw(step, gradient=direction_np)
                        break

            else:
                new_topo_phi = topo_phi.x.array - direction_np * step_length_last  # type: ignore
                new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
                topo_phi.x.array[:] = new_topo_phi  # type: ignore
                loss_new = self.opt_problem(values=[topo_phi])  # type: ignore
                if loss_new < loss_current or step < hot_step or full_run:
                    find_best_step = True
                    loss_last = loss_current
                    loss_current = loss_new * 0.2 + loss_last * 0.8
                    self.record_Upw(step, gradient=direction_np)
                    
            if not find_best_step:
                print(f"[Warning]: line search fail.")
                is_solved = False
                break
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        if result_vtu_file is not None:
            last_step_record_file = self.recorder.get_step_file_path(step)
            if not os.path.exists(last_step_record_file):
                last_step_record_file = self.recorder.get_step_file_path(step - 1)
            shutil.copy(last_step_record_file, result_vtu_file)        
        return {'is_success': is_solved}  # type: ignore
    
