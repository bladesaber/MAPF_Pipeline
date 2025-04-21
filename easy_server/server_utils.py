import threading
import json
from typing import Optional, Callable
import time

from pathSearch_pkg.build import MapfPkg
import numpy as np


class LongTaskResponse:
    def __init__(self):
        self.interval_time = 0.05
        self.last_update_time = 0
        self.is_finished = False
 
    def generate(self):
        try:
            self.before_run()

            while True:
                self.before_task()
                self.run_iteration_task()
                self.after_task()

                progress_data = self.update_progress()

                if time.time() - self.last_update_time > self.interval_time:
                    yield json.dumps(progress_data)
                    self.last_update_time = time.time()
                

            
            self.after_run()

        except Exception as e:
            self.error_process()
            raise e
        finally:
            self.final_process()
    
    def update_progress(self) -> dict:
        progress_data = {
            'progress': 0, 
            'run_state_tag': 'task1',
            'is_finished': True,
            'msg': 'Processing...',
            'data': {}
        }
        self.is_finished = True
        return progress_data

    def run_iteration_task(self) -> dict:
        raise NotImplementedError("run_iteration_task must be implemented")

    def before_run(self):
        raise NotImplementedError("before_run must be implemented")

    def after_run(self):
        raise NotImplementedError("after_run must be implemented")

    def before_task(self):
        raise NotImplementedError("before_task must be implemented")

    def after_task(self):
        raise NotImplementedError("after_task must be implemented")

    def error_process(self):
        raise NotImplementedError("error_process must be implemented")

    def final_process(self):
        raise NotImplementedError("final_process must be implemented")


class PathSearchSocketResponse(object):
    def __init__(self, id: int):
        pass

