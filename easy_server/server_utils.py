import threading
import json
from typing import Optional, Callable
import time
from typing import Dict, List, Tuple, Union
import queue
from functools import partial


class StreamDataResponse:
    def __init__(
        self, name: str, 
        get_data_size_callback: Callable, 
        get_chunk_data_callback: Callable,
        chunk_size: int = 100, time_interval: float = 0.1
    ):
        """
        get_data_size_callback must be a function that returns int
        get_chunk_data_callback must be a function including data_index and chunk_size
        """
        self.name = name
        self.chunk_size = chunk_size
        self.get_data_size_callback = get_data_size_callback
        self.get_chunk_data_callback = get_chunk_data_callback
        self.last_data_index = 0
        self.time_interval = time_interval
 
    def generate(self):
        try:
            while True:
                chunk_data = self.update_progress()
                yield json.dumps(chunk_data)
                time.sleep(self.time_interval)
                if chunk_data['is_finished']:
                    break
        except Exception as e:
            self.error_process(e)
            raise e
        finally:
            self.final_process()

    def update_progress(self) -> dict:
        data_size = self.get_data_size_callback()
        if self.last_data_index + self.chunk_size >= data_size:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=True,
                progress=100,
                msg=f'Processing Finished',
            )
        else:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=False,
                progress=int(float(self.last_data_index) / float(data_size) * 100),
                msg=f'Processing... {self.last_data_index}/{data_size}'
            )
            self.last_data_index += self.chunk_size
        return chunk_data

    def error_process(self, e: Exception):
        print('[Error]: ', e)

    def final_process(self):
        print(f'[Debug]: Task {self.name} finished')

    def get_progress(self, data, chunk_index: int, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'name': self.name,
            'data': data,
            'chunk_index': chunk_index,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }


class StreamTaskResponse:
    def __init__(self, name: str, callback_func: Callable, interval_time: float = 0.05):
        """
        callback_func must be a function that returns dict which has 'is_finished' key and 
        must include data_queue and last_result in its arguments
        """
        self.name = name
        self.interval_time = interval_time
        self.last_update_time = 0.0
        self.data_queue = queue.Queue()
        self.last_result = None
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.thread = None
        self.run_count = 0
    
    def start(self):
        self.thread = threading.Thread(target=self.callback_func)
        self.thread.start()
        while True:
            time.sleep(self.interval_time)
            progress_data = self.update_progress()
            yield json.dumps(progress_data)
            if progress_data['is_finished']:
                break
        self.thread.join()

    def update_progress(self) -> dict:
        if not self.data_queue.empty():
            run_data = self.data_queue.get()
            self.run_count += 1
            if run_data['is_finished']:
                self.last_result = run_data['data']
                return self.get_progress(
                    is_finished=True,
                    progress=self.run_count,
                    msg=f'[Debug]: Task {self.name} finished',
                )
            else:
                return self.get_progress(
                    is_finished=False,
                    progress=self.run_count,
                    msg=run_data['msg'],
                )
        else:
            return self.get_progress(
                is_finished=False,
                progress=self.run_count,
                msg=f'[Debug]: Waiting for task {self.name}...',
            )

    def get_progress(self, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'name': self.name,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }
    

class SocketManager:
    socket_map = {}

    def __init__(self, ws):
        self.id = id(ws)
        self.ws = ws
        SocketManager.socket_map[self.id] = self
    
    def send_msg(self, msg: dict):
        self.ws.send(json.dumps(msg))

    def close(self):
        self.ws.close()

    @staticmethod
    def remove_socket(manager):
        manager.close()
        del SocketManager.socket_map[manager.id]


class SocketDataResponse:
    def __init__(
            self, manager: SocketManager, 
            run_type: str, name: str, 
            get_data_size_callback: Callable, 
            get_chunk_data_callback: Callable,
            chunk_size: int = 100, time_interval: float = 0.1
    ):
        self.manager = manager
        self.run_type = run_type
        self.name = name
        self.chunk_size = chunk_size
        self.get_data_size_callback = get_data_size_callback
        self.get_chunk_data_callback = get_chunk_data_callback
        self.last_data_index = 0
        self.time_interval = time_interval

    def generate(self):
        try:
            while True:
                chunk_data = self.update_progress()
                self.manager.send_msg(chunk_data)
                time.sleep(self.time_interval)
                if chunk_data['is_finished']:
                    break
        except Exception as e:
            self.error_process(e)
            raise e
        finally:
            self.final_process()

    def update_progress(self) -> dict:
        data_size = self.get_data_size_callback()
        if self.last_data_index + self.chunk_size >= data_size:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=True,
                progress=100,
                msg=f'Processing Finished',
            )
        else:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=False,
                progress=int(float(self.last_data_index) / float(data_size) * 100),
                msg=f'Processing... {self.last_data_index}/{data_size}',
            )
            self.last_data_index += self.chunk_size
        return chunk_data
    
    def error_process(self, e: Exception):
        print('[Error]: ', e)
    
    def final_process(self):
        # print(f'[Debug]: Task {self.name} finished')
        pass

    def get_progress(self, data, chunk_index: int, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'run_type': self.run_type,
            'name': self.name,
            'data': data,
            'chunk_index': chunk_index,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }


class SocketTaskResponse:
    def __init__(self, manager: SocketManager, run_type: str, name: str, callback_func: Callable, interval_time: float = 0.05):
        self.manager = manager
        self.run_type = run_type
        self.name = name
        self.interval_time = interval_time
        self.last_update_time = 0.0
        self.data_queue = queue.Queue()
        self.last_result = None
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.thread = None
        self.run_count = 0

    def start(self):
        self.thread = threading.Thread(target=self.callback_func)
        self.thread.start()
        while True:
            time.sleep(self.interval_time)
            progress_data = self.update_progress()
            self.manager.send_msg(progress_data)
            if progress_data['is_finished']:
                break
        self.thread.join()
    
    def update_progress(self) -> dict:
        if not self.data_queue.empty():
            run_data = self.data_queue.get()
            self.run_count += 1
            if run_data['is_finished']:
                self.last_result = run_data['data']
                return self.get_progress(
                    is_finished=True,
                    progress=self.run_count,
                    msg=f'[Debug]: Task {self.run_type}/{self.name} finished',
                )
            else:
                return self.get_progress(
                    is_finished=False,
                    progress=self.run_count,
                    msg=run_data['msg'],
                    progress_data=run_data.get('progress_data', {}),
                )
        else:
            return self.get_progress(
                is_finished=False,
                progress=self.run_count,
                msg=f'[Debug]: Waiting for task {self.run_type}/{self.name}...',
            )
    
    def get_progress(self, is_finished: bool, progress: int, msg: str, progress_data: dict = None) -> dict:
        return {
            'run_type': self.run_type,
            'name': self.name,
            'is_finished': is_finished,
            'progress': progress,
            'progress_data': progress_data,
            'msg': msg,
            'total_finished': False,
        }
