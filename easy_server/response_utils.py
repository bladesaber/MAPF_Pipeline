from typing import Optional, Callable
from functools import partial
import queue
import threading
import time
import json
from multiprocessing import Queue, Event
from multiprocessing import Process

class StreamThreadTaskResponse:
    def __init__(
            self, name: str, 
            callback_func: Callable, 
            interval_time: float = 0.2, 
            max_running_time: float = 10, 
            min_yield_interval: float = 0.1
        ):
        self.name = name
        self.data_queue = queue.Queue()
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.thread = None
        self.interval_time = interval_time
        self.start_time = None
        self.last_update_time = 0.0
        self.max_running_time = max_running_time
        self.is_finished = False
        self.result = None
        self.min_yield_interval = min_yield_interval
        self.last_yield_time = 0.0
    
    def start(self):
        self.thread = threading.Thread(target=self.callback_func)
        self.thread.start()

        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_yield_time = time.time()
        
        while not self.is_finished:
            try:
                # Try to get a record from the queue with a timeout
                record = self.data_queue.get(timeout=self.interval_time)
                self.is_finished = record['is_finished']
                
                # Check if enough time has passed since last yield
                current_time = time.time()
                if current_time - self.last_yield_time < self.min_yield_interval:
                    time.sleep(self.min_yield_interval - (current_time - self.last_yield_time))
                
                if self.is_finished:
                    self.result = {
                        'is_success': True,
                        'result': record['result']
                    }
                    yield json.dumps(record)
                    break
                
                # Add name and progress to the record
                record.update({
                    'name': self.name,
                    'progress': time.time() - self.start_time
                })
                yield json.dumps(record)
                self.last_yield_time = time.time()
                    
            except queue.Empty:
                # Check if we've exceeded max running time
                if time.time() - self.start_time > self.max_running_time:
                    self.is_finished = True
                    self.result = {
                        'is_success': False,
                        'msg': 'Max running time reached',
                    }
                    yield json.dumps({
                        'name': self.name,
                        'is_finished': True,
                        'is_success': False,
                        'progress': time.time() - self.start_time,
                        'msg': 'Max running time reached'
                    })
                    break

                # If no record is available, yield a waiting message
                current_time = time.time()
                if current_time - self.last_yield_time < self.min_yield_interval:
                    time.sleep(self.min_yield_interval - (current_time - self.last_yield_time))
                
                yield json.dumps({
                    'name': self.name,
                    'is_finished': False,
                    'is_success': False,
                    'progress': time.time() - self.start_time,
                    'msg': f'[Debug]: Waiting for task {self.name}...'
                })
                self.last_yield_time = time.time()

        self.thread.join()
    
    def __del__(self):
        # Ensure process is cleaned up
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                self.thread.join()


class StreamProcessTaskResponse:
    def __init__(
            self, name: str, 
            callback_func: Callable, 
            interval_time: float = 0.2, 
            max_running_time: float = 10, 
            min_yield_interval: float = 0.1
        ):
        self.name = name
        self.data_queue = Queue()  # Using multiprocessing Queue
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.process = None
        self.interval_time = interval_time
        self.start_time = None
        self.last_update_time = 0.0
        self.max_running_time = max_running_time
        self.is_finished = False
        self.result = None
        self.stop_event = Event()
        self.min_yield_interval = min_yield_interval
        self.last_yield_time = 0.0
    
    def start(self):
        self.process = Process(target=self.callback_func)
        self.process.start()

        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_yield_time = time.time()
        
        while not self.is_finished and not self.stop_event.is_set():
            try:
                # Try to get a record from the queue with a timeout
                record = self.data_queue.get(timeout=self.interval_time)
                print('get record at time:', time.time())

                self.is_finished = record['is_finished']
                
                # Check if enough time has passed since last yield
                current_time = time.time()
                if current_time - self.last_yield_time < self.min_yield_interval:
                    time.sleep(self.min_yield_interval - (current_time - self.last_yield_time))
                
                if self.is_finished:
                    self.result = {
                        'is_success': True,
                        'result': record['result']
                    }
                    yield json.dumps(record)
                    break
                
                # Add name and progress to the record
                record.update({
                    'name': self.name,
                    'progress': time.time() - self.start_time,
                    'msg': ''
                })
                yield json.dumps(record)
                self.last_yield_time = time.time()
                    
            except queue.Empty:
                # Check if we've exceeded max running time
                if time.time() - self.start_time > self.max_running_time:
                    self.is_finished = True
                    self.result = {
                        'is_success': False,
                        'msg': 'Max running time reached',
                    }
                    yield json.dumps({
                        'name': self.name,
                        'is_finished': True,
                        'is_success': False,
                        'progress': time.time() - self.start_time,
                        'msg': 'Max running time reached'
                    })
                    break

                # If no record is available, yield a waiting message
                current_time = time.time()
                if current_time - self.last_yield_time < self.min_yield_interval:
                    time.sleep(self.min_yield_interval - (current_time - self.last_yield_time))
                
                yield json.dumps({
                    'name': self.name,
                    'is_finished': False,
                    'is_success': False,
                    'progress': time.time() - self.start_time,
                    'msg': f'[Debug]: Waiting for task {self.name}...'
                })
                self.last_yield_time = time.time()

        # Clean up the process
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
    
    def __del__(self):
        # Ensure process is cleaned up
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
