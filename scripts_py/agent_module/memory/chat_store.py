import os

import numpy as np
import pandas as pd

from agent_module.openai_llm import MessageItem


class BaseChatStore(object):
    '''Short-Term Memory'''
    
    def __init__(self, token_limit: int, chat_store_key: str):
        self.chat_memory: list[MessageItem] = []

    def add_msg(self, msg: dict):
        self.chat_memory.append(MessageItem.to_msg(msg))
    
    def get_msg(self, last_n: int, role_filter: list[str]):
        raise NotImplemented
    
    def save(self, file: str):
        raise NotImplementedError
    
    @staticmethod
    def load(file: str):
        raise NotImplementedError
    