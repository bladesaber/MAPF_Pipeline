import numpy as np
import pandas as pd
import openai
from typing import Literal, Union
import yaml

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion


class MessageItem(object):
    roles = ['system', 'user', 'assistant']
    
    def __init__(self, role, content):
        self.role = role
        self.content = content
    
    def to_dict(self) -> dict:
        return {'role': self.role, 'content': self.content}

    @staticmethod
    def to_msg(msg: dict):
        return MessageItem(role=msg['role'], content=msg['content'])


class OpenaiConfig(object):
    def __init__(self, api_key, model_name='gpt-4o', base_url='https://api.openai.com/v1'):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url


class OpenaiLlm(object):
    def __init__(self, config: OpenaiConfig, setup=True):
        self.config = config
        self.client: openai.OpenAI
        if setup:
            self.client = self.setup_client(self.config)
    
    @staticmethod
    def setup_client(config: OpenaiConfig) -> openai.OpenAI:
        return openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
    
    def chat_completion(
        self,
        msgs: list[Union[MessageItem, dict]], 
        n=1, temperature=0.1, max_tokens=1000,
        frequency_penalty=0, presence_penalty=0, logprobs=False,
        modalities: list[Literal["text", "audio"]] = ['text'],
        stream=False, client: openai.OpenAI | None = None,
        **kwargs,
    ) -> ChatCompletion:
        if client is None:
            if self.client is None:
                client = self.setup_client(self.config)
            else:
                client = self.client
        
        if isinstance(msgs[0], MessageItem):
            msgs = [msg.to_dict() for msg in msgs]
        else:
            msgs = msgs

        response: ChatCompletion = client.chat.completions.create(
            model=self.config.model_name,
            messages=msgs,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            modalities=modalities,
            stream=stream,
            **kwargs,
        )
        
        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        
        return response_message, metadata

    def completion(
        self,
        message: str, 
        temperature=0.1, max_tokens=1000, n=1,
        frequency_penalty=0, presence_penalty=0, logprobs=0,
        stream=False, client: openai.OpenAI | None = None,
    ) -> Completion:
        if client is None:
            if self.client is None:
                client = self.setup_client(self.config)
            else:
                client = self.client
        
        response: Completion = client.completions.create(
            model=self.config.model_name,
            prompt=message,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            stream=stream,
        )        
        
        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        
        return response_message, metadata

    def embedding(self, model_id:str, doc_list:list[str], client: openai.OpenAI | None = None) -> np.ndarray:
        if client is None:
            client = self.setup_client(self.config)
        responses = client.embeddings.create(
            input=doc_list,
            model=model_id
        )
        return np.array([data.embedding for data in responses.data])
    
    def model_list(self, client: openai.OpenAI | None = None):
        if client is None:
            client = self.setup_client(self.config)
        return client.models.list()


# llm = OpenaiLlm(
#     config=OpenaiConfig(
#         api_key='123456',
#         model_name='Qwen3_Embedding',
#         base_url='http://192.168.110.79:8888/v1'
#     ),
#     setup=True
# )
# # print(llm.model_list(client=llm.client))
# res = llm.embedding(
#     model_id='Qwen3_Embedding',
#     client=llm.client,
#     doc_list=[
#         'OpenFoam is CFD software which used widely.',
#         'My name is John.'
#     ]
# )
# print(res)