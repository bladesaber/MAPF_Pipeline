import os
import uuid
from typing import Callable
import yaml

import numpy as np

from langchain_core.documents import Document as Langchain_Document
from langchain_community.retrievers import BM25Retriever


class LangChain_BM25SpareStore(object):
    @staticmethod
    def default_preprocess_func(text: str) -> list[str]:
        return text.split()

    def __init__(self, bm25_params={}, preprocess_func:Callable = default_preprocess_func, load_file: str=None):
        self.bm25_params = bm25_params
        self.preprocess_func = preprocess_func
        self.retriever: BM25Retriever
        
        if (load_file is not None) and os.path.exists(load_file):
            assert load_file.endswith('.yaml')
            with open(load_file, 'r') as f:
                json_content = yaml.safe_load(f.read())
            docs = []
            for content in json_content:
                docs.append(Langchain_Document(
                    page_content=content['content'],
                    metadata=content['metadata'],
                    id=content['id']
                ))
            self.add_docs(docs)
    
    def add_docs(self, docs: list[dict|Langchain_Document]):
        if isinstance(docs[0], dict):
            docs = [
                Langchain_Document(
                    page_content=doc['content'], 
                    metadata=doc['metadata'], 
                    id=doc.get('id', str(uuid.uuid4()))
                )  for doc in docs
            ]
        
        docs.extend(self.retriever.docs)        
        self.retriever = BM25Retriever.from_documents(
            docs, bm25_params=self.bm25_params, 
            preprocess_func=self.preprocess_func
        )
    
    def delete_docs(self, ids: list[str]):
        raise NotImplementedError
    
    def save(self, file: str):
        assert file.endswith('.yaml')
        json_content =[{
            'id': doc.id,
            'content': doc.page_content, 
            'metadata': doc.metadata
        } for doc in self.retriever.docs]
        with open(file, 'w') as f:
            yaml.dump(json_content, f, sort_keys=False, allow_unicode=True)


class LangChain_KeyWord_BM25Store(LangChain_BM25SpareStore):
    @staticmethod
    def extract_keyword_func():
        # todo non-finish
        raise NotImplementedError

    def __init__(self, bm25_params={}, preprocess_func = extract_keyword_func, load_file = None):
        super().__init__(bm25_params, preprocess_func, load_file)
    