import os
import uuid
from typing import Literal

import numpy as np
import faiss

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document as Langchain_Document
from langchain_core.vectorstores import VectorStoreRetriever


class LangChain_FaissVectorStore(object):
    def __init__(self, embedding_config: dict, load_folder: str=None, index_name: str=None):
        if embedding_config['model_type'] == "openai":
            self.embedding_model = OpenAIEmbeddings(
                model=embedding_config['model_name'],
                api_key=embedding_config['api_key'],
                base_url=embedding_config['base_url']
            )
        elif embedding_config['model_type'] == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_config['model_name_or_path'],
                model_kwargs={
                    "device": embedding_config.get("device", "cpu")
                },
                encode_kwargs={
                    "normalize_embeddings": embedding_config.get('normalize_embeddings', False)
                }
            )
        else:
            raise ValueError(f"[ERROR]: Model {embedding_config['model_type']} not supported")
        
        if (load_folder is not None) and (os.path.exists(load_folder)):
            self.vector_store = FAISS.load_local(
                load_folder, 
                embeddings=self.embedding_model, 
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            self.docs: dict[str, Langchain_Document] = {}  # todo non-finish
        else:
            self.vector_store = FAISS(
                embedding_function=self.embedding_model,
                index=faiss.IndexFlatL2(embedding_config['embedding_dim']),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.docs: dict[str, Langchain_Document] = {}
        
        self.retriever: VectorStoreRetriever
        
    def encode_query(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text)).reshape((1, -1))

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.array(self.embedding_model.embed_documents(texts))

    def cosin_score(self, query_embedding: np.ndarray, docs_embedding: np.ndarray, norm=False) -> np.ndarray:
        if norm:
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            docs_embedding = docs_embedding / np.linalg.norm(docs_embedding, axis=1, keepdims=True)
        similarity_matrix = np.dot(query_embedding, docs_embedding.T)
        return similarity_matrix
    
    def add_docs(self, docs: list[dict|Langchain_Document]):
        """
        Args:
            docs: list[Document], the documents to add to the vector store
        Returns:
            list[str], the ids of the documents added to the vector store
        """
        if isinstance(docs[0], dict):
            docs = [
                Langchain_Document(
                    page_content=doc['content'], 
                    metadata=doc['metadata'], 
                    id=doc.get('id', str(uuid.uuid4()))
                )  for doc in docs
            ]
        id_list = [doc.id for doc in docs]
        self.vector_store.add_documents(documents=docs, ids=id_list)
        for doc in docs:
            self.docs[doc.id] = doc

    def delete_docs(self, ids: list[str]) -> bool:
        return self.vector_store.delete(ids=ids)
    
    def search(self, query: str, k: int, metadata_filter: dict=None, fetch_k=100) -> list[tuple[Langchain_Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k, filter=metadata_filter, fetch_k=fetch_k)
        
    def save(self, folder_path: str, index_name: str):
        self.vector_store.save_local(folder_path=folder_path, index_name=index_name)
    
    def assemble_strategy_retriever(
        self,
        search_type: Literal['mmr', 'similarity', 'similarity_score_threshold'] = "mmr",
        k: int = 6,
        score_threshold: float = 0.25,
        fetch_k: int = 100,
        lambda_mult: float = 0.5,
    ) -> VectorStoreRetriever:
        '''
        Args:
            search_type:
                mmr: Maximal Marginal Relevance, Retrieve more documents with higher diversity
                similarity_score_threshold: Only retrieve documents that have a relevance score above a certain threshold
        '''
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )
        return self.retriever

    def invoke(self, query: str, metadata_filter: dict, with_instruct: bool=False):
        assert self.retriever is not None
        # todo how to get score
        # todo need instruct
        self.retriever.invoke(query, filter=metadata_filter)



