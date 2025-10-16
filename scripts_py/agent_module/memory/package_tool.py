import pandas as pd
import numpy as np

from llama_index.core.schema import Document as LlamaIndex_Document
from langchain_core.documents import Document as Langchain_Document

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


class PackageTool(object):
    
    @staticmethod
    def langchainDoc_to_llamaindexDoc(doc: Langchain_Document) -> LlamaIndex_Document:
        return LlamaIndex_Document(text=doc.page_content, metadata=doc.metadata, id_=doc.id)

    @staticmethod
    def llamaindexDoc_to_langchainDoc(doc: LlamaIndex_Document) -> Langchain_Document:
        return Langchain_Document(page_content=doc.text, metadata=doc.metadata, id=doc.doc_id)

    @staticmethod
    def create_LlamaIndex_Textnode(text: str, metadata: dict, node_id: str) -> TextNode:
        return TextNode(text=text, metadata=metadata, id_=node_id)
    
    @staticmethod
    def split_doc(docs: list[LlamaIndex_Document], chunk_size: int, chunk_overlap: int, separator=" ") -> list[dict]:
        nodes: list[TextNode] = SentenceSplitter(
            separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).get_nodes_from_documents(docs)
        chunk_list = []
        for node in nodes:
            chunk_list.append({
                'text': node.text,
                'id': node.id_,
                'metadata': node.metadata
            })
            node.text
        return chunk_list

    