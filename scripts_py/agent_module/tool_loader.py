import os
from typing import Literal
import json

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.documents import Document


class PdfLoader(object):
    def __init__(self, file: str, mode:Literal['page', 'single'] ='page', extract_images=False):
        assert file.endswith('.pdf')
        self.loader = PyPDFium2Loader(
            file,
            mode=mode,
            images_inner_format="text",
            images_parser=RapidOCRBlobParser(),
            extract_images=extract_images
        )
        self.docs: list[Document] = self.loader.load()

    def __len__(self):
        return len(self.docs)
