import os
import re
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)







class PDFExtractor:
    """
    Reads a PDF file.
    Each page becomes one LangChain Document.
    """

    def extract(self, file_path: str) -> List[Document]:

        # Checks
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF file: {file_path}")

        print(f"\n PDF Extractor")
        print(f"   File  : {os.path.basename(file_path)}")

        #  Extract
        loader    = PyPDFLoader(file_path)
        documents = loader.load()

        # metadata
        for doc in documents:
            doc.metadata.update({
                "source_type" : "pdf",
                "file_name"   : os.path.basename(file_path),
            })

        print(f"   Pages : {len(documents)}")
        print(f" Done  : {len(documents)} documents created")
        return documents

