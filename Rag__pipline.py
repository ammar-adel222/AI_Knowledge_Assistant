import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
from model_configration  import Config

from typing import List
import os



class Chunker:
    """
    Splits Documents into smaller overlapping chunks.

    """

    def __init__(
        self,
        chunk_size : int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size      = chunk_size,
            chunk_overlap   = chunk_overlap,
            length_function = len,
            separators      = ["\n\n", "\n", " ", ""],
        )


    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        return chunks



class Embedder:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name    = Config.EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        )
        print(f"✅ Embedding model loaded!")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return self.embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

class VectorStore:

    def __init__(self):
        self.store      = None
        self.store_path = Config.VECTOR_STORE_PATH

    def build(self, chunks: List[Document], embedder: Embedder) -> None:

        self.store = FAISS.from_documents(
            documents = chunks,
            embedding = embedder.get_embeddings()
        )

    def save(self) -> None:
        if self.store is None:
            raise ValueError(" Nothing to save! Build it first.")
        os.makedirs(self.store_path, exist_ok=True)
        self.store.save_local(self.store_path)
        print(f" Saved to: {self.store_path}")


    def load(self, embedder: Embedder) -> None:
        if not os.path.exists(self.store_path):
            raise FileNotFoundError(f" No saved store at: {self.store_path}")
        self.store = FAISS.load_local(
            folder_path                     = self.store_path,
            embeddings                      = embedder.get_embeddings(),
            allow_dangerous_deserialization = True
        )


    def is_saved(self) -> bool:
        return os.path.exists(os.path.join(self.store_path, "index.faiss"))


    def search(self, query: str, k: int = 4) -> List[Document]:
        if self.store is None:
            raise ValueError(" Vector store empty! Build or load first.")

        results = self.store.similarity_search(query=query, k=k)

        for i, doc in enumerate(results):
            print(f"   [{i+1}] {doc.metadata.get('source_type','?')} "
                  f"| {doc.metadata.get('file_name') or doc.metadata.get('video_id','?')}")
        return results

