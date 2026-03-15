from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class Chunker:
    """
    Splits Documents into smaller overlapping chunks.

    """

    def __init__(
        self,
        chunk_size   : int = 500,
        chunk_overlap: int = 50
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



