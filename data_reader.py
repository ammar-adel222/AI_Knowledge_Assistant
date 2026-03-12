import os
import re
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader







class data_extractor:
    """
    reads any of the 3 data sources and arranges them into a list of documents
    """

    def extract_pdf(self, file_path: str) -> List[Document]:

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



    def extract_spreadsheet(self, file_path: str) -> List[Document]:

        # ── Checks ────────────────────────────────
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext not in [".csv", ".xlsx", ".xls"]:
            raise ValueError(f"Unsupported format: {ext}")

        print(f"\n Sheet Extractor")
        print(f"   File    : {os.path.basename(file_path)}")

        # ── Load ──────────────────────────────────
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df = df.fillna("N/A")

        print(f"   Rows    : {len(df)}")
        print(f"   Columns : {list(df.columns)}")

        # ── Each row → One Document ───────────────
        documents = []

        for idx, row in df.iterrows():

            row_text = " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
            )

            doc = Document(
                page_content = row_text,
                metadata     = {
                    "source_type" : "spreadsheet",
                    "file_name"   : os.path.basename(file_path),
                    "row_number"  : idx + 1,
                }
            )
            documents.append(doc)

        print(f" Done  : {len(documents)} documents created")
        return documents









