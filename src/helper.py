from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


def load_pdf_files(data_path):
    docs = []
    import os
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs


def filter_to_min_docs(docs:List[Document])-> List[Document]:
    minimal_docs:List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    text_chuncks = text_splitter.split_documents(minimal_docs)
    return text_chuncks


def download_embeddings():
    model = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model
    )
    return embeddings

embeddings = download_embeddings()